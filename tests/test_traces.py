"""Selected-identity witnesses checked against an occurrence-event oracle."""
import itertools
import json
import unittest
from collections import Counter

from batchscope.core import InvalidPlan, MAX_TRACE_IDS, analyze


def config(n=5, mode='distributed_map', ranks=2, workers=1, batch=2, drop=False,
           sampler=False):
    p = dict(ids=[str(i) for i in range(n)], mode=mode, ranks=ranks,
             workers=workers, batch_size=batch, loader_drop_last=drop)
    if mode == 'distributed_map':
        p['sampler_drop_last'] = sampler
    return p


def event_oracle(p):
    # Enumerate occurrences and their ownership before counting retained events.
    events = []
    ids = p['ids']
    n = len(ids)
    removed = set()
    if p['mode'] == 'distributed_map':
        total = n
        if p['sampler_drop_last']:
            total -= n % p['ranks']
            removed.update(ids[total:])
        elif n % p['ranks']:
            total += p['ranks'] - n % p['ranks']
        for i in range(total):
            events.append((i % p['ranks'], None, ids[i % n]))
    elif p['mode'] == 'iterable_strided':
        for i, x in enumerate(ids):
            owner = i % (p['ranks'] * p['workers'])
            events.append((owner // p['workers'], owner % p['workers'], x))
    else:
        for x in ids:
            for rank in range(p['ranks']):
                for worker in range(p['workers']):
                    events.append((rank, worker, x))
    lengths = Counter((r, w) for r, w, x in events)
    cursors = Counter()
    counts = {}
    for r, w, x in events:
        key = (r, w)
        keep = lengths[key]
        if p['loader_drop_last']:
            keep -= keep % p['batch_size']
        row = counts.setdefault((x, r, w), [0, 0, 0])
        row[0] += 1
        row[1 if cursors[key] < keep else 2] += 1
        cursors[key] += 1
    return [dict(id=x, sampler_removed=int(x in removed), consumers=[
        dict(rank=r, worker=w, assigned=v[0], emitted=v[1], loader_dropped=v[2])
        for (y, r, w), v in sorted(counts.items(), key=lambda item: (item[0][1], item[0][2] or 0))
        if y == x]) for x in p['trace_ids']]


class Traces(unittest.TestCase):
    def test_small_event_oracle(self):
        checked = 0
        for mode, n, ranks, workers, batch, drop in itertools.product(
                ('distributed_map', 'iterable_strided', 'iterable_replicated'),
                range(9), range(1, 4), range(1, 4), range(1, 6), (False, True)):
            for sampler in ((False, True) if mode == 'distributed_map' else (False,)):
                p = config(n, mode, ranks, workers, batch, drop, sampler)
                p['trace_ids'] = list(reversed(p['ids']))
                report = analyze(p)
                self.assertEqual(report['traces'], event_oracle(p), p)
                totals = {row['id']: row for row in report['ids']}
                for trace in report['traces']:
                    for name in ('assigned', 'emitted', 'loader_dropped'):
                        self.assertEqual(sum(c[name] for c in trace['consumers']),
                                         totals[trace['id']][name])
                checked += 1
        self.assertEqual(checked, 3240)

    def test_default_unchanged_and_empty_opt_in(self):
        p = config()
        plain = analyze(p)
        self.assertNotIn('traces', plain)
        p['trace_ids'] = []
        traced = analyze(p)
        self.assertEqual(traced.pop('traces'), [])
        self.assertEqual(traced, plain)

    def test_padding_and_tail_same_id(self):
        p = config(n=1, ranks=3, batch=2, drop=True)
        p['trace_ids'] = ['0']
        rows = analyze(p)['traces'][0]['consumers']
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(c['assigned'] == c['loader_dropped'] == 1 for c in rows))
        self.assertTrue(all(c['worker'] is None and c['emitted'] == 0 for c in rows))

    def test_removed_id_has_no_consumer(self):
        p = config(sampler=True)
        p['trace_ids'] = ['4']
        self.assertEqual(analyze(p)['traces'], [dict(id='4', sampler_removed=1, consumers=[])])

    def test_invalid_selection(self):
        for selected in (None, False, {}, '0', [0], [[]], ['absent'], ['0', '0']):
            with self.subTest(selected=selected), self.assertRaises(InvalidPlan):
                analyze(dict(config(), trace_ids=selected))
        p = config(MAX_TRACE_IDS + 1)
        with self.assertRaises(InvalidPlan):
            analyze(dict(p, trace_ids=p['ids']))
        p = config(MAX_TRACE_IDS)
        self.assertEqual(len(analyze(dict(p, trace_ids=p['ids']))['traces']), MAX_TRACE_IDS)

    def test_maximum_consumers_sparse_trace(self):
        p = config(2, mode='iterable_replicated', ranks=32, workers=32, batch=3, drop=True)
        p['trace_ids'] = ['1']
        rows = analyze(p)['traces'][0]['consumers']
        self.assertEqual(len(rows), 1024)
        self.assertEqual([(r['rank'], r['worker']) for r in rows],
                         list(itertools.product(range(32), repeat=2)))
        self.assertTrue(all((r['assigned'], r['emitted'], r['loader_dropped']) == (1, 0, 1)
                            for r in rows))

    def test_subset_unicode_and_json_roundtrip(self):
        p = config(4, mode='iterable_replicated', workers=2)
        p['ids'] = ['é', 'e\u0301', '😀', 'a']
        p['trace_ids'] = ['😀', 'é']
        got = analyze(p)
        self.assertEqual(got['traces'], event_oracle(p))
        self.assertEqual(json.loads(json.dumps(got)), got)
        self.assertEqual([x['id'] for x in got['traces']], p['trace_ids'])
