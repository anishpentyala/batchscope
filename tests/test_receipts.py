import copy
import io
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from batchscope import analyze, audit_receipts, InvalidPlan
from test_core import plan


def independent_events(p):
    # Dispatch a sequential stream before batching. Does not call production
    # model code, use trace output, or use its strided slices.
    queues = {(r, None if p['mode'] == 'distributed_map' else w): []
              for r in range(p['ranks'])
              for w in (range(1) if p['mode'] == 'distributed_map' else range(p['workers']))}
    stream = list(p['ids'])
    if p['mode'] == 'distributed_map':
        while len(stream) % p['ranks']:
            if p['sampler_drop_last']:
                stream.pop()
            else:
                stream.append(p['ids'][(len(stream) - len(p['ids'])) % len(p['ids'])])
    if p['mode'] == 'iterable_replicated':
        for key in queues:
            queues[key].extend(stream)
    else:
        keys = list(queues)
        for offset, identity in enumerate(stream):
            queues[keys[offset % len(keys)]].append(identity)
    events = []
    for (rank, worker), queue in queues.items():
        pending = []
        for identity in queue:
            pending.append(identity)
            if len(pending) == p['batch_size']:
                events.extend(dict(id=x, rank=rank, worker=worker) for x in pending)
                pending = []
        if not p['loader_drop_last']:
            events.extend(dict(id=x, rank=rank, worker=worker) for x in pending)
    return events


class Receipts(unittest.TestCase):
    def test_exhaustive_complete_and_removed_occurrence(self):
        count = 0
        for mode, n, ranks, workers, batch, drop, sampler in itertools.product(
                ('distributed_map', 'iterable_strided', 'iterable_replicated'),
                range(7), range(1, 4), range(1, 3), range(1, 4), (False, True), (False, True)):
            p = plan(n, mode=mode, ranks=ranks, workers=workers, batch_size=batch,
                     loader_drop_last=drop, sampler_drop_last=sampler)
            events = independent_events(p)
            original = copy.deepcopy((p, events))
            r = audit_receipts(p, events)
            self.assertEqual(r['status'], 'MATCHED_RECEIPTS')
            self.assertEqual(r['expected_occurrences'], len(events))
            self.assertEqual(r['differences'], [])
            self.assertEqual(r['planned_coverage_status'], analyze(p)['status'])
            self.assertEqual((p, events), original)
            self.assertEqual(audit_receipts(p, list(reversed(events))), r)
            if events:
                r = audit_receipts(p, events[1:])
                self.assertEqual(r['missing_occurrences'], 1)
                self.assertEqual(r['extra_occurrences'], 0)
                r = audit_receipts(p, events + events[:1])
                self.assertEqual(r['extra_occurrences'], 1)
                self.assertEqual(r['missing_occurrences'], 0)
            count += 1
        self.assertEqual(count, 1512)

    def test_equal_total_swapped_consumers(self):
        p = plan(4)
        events = independent_events(p)
        events[0]['rank'] = 1
        r = audit_receipts(p, events)
        self.assertEqual(r['missing_occurrences'], 1)
        self.assertEqual(r['extra_occurrences'], 1)
        self.assertEqual(r['expected_occurrences'], r['observed_occurrences'])
        self.assertEqual(r['unknown_id_occurrences'], 0)
        self.assertEqual(r['status'], 'RECEIPT_MISMATCH')

    def test_worker_swap_unknown_and_dropped(self):
        p = plan(5, mode='iterable_strided', ranks=1, workers=2,
                 batch_size=2, loader_drop_last=True)
        events = independent_events(p)
        events[0]['worker'] = 1
        events.append(dict(id='unlisted', rank=0, worker=1))
        events.append(dict(id='id-4', rank=0, worker=0)) # known dropped tail
        r = audit_receipts(p, events)
        self.assertEqual(r['missing_occurrences'], 1)
        self.assertEqual(r['extra_occurrences'], 3)
        self.assertEqual(r['unknown_id_occurrences'], 1)
        self.assertEqual(r['differences'][-1]['id'], 'unlisted')

    def test_match_does_not_claim_coverage(self):
        for p in (plan(0), plan(5), plan(4, mode='iterable_replicated', workers=2)):
            r = audit_receipts(p, independent_events(p))
            self.assertEqual(r['status'], 'MATCHED_RECEIPTS')
            self.assertNotEqual(r['planned_coverage_status'], 'EXACTLY_ONCE')

    def test_strict_inputs(self):
        p = plan(4)
        event = dict(id='id-0', rank=0, worker=None)
        invalid = [None, {}, iter([]), [None], [dict(event, extra=1)],
                   [dict(event, id='')], [dict(event, id=5)], [dict(event, id='x'*257)],
                   [dict(event, rank=True)], [dict(event, rank=-1)],
                   [dict(event, rank=2)], [dict(event, worker=0)], [dict(id='x', rank=0)]]
        for events in invalid:
            with self.subTest(events=events), self.assertRaises(InvalidPlan):
                audit_receipts(p, events)
        with patch('batchscope.receipts.MAX_ASSIGNMENTS', 1):
            with self.assertRaises(InvalidPlan):audit_receipts(p, [event, event])
        for worker in (True, None, -1, 2, 0.0):
            with self.assertRaises(InvalidPlan):
                audit_receipts(plan(1, mode='iterable_strided', workers=2),
                               [dict(event, worker=worker)])
        with self.assertRaises(InvalidPlan):audit_receipts({}, [])

    def test_cli_and_delivery(self):
        from batchscope.__main__ import main
        with tempfile.TemporaryDirectory() as d:
            p, e = Path(d)/'plan.json', Path(d)/'events.json'
            model = plan(4)
            p.write_text(json.dumps(model));e.write_text(json.dumps(independent_events(model)))
            def run(*args):
                return subprocess.run([sys.executable, '-m', 'batchscope', *args],
                                      capture_output=True, timeout=5)
            args = ['--receipts', str(p), str(e)]
            result = run(*args)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(json.loads(result.stdout)['status'], 'MATCHED_RECEIPTS')
            e.write_text('[]')
            self.assertEqual(run(*args).returncode, 1)
            for value in ('{}', '[{"id":"a","id":"b"}]', '[NaN]'):
                e.write_text(value)
                result = run(*args)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, b'')
            e.write_text('[]')
            class Short(io.StringIO):
                def write(self, text):return super().write(text[:7])
            target = Short()
            with patch.object(sys, 'argv', ['batchscope'] + args), \
                    patch.object(sys, 'stdout', target), patch.object(sys, 'stderr', io.StringIO()):
                self.assertEqual(main(), 2)
                self.assertIs(sys.stdout, target)
            if Path('/dev/full').exists():
                with open('/dev/full', 'wb') as output:
                    result = subprocess.run([sys.executable, '-m', 'batchscope', *args],
                                            stdout=output, stderr=subprocess.PIPE, timeout=5)
                self.assertEqual(result.returncode, 2)
            self.assertEqual(run('--receipts', str(p)).returncode, 2)
