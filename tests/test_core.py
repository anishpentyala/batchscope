import copy
import itertools
import unittest
from collections import Counter
from batchscope import analyze, InvalidPlan


def plan(n=7, **kw):
    p = dict(ids=[f'id-{i}' for i in range(n)], mode='distributed_map', ranks=2,
             workers=1, batch_size=2, sampler_drop_last=False, loader_drop_last=False)
    p.update(kw)
    if p['mode'] != 'distributed_map':
        p.pop('sampler_drop_last', None)
    return p


def oracle(p):
    # Event-wise reference implementation: dispatch occurrences, then consume batches.
    queues = {}
    ids = p['ids']
    r, w, b = p['ranks'], p['workers'], p['batch_size']
    if p['mode'] == 'distributed_map':
        for rank in range(r):
            queues[rank, None] = []
        stream = list(ids)
        while len(stream) % r:
            if p['sampler_drop_last']:
                stream.pop()
            else:
                stream.append(ids[(len(stream) - len(ids)) % len(ids)])
        for i, item in enumerate(stream):
            queues[i % r, None].append(item)
    else:
        for rank in range(r):
            for worker in range(w):
                queues[rank, worker] = []
        if p['mode'] == 'iterable_replicated':
            for queue in queues.values():
                queue.extend(ids)
        else:
            keys = list(queues)
            for i, item in enumerate(ids):
                queues[keys[i % len(keys)]].append(item)
    output, tails = Counter(), Counter()
    steps = [0] * r
    for (rank, _), queue in queues.items():
        pending = []
        for item in queue:
            pending.append(item)
            if len(pending) == b:
                output.update(pending)
                pending = []
                steps[rank] += 1
        if pending:
            if p['loader_drop_last']:
                tails.update(pending)
            else:
                output.update(pending)
                steps[rank] += 1
    return output, tails, steps


class Core(unittest.TestCase):
    def test_small_exhaustive(self):
        for n, ranks, workers, batch, drop, sampler, mode in itertools.product(
                range(13), range(1, 5), range(1, 4), range(1, 6), (False, True),
                (False, True), ('distributed_map', 'iterable_strided', 'iterable_replicated')):
            p = plan(n, ranks=ranks, workers=workers, batch_size=batch,
                     loader_drop_last=drop, sampler_drop_last=sampler, mode=mode)
            with self.subTest(n=n, ranks=ranks, workers=workers, batch=batch,
                              drop=drop, sampler=sampler, mode=mode):
                out, tails, steps = oracle(p)
                got = analyze(p)
                self.assertEqual({x['id']: x['emitted'] for x in got['ids']},
                                 {x: out[x] for x in p['ids']})
                self.assertEqual({x['id']: x['loader_dropped'] for x in got['ids']},
                                 {x: tails[x] for x in p['ids']})
                self.assertEqual(got['rank_batches'], steps)
                self.assertEqual(got['distinct_emitted'], len(out))
                self.assertEqual(got['status'] == 'EXACTLY_ONCE',
                                 bool(p['ids']) and all(out[x] == 1 for x in p['ids']))

    def test_padding_can_be_removed_by_loader_tail(self):
        got = analyze(plan(5, loader_drop_last=True))
        self.assertEqual(got['sampler_padding_count'], 1)
        self.assertEqual(got['loader_dropped_occurrences'], 2)
        self.assertEqual(got['emitted_occurrences'], 4)
        self.assertEqual(got['repeated_id_count'], 0)
        self.assertEqual(got['missing_count'], 1)

    def test_fragmented_worker_tails(self):
        got = analyze(plan(10, mode='iterable_strided', ranks=1, workers=3,
                           batch_size=4, loader_drop_last=True))
        self.assertEqual(got['emitted_occurrences'], 4)
        self.assertEqual(got['loader_dropped_occurrences'], 6)

    def test_map_workers_do_not_change_assignment(self):
        a = analyze(plan(11, workers=1))
        b = analyze(plan(11, workers=4))
        self.assertEqual(a['ids'], b['ids'])
        self.assertEqual(a['rank_batches'], b['rank_batches'])

    def test_does_not_mutate(self):
        p = plan(); original = copy.deepcopy(p); analyze(p)
        self.assertEqual(p, original)

    def test_invalid(self):
        cases = [None, [], {}, plan(ids=['a','a']), plan(ids=[True]),
                 plan(mode='other'), plan(mode=[]), plan(ranks=True), plan(workers=0),
                 plan(batch_size=0), plan(loader_drop_last=1),
                 plan(sampler_drop_last=0), plan(unknown=3), plan(ranks=1024,workers=2)]
        for p in cases:
            with self.subTest(p=p), self.assertRaises(InvalidPlan):analyze(p)

    def test_non_map_sampler_rejected(self):
        p=plan(mode='iterable_strided');p['sampler_drop_last']=False
        with self.assertRaises(InvalidPlan):analyze(p)

    def test_empty_is_explicit(self):
        got=analyze(plan(0));self.assertEqual(got['warnings'],['EMPTY_UNIVERSE'])

    def test_uneven_steps(self):
        got=analyze(plan(5,mode='iterable_strided',ranks=2,batch_size=2))
        self.assertIn('UNEQUAL_RANK_BATCH_COUNTS',got['warnings'])
