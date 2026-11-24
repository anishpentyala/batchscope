"""Optional real-framework controls. No claims about other versions/configurations."""
import itertools
import unittest
from collections import Counter
from batchscope import analyze
from test_core import plan

try:
    import torch
    from torch.utils.data import DataLoader, DistributedSampler, IterableDataset, get_worker_info
except ImportError:
    torch = None
    IterableDataset = object


class DeclaredStream(IterableDataset):
    def __init__(self, ids, rank, ranks, mode):
        self.ids, self.rank, self.ranks, self.mode = ids, rank, ranks, mode

    def __iter__(self):
        info = get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        if self.mode == 'iterable_replicated':
            return iter(self.ids)
        return iter(self.ids[self.rank * workers + worker::self.ranks * workers])


@unittest.skipIf(torch is None, 'optional real PyTorch not installed')
class TorchParity(unittest.TestCase):
    def compare(self, p, actual, steps):
        got = analyze(p)
        self.assertEqual({x['id']: x['emitted'] for x in got['ids']},
                         {x: actual[x] for x in p['ids']})
        self.assertEqual(got['rank_batches'], steps)
        self.assertEqual(got['emitted_occurrences'], sum(actual.values()))

    def test_unshuffled_map(self):
        for n, ranks, batch, sampler_drop, loader_drop, workers in itertools.product(
                (0, 1, 5, 8), (1, 2, 3), (1, 3), (False, True), (False, True), (0,)):
            p = plan(n, ranks=ranks, batch_size=batch, workers=max(1, workers),
                     sampler_drop_last=sampler_drop, loader_drop_last=loader_drop)
            actual, steps = Counter(), []
            with self.subTest(n=n, ranks=ranks, batch=batch, sampler=sampler_drop, loader=loader_drop):
                for rank in range(ranks):
                    sampler = DistributedSampler(p['ids'], num_replicas=ranks, rank=rank,
                                                 shuffle=False, drop_last=sampler_drop)
                    loader = DataLoader(p['ids'], sampler=sampler, batch_size=batch,
                                        drop_last=loader_drop, num_workers=workers)
                    count = 0
                    for values in loader:
                        actual.update(values)
                        count += 1
                    steps.append(count)
                self.compare(p, actual, steps)

    def test_two_actual_map_workers(self):
        p = plan(11, ranks=2, workers=2, batch_size=3, loader_drop_last=True)
        actual, steps = Counter(), []
        for rank in range(p['ranks']):
            sampler = DistributedSampler(p['ids'], num_replicas=2, rank=rank,
                                         shuffle=False, drop_last=False)
            loader = DataLoader(p['ids'], sampler=sampler, batch_size=3,
                                drop_last=True, num_workers=2)
            count = 0
            for values in loader:
                actual.update(values); count += 1
            steps.append(count)
        self.compare(p, actual, steps)

    def test_declared_iterable_sources(self):
        # A few real multiprocess cases exercise per-worker tails without a huge spawn matrix.
        for mode, n, workers, drop in itertools.product(
                ('iterable_strided', 'iterable_replicated'), (0, 5, 10), (0, 2), (False, True)):
            p = plan(n, mode=mode, ranks=2, workers=max(1, workers), batch_size=3,
                     loader_drop_last=drop)
            actual, steps = Counter(), []
            with self.subTest(mode=mode, n=n, workers=workers, drop=drop):
                for rank in range(2):
                    source = DeclaredStream(p['ids'], rank, 2, mode)
                    loader = DataLoader(source, batch_size=3, drop_last=drop, num_workers=workers)
                    count = 0
                    for values in loader:
                        actual.update(values); count += 1
                    steps.append(count)
                self.compare(p, actual, steps)
