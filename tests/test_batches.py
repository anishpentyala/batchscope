import copy
import itertools
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from batchscope import InvalidPlan, audit_batches
from test_core import plan
from test_receipts import independent_events


def independent_batches(p):
    groups = {}
    for event in independent_events(p):
        groups.setdefault((event['rank'], event['worker']), []).append(event['id'])
    output = []
    for (rank, worker), ids in groups.items():
        for offset in range(0, len(ids), p['batch_size']):
            output.append({'rank': rank, 'worker': worker,
                           'ids': ids[offset:offset+p['batch_size']]})
    return output


class BatchReceipts(unittest.TestCase):
    def test_flattened_order_match_does_not_hide_boundary_change(self):
        p = plan(3, ranks=1, batch_size=2)
        got = audit_batches(p, [{'rank': 0, 'worker': None, 'ids': ['id-0']},
                                {'rank': 0, 'worker': None, 'ids': ['id-1', 'id-2']}])
        self.assertEqual(got['status'], 'BATCH_RECEIPT_MISMATCH')
        self.assertEqual(got['ordered_receipts']['status'], 'MATCHED_ORDERED_RECEIPTS')
        self.assertEqual(got['boundary_differences'], [{'rank': 0, 'worker': None,
            'expected_batches': 2, 'observed_batches': 2, 'first_difference': {
            'index': 0, 'expected_ids': ['id-0', 'id-1'], 'observed_ids': ['id-0']}}])

    def test_missing_extra_reorder_and_unknown_evidence(self):
        p = plan(5, ranks=1, batch_size=2)
        baseline = independent_batches(p)
        for batches, index, left, right in [
            (baseline[:2], 2, ['id-4'], None),
            (baseline+[baseline[0]], 3, None, ['id-0', 'id-1']),
            ([baseline[1], baseline[0], baseline[2]], 0, ['id-0', 'id-1'], ['id-2', 'id-3']),
            ([{'rank': 0, 'worker': None, 'ids': ['unknown']}]+baseline[1:], 0,
             ['id-0', 'id-1'], ['unknown'])]:
            r = audit_batches(p, batches)
            self.assertEqual(r['status'], 'BATCH_RECEIPT_MISMATCH')
            self.assertEqual(r['boundary_differences'][0]['first_difference'],
                             {'index': index, 'expected_ids': left, 'observed_ids': right})
        self.assertEqual(r['ordered_receipts']['unknown_id_occurrences'], 1)

    def test_empty_plan_no_batches_and_duplicating_plan_can_match(self):
        for p in [plan(0), plan(1, ranks=2), plan(3, mode='iterable_replicated', workers=2)]:
            r = audit_batches(p, independent_batches(p))
            self.assertEqual(r['status'], 'MATCHED_BATCH_RECEIPTS')
            self.assertNotEqual(r['ordered_receipts']['planned_coverage_status'], 'EXACTLY_ONCE')
        r = audit_batches(plan(0), [{'rank': 0, 'worker': None, 'ids': ['alien']}])
        self.assertEqual(r['boundary_differences'][0]['first_difference']['expected_ids'], None)

    def test_exhaustive_small_plans_and_free_consumer_interleaving(self):
        count = 0
        for mode, n, ranks, workers, size, drop, sampler in itertools.product(
                ('distributed_map', 'iterable_strided', 'iterable_replicated'),
                range(7), range(1, 4), range(1, 3), range(1, 4), (False, True), (False, True)):
            p = plan(n, mode=mode, ranks=ranks, workers=workers, batch_size=size,
                     loader_drop_last=drop, sampler_drop_last=sampler)
            batches = independent_batches(p)
            groups = {}
            for batch in batches:
                groups.setdefault((batch['rank'], batch['worker']), []).append(batch)
            interleaved = [b for group in itertools.zip_longest(*reversed(list(groups.values())))
                           for b in group if b is not None]
            untouched = copy.deepcopy((p, interleaved))
            r = audit_batches(p, interleaved)
            self.assertEqual(r['status'], 'MATCHED_BATCH_RECEIPTS')
            self.assertEqual(r['expected_batches'], len(batches))
            self.assertEqual(r['observed_batches'], len(batches))
            self.assertEqual(r['boundary_differences'], [])
            self.assertEqual((p, interleaved), untouched)
            if batches and len(batches[0]['ids']) > 1:
                split = copy.deepcopy(batches)
                first = split.pop(0)
                split[0:0] = [{**first, 'ids': first['ids'][:1]}, {**first, 'ids': first['ids'][1:]}]
                wrong = audit_batches(p, split)
                self.assertEqual(wrong['status'], 'BATCH_RECEIPT_MISMATCH')
                self.assertEqual(wrong['ordered_receipts']['status'], 'MATCHED_ORDERED_RECEIPTS')
                self.assertEqual(wrong['boundary_differences'][0]['first_difference']['index'], 0)
            count += 1
        self.assertEqual(count, 1512)

    def test_invalid_contract_and_preallocation_cap(self):
        p = plan(2, ranks=1)
        good = {'rank': 0, 'worker': None, 'ids': ['id-0']}
        values = [None, {}, [None], [{**good, 'ids': []}], [{**good, 'ids': 'id-0'}],
                  [{**good, 'ids': [True]}], [{**good, 'ids': ['']}],
                  [{**good, 'ids': ['a'*257]}], [{**good, 'rank': True}],
                  [{**good, 'rank': 1}], [{**good, 'worker': 0}],
                  [{**good, 'extra': 0}], [{'ids': ['id-0'], 'rank': 0}]]
        for value in values:
            with self.subTest(value=value), self.assertRaises(InvalidPlan):
                audit_batches(p, value)
        for worker in [None, True, -1, 2]:
            with self.subTest(worker=worker), self.assertRaises(InvalidPlan):
                audit_batches(plan(2, mode='iterable_strided', workers=2), [{**good, 'worker': worker}])
        with patch('batchscope.batches.MAX_ASSIGNMENTS', 2):
            with self.assertRaises(InvalidPlan):
                audit_batches(p, [{**good, 'ids': ['a', 'b', 'c']}])
            with self.assertRaises(InvalidPlan):
                audit_batches(p, [good, good, good])
            with self.assertRaises(InvalidPlan):
                audit_batches(p, [good, {**good, 'ids': ['b', 'c']}])

    def test_reports_do_not_alias_caller_batches(self):
        p = plan(2, ranks=1)
        batches = [{'rank': 0, 'worker': None, 'ids': ['unknown']}]
        r = audit_batches(p, batches)
        batches[0]['ids'][0] = 'changed'
        self.assertEqual(r['boundary_differences'][0]['first_difference']['observed_ids'], ['unknown'])

    def test_cli_match_mismatch_invalid_and_delivery(self):
        from batchscope.__main__ import main
        p = plan(3, ranks=1, batch_size=2)
        with tempfile.TemporaryDirectory() as directory:
            path, batchpath = Path(directory)/'plan.json', Path(directory)/'batches.json'
            path.write_text(json.dumps(p))
            for batches, status in [(independent_batches(p), 0), ([], 1),
                                    ([{'ids': [], 'rank': 0, 'worker': None}], 2)]:
                batchpath.write_text(json.dumps(batches))
                args = ['--batch-receipts', str(path), str(batchpath)]
                r = subprocess.run([sys.executable, '-m', 'batchscope', *args], capture_output=True)
                self.assertEqual(r.returncode, status, r.stderr)
                if status != 2:
                    self.assertEqual(json.loads(r.stdout)['schema'], 'batchscope-batch-receipts-v1')
                else:
                    self.assertEqual(r.stdout, b'')
            class Short(io.StringIO):
                def write(self, value):
                    return 0
            batchpath.write_text('[]')
            with patch('sys.argv', ['batchscope', *args]), patch('sys.stdout', Short()), patch('sys.stderr', io.StringIO()):
                self.assertEqual(main(), 2)
