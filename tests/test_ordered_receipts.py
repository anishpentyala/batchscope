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
from batchscope import audit_receipts, InvalidPlan
from test_core import plan
from test_receipts import independent_events


class OrderedReceipts(unittest.TestCase):
    def test_same_counts_wrong_order(self):
        p = plan(4, ranks=1)
        events = independent_events(p)
        events[0], events[1] = events[1], events[0]
        self.assertEqual(audit_receipts(p, events)['status'], 'MATCHED_RECEIPTS')
        got = audit_receipts(p, events, check_order=True)
        self.assertEqual(got['status'], 'ORDERED_RECEIPT_MISMATCH')
        self.assertEqual(got['differences'], [])
        self.assertEqual(got['order_differences'], [dict(rank=0, worker=None,
            expected_length=4, observed_length=4,
            first_difference=dict(index=0, expected_id='id-0', observed_id='id-1'))])

    def test_interleave_consumers_and_oracle(self):
        count = 0
        for mode, n, r, w, b, drop, sampler in itertools.product(
                ('distributed_map', 'iterable_strided', 'iterable_replicated'),
                range(7), range(1, 4), range(1, 3), range(1, 4), (False, True), (False, True)):
            p = plan(n, mode=mode, ranks=r, workers=w, batch_size=b,
                     loader_drop_last=drop, sampler_drop_last=sampler)
            events = independent_events(p)
            # Round-robin interleave preserving each consumer's local order.
            groups = {}
            for event in events:
                groups.setdefault((event['rank'], event['worker']), []).append(event)
            interleaved = [event for batch in itertools.zip_longest(*reversed(list(groups.values())))
                           for event in batch if event is not None]
            untouched = copy.deepcopy((p, interleaved))
            got = audit_receipts(p, interleaved, check_order=True)
            self.assertEqual(got['status'], 'MATCHED_ORDERED_RECEIPTS')
            self.assertEqual(got['order_differences'], [])
            self.assertEqual((p, interleaved), untouched)
            for key, group in groups.items():
                if len(group) > 1 and group[0]['id'] != group[1]['id']:
                    swapped = list(events)
                    i, j = [i for i, e in enumerate(swapped) if (e['rank'],e['worker'])==key][:2]
                    swapped[i], swapped[j] = swapped[j], swapped[i]
                    got = audit_receipts(p, swapped, check_order=True)
                    self.assertEqual(len(got['order_differences']), 1)
                    self.assertEqual(got['order_differences'][0]['first_difference']['index'], 0)
                    break
            count += 1
        self.assertEqual(count, 1512)

    def test_prefix_missing_extra_and_empty(self):
        p = plan(3, ranks=1)
        events = independent_events(p)
        for receipts, index, expected, observed in (
                (events[:2], 2, 'id-2', None),
                (events + events[:1], 3, None, 'id-0'),
                ([], 0, 'id-0', None),
                ([dict(id='unknown',rank=0,worker=None)], 0, 'id-0','unknown')):
            got = audit_receipts(p, receipts, check_order=True)
            self.assertEqual(got['order_differences'][0]['first_difference'],
                             dict(index=index, expected_id=expected, observed_id=observed))
        for p in (plan(0), plan(3, mode='iterable_strided', workers=2,
                                batch_size=4, loader_drop_last=True)):
            got = audit_receipts(p, [], check_order=True)
            self.assertEqual(got['status'], 'MATCHED_ORDERED_RECEIPTS')
            self.assertNotEqual(got['planned_coverage_status'], 'EXACTLY_ONCE')

    def test_first_difference_not_last_and_consumer_order(self):
        p=plan(6,mode='iterable_replicated',ranks=2,workers=2)
        events=independent_events(p)
        for e in events:e['id']='unknown'
        got=audit_receipts(p,list(reversed(events)),check_order=True)
        self.assertEqual([(x['rank'],x['worker']) for x in got['order_differences']],
                         [(0,0),(0,1),(1,0),(1,1)])
        self.assertTrue(all(x['first_difference']['index']==0 for x in got['order_differences']))
        for value in (1,0,None,'yes'):
            with self.assertRaises(InvalidPlan):audit_receipts(p,[],check_order=value)

    def test_cli_output_failure(self):
        from batchscope.__main__ import main
        with tempfile.TemporaryDirectory() as d:
            p,e=Path(d)/'p.json',Path(d)/'e.json'
            model=plan(3,ranks=1)
            p.write_text(json.dumps(model));e.write_text(json.dumps(independent_events(model)))
            args=['--ordered-receipts',str(p),str(e)]
            def run():return subprocess.run([sys.executable,'-m','batchscope',*args],capture_output=True,timeout=5)
            got=run();self.assertEqual(got.returncode,0)
            self.assertEqual(json.loads(got.stdout)['schema'],'batchscope-ordered-receipts-v1')
            e.write_text('[]');self.assertEqual(run().returncode,1)
            e.write_text('{}');self.assertEqual(run().returncode,2)
            e.write_text('[]')
            class Short(io.StringIO):
                def write(self,text):return super().write(text[:8])
            target=Short()
            with patch.object(sys,'argv',['batchscope']+args),patch.object(sys,'stdout',target),patch.object(sys,'stderr',io.StringIO()):
                self.assertEqual(main(),2);self.assertIs(sys.stdout,target)
            if Path('/dev/full').exists():
                with open('/dev/full','wb') as output:
                    result=subprocess.run([sys.executable,'-m','batchscope',*args],stdout=output,stderr=subprocess.PIPE,timeout=5)
                self.assertEqual(result.returncode,2)
