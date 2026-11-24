import copy
import itertools
import json
import os
import io
from unittest.mock import patch
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from batchscope import InvalidPlan, analyze, compare_plans
from test_core import plan, oracle


class Comparison(unittest.TestCase):
    def test_identity_swap_has_zero_net_but_nonzero_gains_and_losses(self):
        left = plan(5, loader_drop_last=True)
        right = copy.deepcopy(left)
        right['ids'] = list(reversed(right['ids']))
        r = compare_plans(left, right)
        self.assertEqual(r['totals']['emitted_occurrences']['delta'], 0)
        self.assertEqual(r['emitted_gain_occurrences'], 1)
        self.assertEqual(r['emitted_loss_occurrences'], 1)
        self.assertEqual({x['id'] for x in r['ids']}, {'id-0', 'id-4'})
        self.assertEqual(r['status'], 'CHANGED_ACCOUNTING')
        self.assertTrue(r['input_order_changed'])

    def test_oracle_pairs_and_reversal(self):
        checked = 0
        for mode, n, ranks, workers, batch in itertools.product(
                ('distributed_map', 'iterable_strided', 'iterable_replicated'),
                range(8), range(1, 4), range(1, 3), range(1, 5)):
            a = plan(n, mode=mode, ranks=ranks, workers=workers, batch_size=batch)
            b = copy.deepcopy(a); b['ids'].reverse(); b['loader_drop_last'] = True
            old, tails_old, _ = oracle(a); new, tails_new, _ = oracle(b)
            r = compare_plans(a, b); reverse = compare_plans(b, a)
            expected = {x: new[x] - old[x] for x in a['ids']}
            self.assertEqual(r['emitted_gain_occurrences'], sum(max(0, x) for x in expected.values()))
            self.assertEqual(r['emitted_loss_occurrences'], sum(max(0, -x) for x in expected.values()))
            rows = {x['id']: x for x in r['ids']}
            for x in a['ids']:
                self.assertEqual(rows.get(x, {}).get('delta', {}).get('emitted', 0), expected[x])
                self.assertEqual(rows.get(x, {}).get('delta', {}).get('loader_dropped', 0),
                                 tails_new[x] - tails_old[x])
            for key, value in r['totals'].items():
                self.assertEqual(value['delta'], -reverse['totals'][key]['delta'])
            self.assertEqual(a['ids'], [f'id-{i}' for i in range(n)])
            checked += 1
        self.assertEqual(checked, 576)

    def test_same_counts_not_same_order_or_safe_coverage(self):
        a = plan(4); b = copy.deepcopy(a); b['ids'].reverse()
        r = compare_plans(a, b)
        self.assertEqual(r['status'], 'SAME_ACCOUNTING')
        self.assertTrue(r['input_order_changed'])
        a = plan(5)
        r = compare_plans(a, a)
        self.assertEqual(r['status'], 'SAME_ACCOUNTING')
        self.assertEqual(r['before']['status'], 'COVERAGE_CHANGED')
        self.assertEqual(r['ids'], [])

    def test_models_workers_and_ranks(self):
        a = plan(4, ranks=1)
        b = plan(4, mode='iterable_strided', ranks=1)
        self.assertEqual(compare_plans(a, b)['status'], 'CHANGED_ACCOUNTING')
        a = plan(4, mode='iterable_strided', ranks=1)
        b = plan(4, mode='iterable_replicated', ranks=1)
        self.assertEqual(compare_plans(a, b)['status'], 'CHANGED_ACCOUNTING')
        r = compare_plans(plan(4, ranks=1), plan(4, ranks=2))
        self.assertEqual(r['rank_batches'][0]['delta'], -1)
        self.assertEqual(r['rank_batches'][1], dict(rank=1, before=None, after=1, delta=None))
        r = compare_plans(plan(4), plan(4, workers=2))
        self.assertEqual(r['status'], 'CHANGED_ACCOUNTING') # map warning differs

    def test_invalid_universe_and_inputs(self):
        a = plan(2); b = plan(3)
        with self.assertRaises(InvalidPlan):compare_plans(a,b)
        for invalid in ([],{},None,dict(a, trace_ids=['missing'])):
            with self.assertRaises(InvalidPlan):compare_plans(a,invalid)
        r=compare_plans(plan(0),plan(0))
        self.assertEqual(r['before']['status'],'EMPTY_UNIVERSE')
        self.assertEqual(r['status'],'SAME_ACCOUNTING')

    def test_trace_selection_does_not_change_accounting(self):
        a=plan(5); b=dict(a,trace_ids=['id-1'])
        self.assertEqual(compare_plans(a,b)['status'],'SAME_ACCOUNTING')

    def test_comparison_delivery_failure(self):
        from batchscope.__main__ import main
        class Short(io.StringIO):
            def write(self, text):return super().write(text[:9])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'p.json';p.write_text(json.dumps(plan(4)))
            target=Short()
            with patch.object(sys,'argv',['batchscope','--compare',str(p),str(p)]), \
                    patch.object(sys,'stdout',target), patch.object(sys,'stderr',io.StringIO()):
                self.assertEqual(main(),2)
                self.assertIs(sys.stdout,target)
            if Path('/dev/full').exists():
                with open('/dev/full','wb') as output:
                    got=subprocess.run([sys.executable,'-m','batchscope','--compare',str(p),str(p)],
                                       stdout=output,stderr=subprocess.PIPE,timeout=5)
                self.assertEqual(got.returncode,2)

    def test_cli(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a.json',Path(d)/'b.json'
            a.write_text(json.dumps(plan(5)));b.write_text(json.dumps(plan(5,loader_drop_last=True)))
            def run(*args):return subprocess.run([sys.executable,'-m','batchscope',*args],capture_output=True,timeout=5)
            got=run('--compare',str(a),str(b));self.assertEqual(got.returncode,1)
            self.assertEqual(json.loads(got.stdout)['schema'],'batchscope-comparison-v1')
            self.assertEqual(run('--compare',str(a),str(a)).returncode,0)
            b.write_text(json.dumps(plan(3)))
            got=run('--compare',str(a),str(b));self.assertEqual(got.returncode,2);self.assertEqual(got.stdout,b'')
            for args in (('--compare',str(a)),('--compare',str(a),str(b),'extra')):
                self.assertEqual(run(*args).returncode,2)
