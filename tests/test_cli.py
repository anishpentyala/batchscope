import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_core import plan


class CLI(unittest.TestCase):
    def run_cli(self, raw, **kw):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'plan.json';p.write_bytes(raw)
            return subprocess.run([sys.executable,'-m','batchscope',str(p)],
                                  capture_output=True,timeout=5,**kw)

    def test_clean_and_changed(self):
        for n,code in [(4,0),(5,1)]:
            got=self.run_cli(json.dumps(plan(n)).encode())
            self.assertEqual(got.returncode,code)
            self.assertEqual(json.loads(got.stdout)['universe_size'],n)
            self.assertEqual(got.stderr,b'')

    def test_invalid_json(self):
        for raw in [b'{',b'[]',b'{"ids":[],"ids":[]}',b'{"x":NaN}',b'\xff',b'{}']:
            with self.subTest(raw=raw):
                got=self.run_cli(raw);self.assertEqual(got.returncode,2)
                self.assertEqual(got.stdout,b'')

    def test_cap(self):
        got=self.run_cli(b' '*((8*1024*1024)+1))
        self.assertEqual(got.returncode,2);self.assertEqual(got.stdout,b'')

    @unittest.skipUnless(hasattr(os,'mkfifo'),'requires FIFOs')
    def test_writerless_fifo(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pipe';os.mkfifo(p)
            got=subprocess.run([sys.executable,'-m','batchscope',str(p)],
                               capture_output=True,timeout=3)
            self.assertEqual(got.returncode,2);self.assertEqual(got.stdout,b'')

    @unittest.skipUnless(Path('/dev/full').exists(),'requires dev/full')
    def test_output_failure(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'plan.json';p.write_text(json.dumps(plan(4)))
            with open('/dev/full','wb') as target:
                got=subprocess.run([sys.executable,'-m','batchscope',str(p)],
                                   stdout=target,stderr=subprocess.PIPE,timeout=3)
            self.assertEqual(got.returncode,2)

    def test_unicode_ids_are_not_normalized(self):
        p=plan(ids=['é','e\u0301','😀','x'])
        got=self.run_cli(json.dumps(p).encode())
        self.assertEqual([x['id'] for x in json.loads(got.stdout)['ids']],p['ids'])

    def test_trace_selection(self):
        p = plan(5)
        p['trace_ids'] = [p['ids'][-1], p['ids'][0]]
        got = self.run_cli(json.dumps(p).encode())
        self.assertEqual(got.returncode, 1)
        self.assertEqual([t['id'] for t in json.loads(got.stdout)['traces']], p['trace_ids'])
        p['trace_ids'] = ['not-declared']
        got = self.run_cli(json.dumps(p).encode())
        self.assertEqual(got.returncode, 2)
        self.assertEqual(got.stdout, b'')

    def test_help(self):
        got=subprocess.run([sys.executable,'-m','batchscope','--help'],capture_output=True)
        self.assertEqual(got.returncode,0)
