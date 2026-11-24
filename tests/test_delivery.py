import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from batchscope.__main__ import main
from test_core import plan


class ShortWriter(io.StringIO):
    def write(self, value):
        return super().write(value[:7])


class NoneWriter(io.StringIO):
    def write(self, value):
        super().write(value)
        return None


class RaisingWriter(io.StringIO):
    def write(self, value):
        raise OSError('write failed')


class FailingFlush(io.StringIO):
    def flush(self):
        raise OSError('flush failed')


class Delivery(unittest.TestCase):
    def invoke(self, target, n=4, help=False):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'plan.json'
            p.write_text(json.dumps(plan(n)))
            args = ['batchscope', '--help' if help else str(p)]
            errors = io.StringIO()
            with patch.object(sys, 'argv', args), patch.object(sys, 'stdout', target), \
                    patch.object(sys, 'stderr', errors):
                code = main()
                self.assertIs(sys.stdout, target)
                self.assertFalse(target.closed)
            return code, errors.getvalue()

    def test_short_write_and_flush_failure(self):
        for target_type in (ShortWriter, NoneWriter, RaisingWriter, FailingFlush):
            for n in (4, 5):
                with self.subTest(stream=target_type, n=n):
                    code, error = self.invoke(target_type(), n)
                    self.assertEqual(code, 2)
                    self.assertIn('output failure', error)

    def test_help_uses_same_delivery_contract(self):
        for target_type in (ShortWriter, NoneWriter, RaisingWriter, FailingFlush):
            code, _ = self.invoke(target_type(), help=True)
            self.assertEqual(code, 2)

    def test_normal_caller_stream_and_status(self):
        for n, status in ((4, 0), (5, 1)):
            target = io.StringIO()
            code, error = self.invoke(target, n)
            self.assertEqual(code, status)
            self.assertEqual(error, '')
            self.assertEqual(json.loads(target.getvalue())['universe_size'], n)
        target = io.StringIO()
        self.assertEqual(self.invoke(target, help=True), (0, ''))
        self.assertTrue(target.getvalue().startswith('Usage:'))

    def test_closed_caller_stream_and_stderr(self):
        target, errors = io.StringIO(), io.StringIO()
        target.close(); errors.close()
        with patch.object(sys, 'argv', ['batchscope', '--help']), \
                patch.object(sys, 'stdout', target), patch.object(sys, 'stderr', errors):
            self.assertEqual(main(), 2)
            self.assertIs(sys.stdout, target)

    @unittest.skipUnless(os.name == 'posix', 'POSIX pipe control')
    def test_process_closed_pipe_report_and_help(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'plan.json'; p.write_text(json.dumps(plan(4)))
            for value in (str(p), '--help'):
                read_fd, write_fd = os.pipe(); os.close(read_fd)
                try:
                    got = subprocess.run([sys.executable, '-m', 'batchscope', value],
                                         stdout=write_fd, stderr=subprocess.PIPE, timeout=5)
                finally:
                    os.close(write_fd)
                self.assertEqual(got.returncode, 2)
                self.assertNotIn(b'Exception ignored', got.stderr)
