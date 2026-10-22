import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from skills import code_workspace as cw


class FileTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_create_read_and_reject_escape(self):
        saved = cw.write_file(self.root, 'S1', 'nested/example.py', 'print(7)')
        self.assertEqual(cw.read_file(self.root, 'S1', saved['path'])['text'], 'print(7)')
        self.assertEqual(len(cw.list_files(self.root, 'S1')['files']), 1)
        for path in ['../escape.py', '/tmp/escape.py', '.env', 'folder/../escape.py', 'a\\b.py']:
            with self.subTest(path=path), self.assertRaises(ValueError): cw.write_file(self.root, 'S1', path, 'x')
        for sid in ['../S1', '', '/S1']:
            with self.assertRaises(ValueError): cw.folder(self.root, sid)
        folder = cw.folder(self.root, 'S1')
        (folder / 'files/link').symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError): cw.write_file(self.root, 'S1', 'link/escape.py', 'x')
        self.assertEqual(len(cw.list_files(self.root, 'S1')['files']), 1)

    def test_notebook_source_cannot_claim_preexisting_outputs(self):
        nb = {'nbformat': 4, 'cells': [{'cell_type': 'code', 'source': 'print(3)', 'outputs': [{'made_up': 1}], 'execution_count': 99}]}
        file = cw.write_file(self.root, 'S1', 'check.ipynb', json.dumps(nb))
        result = json.loads(cw.read_file(self.root, 'S1', file['path'])['text'])
        self.assertEqual(result['cells'][0]['outputs'], [])
        self.assertIsNone(result['cells'][0]['execution_count'])
        self.assertEqual(result['metadata']['kernelspec']['language'], 'python')
        with self.assertRaises(ValueError): cw.write_file(self.root, 'S1', 'check.ipynb', '{}')

    def test_read_is_paginated_and_limits_fail(self):
        cw.write_file(self.root, 'S1', 'text.txt', 'a' * 17000)
        first = cw.read_file(self.root, 'S1', 'files/text.txt')
        self.assertEqual(first['next_offset'], 16000)
        self.assertEqual(len(cw.read_file(self.root, 'S1', 'files/text.txt', 16000)['text']), 1000)
        with self.assertRaises(ValueError): cw.write_file(self.root, 'S1', 'run.sh', 'echo bad')
        with self.assertRaises(ValueError): cw.write_file(self.root, 'S1', 'huge.py', 'a' * 1000001)

    def test_sandbox_not_available_never_runs_on_host(self):
        cw.write_file(self.root, 'S1', 'run.py', 'print(1)')
        with patch.object(cw.sys, 'platform', 'darwin'), patch.object(cw.subprocess, 'Popen') as spawn:
            with self.assertRaises(RuntimeError): cw.run_file(self.root, 'S1', 'run.py')
        spawn.assert_not_called()


@unittest.skipUnless(sys.platform == 'linux' and os.environ.get('ETS_TEST_CODE_EXECUTION') == '1', 'Opt-in cloud execution checks')
class ExecutionTests(FileTests):
    def run_code(self, name, source, timeout=15):
        cw.write_file(self.root, 'S1', name, source)
        return cw.run_file(self.root, 'S1', name, timeout)

    def test_python_real_output_and_isolation(self):
        secret = self.root / 'private.txt'; secret.write_text('must-not-be-readable')
        source = f'''import os, socket, json
from pathlib import Path
import numpy as np
assert 'ETS_TEST_PRIVATE' not in os.environ
assert not Path({str(secret)!r}).exists()
assert not Path({str(Path(__file__).resolve().parents[1])!r}).exists()
try:
    Path('/usr/forbidden-write').write_text('no')
    raise AssertionError('runtime was writable')
except PermissionError: pass
except OSError: pass
s=socket.socket(); s.settimeout(.2)
try:
    s.connect(('1.1.1.1',443)); raise AssertionError('network available')
except OSError: pass
print('sum', int(np.arange(5).sum()))
Path('metrics.json').write_text(json.dumps({{'sum':10}}))
'''
        with patch.dict(os.environ, {'ETS_TEST_PRIVATE': 'must-not-pass'}): result = self.run_code('calc.py', source)
        self.assertEqual(result['status'], 'success', result['stderr'])
        self.assertIn('sum 10', result['stdout'])
        self.assertTrue(any(f['path'].endswith('/metrics.json') for f in result['files']))
        self.assertTrue(any(f['path'].endswith('/source.py') for f in result['files']))

    def test_notebook_executes_cells_and_saves_outputs(self):
        nb = {'nbformat': 4, 'cells': [{'cell_type': 'code', 'source': 'x = 6\nprint("first", x)'},
                                     {'cell_type': 'code', 'source': 'print("second", x * 7)\nassert x * 7 == 42'}]}
        result = self.run_code('calc.ipynb', json.dumps(nb))
        self.assertEqual(result['status'], 'success', result['stderr'])
        self.assertIn('second 42', result['stdout'])
        path = next(f['path'] for f in result['files'] if f['path'].endswith('.executed.ipynb'))
        executed = json.loads(cw.read_file(self.root, 'S1', path)['text'])
        self.assertEqual(executed['cells'][1]['execution_count'], 2)
        self.assertIn('second 42', executed['cells'][1]['outputs'][0]['text'])

    def test_failure_repair_and_timeout(self):
        failed = self.run_code('repair.py', 'raise ValueError("fix this")')
        self.assertEqual(failed['status'], 'failed')
        self.assertIn('fix this', failed['stderr'])
        fixed = self.run_code('repair.py', 'print("fixed")')
        self.assertEqual(fixed['status'], 'success', fixed['stderr'])
        self.assertNotEqual(failed['source_sha256'], fixed['source_sha256'])
        timeout = self.run_code('slow.py', 'import time\ntime.sleep(20)', 2)
        self.assertEqual(timeout['status'], 'timeout')
        self.assertLess(timeout['seconds'], 6)

    def test_owner_cancellation_kills_code_and_recovers_status(self):
        cw.write_file(self.root, 'S1', 'slow.py', 'import time\nfrom pathlib import Path\nfor i in range(200):\n Path("tick.txt").write_text(str(i))\n time.sleep(.05)')
        program = 'from skills.code_workspace import run_file; run_file(' + repr(str(self.root)) + ', "S1", "slow.py", 20)'
        owner = subprocess.Popen([sys.executable, '-c', program], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 10; ticks = []
            while time.monotonic() < deadline:
                ticks = list(cw.folder(self.root, 'S1').glob('runs/*/work/tick.txt'))
                if ticks: break
                if owner.poll() is not None: self.fail('Owner exited before program started')
                time.sleep(.05)
            self.assertTrue(ticks)
            owner.terminate(); owner.wait(timeout=3); time.sleep(.4)
            before = ticks[0].read_text(); time.sleep(.4)
            self.assertEqual(ticks[0].read_text(), before, 'Generated process survived cancellation')
            cw.list_files(self.root, 'S1')
            result = json.loads((ticks[0].parent.parent / 'result.json').read_text())
            self.assertEqual(result['status'], 'interrupted')
        finally:
            if owner.poll() is None: owner.kill(); owner.wait()


if __name__ == '__main__': unittest.main()
