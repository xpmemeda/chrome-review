import json
import os
import tempfile
import unittest
from pathlib import Path

from reverse_agent import AgentState, Dispatcher, Request


class ReverseAgentTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dispatcher = Dispatcher(AgentState([self.root], self.root / 'state', None))

    def tearDown(self):
        self.temp.cleanup()

    def request(self, path, body=None, method='POST'):
        response = self.dispatcher.handle(Request(method, path, {}, {}, json.dumps(body).encode()))
        return response.status, json.loads(response.body)

    def test_health(self):
        status, result = self.request('/v1/health', method='GET')
        self.assertEqual(status, 200)
        self.assertTrue(result['hostname'])

    def test_unicode_and_exit_code(self):
        command = "Write-Output '你好'; exit 7" if os.name == 'nt' else "printf '你好'; exit 7"
        status, result = self.request('/v1/exec', {'command': command})
        self.assertEqual(status, 200)
        self.assertEqual(result['exit_code'], 7)
        self.assertEqual(result['stdout'].strip(), '你好')

    def test_timeout(self):
        command = 'Start-Sleep 30' if os.name == 'nt' else 'sleep 30'
        status, _ = self.request('/v1/exec', {'command': command, 'timeout': 0.2})
        self.assertEqual(status, 408)

    def test_job_stop(self):
        command = 'Start-Sleep 30' if os.name == 'nt' else 'sleep 30'
        status, job = self.request('/v1/jobs', {'command': command})
        self.assertEqual(status, 201)
        status, result = self.request(f"/v1/jobs/{job['id']}/stop")
        self.assertEqual(status, 200)
        self.assertFalse(result['running'])

    def test_path_boundary(self):
        with self.assertRaises(PermissionError):
            self.dispatcher.state.resolve_path('../outside')


if __name__ == '__main__':
    unittest.main()
