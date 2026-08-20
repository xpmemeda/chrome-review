#!/usr/bin/env python3

import hashlib
import json
import socket
import tempfile
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from argparse import Namespace
from pathlib import Path

from agent import AgentServer, AgentState, Handler
from client import command_payload


class AgentTest(unittest.TestCase):
    def test_client_strips_argument_separator(self):
        args = Namespace(command=["--", "printf hello"], env=[], cwd=None, timeout=3)
        self.assertEqual(command_payload(args, timeout=True)["command"], "printf hello")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / "root"
        self.root.mkdir()
        state = AgentState([self.root], Path(self.temp.name) / "state", 1024 * 1024)
        self.server = AgentServer(("127.0.0.1", 0), Handler, state)
        self.assertEqual(self.server.address_family, socket.AF_INET)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.temp.cleanup()

    def request(self, method, path, data=None, headers=None):
        request_headers = dict(headers or {})
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=request_headers)
        return urllib.request.urlopen(req)

    def json_request(self, method, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        with self.request(method, path, data, headers={"Content-Type": "application/json"}) as response:
            return json.load(response)

    def test_health(self):
        self.assertTrue(self.json_request("GET", "/v1/health")["ok"])

    def test_exec(self):
        result = self.json_request("POST", "/v1/exec", {
            "command": "printf hello", "cwd": str(self.root), "timeout": 5,
        })
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["stdout"], "hello")

    def test_exec_timeout(self):
        data = json.dumps({
            "command": "sleep 10", "cwd": str(self.root), "timeout": 0.05,
        }).encode()
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("POST", "/v1/exec", data, headers={"Content-Type": "application/json"})
        self.assertEqual(caught.exception.code, 408)

    def test_upload_download_and_path_boundary(self):
        data = b"file-content"
        digest = hashlib.sha256(data).hexdigest()
        query = urllib.parse.urlencode({"path": "nested/file.bin"})
        with self.request("PUT", f"/v1/files?{query}", data, headers={"X-Content-SHA256": digest}) as response:
            uploaded = json.load(response)
        self.assertEqual(uploaded["sha256"], digest)
        with self.request("GET", f"/v1/files?{query}") as response:
            self.assertEqual(response.read(), data)
        outside = urllib.parse.urlencode({"path": "/etc/passwd"})
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.request("GET", f"/v1/files?{outside}")
        self.assertEqual(caught.exception.code, 400)

    def test_background_job(self):
        job = self.json_request("POST", "/v1/jobs", {
            "command": "printf job-output", "cwd": str(self.root),
        })
        info = self.json_request("GET", f"/v1/jobs/{job['id']}/logs")
        for _ in range(20):
            if not info["running"]:
                break
            import time
            time.sleep(0.05)
            info = self.json_request("GET", f"/v1/jobs/{job['id']}/logs")
        self.assertEqual(info["exit_code"], 0)
        self.assertEqual(info["log"], "job-output")


if __name__ == "__main__":
    unittest.main()
