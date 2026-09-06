#!/usr/bin/env python3

import hashlib
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path

from agent import AgentServer, AgentState, Handler
from mcp_server import RemoteFleet, SSHSession, TransferManager


class LocalSSHSession(SSHSession):
    def __init__(self):
        pass

    def run(self, command, *, timeout=300, input_data=None):
        return subprocess.run(
            ["bash", "-lc", command],
            input=input_data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )


class AsyncTransferTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.remote_root = self.base / "remote"
        self.local_root = self.base / "local"
        self.remote_root.mkdir()
        self.local_root.mkdir()
        state = AgentState(
            [self.remote_root],
            self.base / "agent-state",
            1024 * 1024,
        )
        self.server = AgentServer(("127.0.0.1", 0), Handler, state)
        self.server_thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )
        self.server_thread.start()
        config = self.base / "hosts.toml"
        config.write_text(
            "\n".join(
                [
                    "[hosts.test]",
                    'backend = "agent"',
                    f'url = "http://127.0.0.1:{self.server.server_port}"',
                    f'root = "{self.remote_root}"',
                ]
            ),
            encoding="utf-8",
        )
        self.manager = TransferManager(RemoteFleet(config))
        self.manager.state_dir = self.base / "transfer-state"
        self.manager.state_dir.mkdir()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.temp.cleanup()

    def wait(self, transfer_ref: str) -> dict:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            state = self.manager.status(transfer_ref)
            if state["state"] not in {"queued", "running"}:
                return state
            time.sleep(0.01)
        self.fail(f"transfer did not finish: {transfer_ref}")

    def test_download_and_upload_resume_then_verify_and_rename(self):
        download_data = (b"download-block-" * 8192) + b"tail"
        remote_source = self.remote_root / "source.bin"
        remote_source.write_bytes(download_data)
        local_destination = self.local_root / "download.bin"
        local_partial = Path(str(local_destination) + ".partial")
        local_partial.write_bytes(download_data[:12345])
        expected = hashlib.sha256(download_data).hexdigest()

        started = self.manager.start_download(
            "test",
            str(remote_source),
            str(local_destination),
            expected,
            False,
        )
        finished = self.wait(started["transfer_ref"])
        self.assertEqual(finished["state"], "completed", finished)
        self.assertTrue(finished["resumed"])
        self.assertTrue(finished["verified_size"])
        self.assertTrue(finished["verified_sha256"])
        self.assertEqual(local_destination.read_bytes(), download_data)
        self.assertFalse(local_partial.exists())

        upload_data = (b"upload-block-" * 8192) + b"tail"
        local_source = self.local_root / "upload.bin"
        local_source.write_bytes(upload_data)
        remote_destination = self.remote_root / "destination.bin"
        remote_partial = Path(str(remote_destination) + ".partial")
        remote_partial.write_bytes(upload_data[:23456])
        expected = hashlib.sha256(upload_data).hexdigest()

        started = self.manager.start_upload(
            "test",
            str(local_source),
            str(remote_destination),
            expected,
            False,
        )
        finished = self.wait(started["transfer_ref"])
        self.assertEqual(finished["state"], "completed", finished)
        self.assertTrue(finished["resumed"])
        self.assertTrue(finished["verified_size"])
        self.assertTrue(finished["verified_sha256"])
        self.assertEqual(remote_destination.read_bytes(), upload_data)
        self.assertFalse(remote_partial.exists())

    def test_ssh_backend_uses_the_same_resumable_protocol(self):
        self.manager.fleet.sessions["test"] = LocalSSHSession()
        data = (b"ssh-transfer-" * 4096) + b"tail"
        expected = hashlib.sha256(data).hexdigest()

        remote_source = self.remote_root / "ssh-source.bin"
        remote_source.write_bytes(data)
        local_destination = self.local_root / "ssh-download.bin"
        Path(str(local_destination) + ".partial").write_bytes(data[:4321])
        started = self.manager.start_download(
            "test", str(remote_source), str(local_destination), expected, False
        )
        finished = self.wait(started["transfer_ref"])
        self.assertEqual(finished["state"], "completed", finished)
        self.assertEqual(local_destination.read_bytes(), data)

        local_source = self.local_root / "ssh-upload.bin"
        local_source.write_bytes(data)
        remote_destination = self.remote_root / "ssh-destination.bin"
        Path(str(remote_destination) + ".partial").write_bytes(data[:5432])
        started = self.manager.start_upload(
            "test", str(local_source), str(remote_destination), expected, False
        )
        finished = self.wait(started["transfer_ref"])
        self.assertEqual(finished["state"], "completed", finished)
        self.assertEqual(remote_destination.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
