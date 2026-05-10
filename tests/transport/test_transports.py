from __future__ import annotations

import socket
import sys
import time
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from systems.transport.messages import ActionCommand, CapabilityReport, FrameHeader, TaskRequest
from systems.transport.shm import SharedMemoryRing
from systems.transport.bus.zmq_bus import ZmqBus


def _free_tcp_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_shared_memory_ring_roundtrip() -> None:
    name = f"isaac_aura_test_{uuid.uuid4().hex[:12]}"
    writer = SharedMemoryRing(name=name, slot_size=1024, capacity=4, create=True)
    reader = SharedMemoryRing(name=name, slot_size=1024, capacity=4, create=False)
    try:
        ref = writer.write(b"payload-123")
        assert reader.read(ref) == b"payload-123"
    finally:
        reader.close()
        writer.close(unlink=True)


def test_shared_memory_ring_reclaims_stale_writer_segment() -> None:
    name = f"isaac_aura_test_{uuid.uuid4().hex[:12]}"
    stale_writer = SharedMemoryRing(name=name, slot_size=1024, capacity=4, create=True)
    stale_writer.close(unlink=False)

    writer = SharedMemoryRing(name=name, slot_size=1024, capacity=4, create=True)
    reader = SharedMemoryRing(name=name, slot_size=1024, capacity=4, create=False)
    try:
        ref = writer.write(b"fresh-payload")
        assert reader.read(ref) == b"fresh-payload"
    finally:
        reader.close()
        writer.close(unlink=True)


def test_zmq_bus_roundtrip() -> None:
    pytest.importorskip("zmq")
    port = _free_tcp_port()
    control_endpoint = f"tcp://127.0.0.1:{port}"
    telemetry_endpoint = f"tcp://127.0.0.1:{port + 1}"
    server = ZmqBus(control_endpoint=control_endpoint, telemetry_endpoint=telemetry_endpoint, role="bridge")
    client = ZmqBus(control_endpoint=control_endpoint, telemetry_endpoint=telemetry_endpoint, role="agent")
    try:
        client.publish("isaac.capability", CapabilityReport(component="memory_agent", status="ready"))
        deadline = time.time() + 2.0
        records = []
        while time.time() < deadline:
            records = server.poll("isaac.capability", max_items=4)
            if records:
                break
            time.sleep(0.01)
        assert len(records) == 1
        assert records[0].message.component == "memory_agent"

        server.publish("isaac.task", TaskRequest(command_text="follow"))
        deadline = time.time() + 2.0
        task_records = []
        while time.time() < deadline:
            task_records = client.poll("isaac.task", max_items=4)
            if task_records:
                break
            time.sleep(0.01)
        assert len(task_records) == 1
        assert task_records[0].message.command_text == "follow"

        server.publish("isaac.observation", FrameHeader(frame_id=1, timestamp_ns=1, source="bridge"))
        deadline = time.time() + 2.0
        observation_records = []
        while time.time() < deadline:
            observation_records = client.poll("isaac.observation", max_items=4)
            if observation_records:
                break
            time.sleep(0.01)
        assert len(observation_records) == 1

        client.publish("isaac.command", ActionCommand(action_type="STOP"))
        deadline = time.time() + 2.0
        command_records = []
        while time.time() < deadline:
            command_records = server.poll("isaac.command", max_items=4)
            if command_records:
                break
            time.sleep(0.01)
        assert len(command_records) == 1
        assert command_records[0].message.action_type == "STOP"
    finally:
        client.close()
        server.close()


def test_zmq_bus_closes_control_socket_when_telemetry_bind_fails() -> None:
    pytest.importorskip("zmq")
    control_port = _free_tcp_port()
    telemetry_port = _free_tcp_port()
    control_endpoint = f"tcp://127.0.0.1:{control_port}"
    telemetry_endpoint = f"tcp://127.0.0.1:{telemetry_port}"
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", telemetry_port))
    blocker.listen(1)
    try:
        with pytest.raises(Exception):
            ZmqBus(control_endpoint=control_endpoint, telemetry_endpoint=telemetry_endpoint, role="bridge")
    finally:
        blocker.close()

    replacement_telemetry_endpoint = f"tcp://127.0.0.1:{_free_tcp_port()}"
    server = ZmqBus(
        control_endpoint=control_endpoint,
        telemetry_endpoint=replacement_telemetry_endpoint,
        role="bridge",
    )
    server.close()


def test_zmq_bus_late_agent_receives_buffered_control_message() -> None:
    pytest.importorskip("zmq")
    port = _free_tcp_port()
    control_endpoint = f"tcp://127.0.0.1:{port}"
    telemetry_endpoint = f"tcp://127.0.0.1:{port + 1}"
    server = ZmqBus(control_endpoint=control_endpoint, telemetry_endpoint=telemetry_endpoint, role="bridge")
    client = None
    try:
        server.publish("isaac.task", TaskRequest(command_text="delayed follow"))
        client = ZmqBus(control_endpoint=control_endpoint, telemetry_endpoint=telemetry_endpoint, role="agent")
        client.publish("isaac.capability", CapabilityReport(component="memory_agent", status="ready"))
        deadline = time.time() + 2.0
        records = []
        while time.time() < deadline:
            server.poll("isaac.capability", max_items=4)
            records = client.poll("isaac.task", max_items=4)
            if records:
                break
            time.sleep(0.01)
        assert len(records) == 1
        assert records[0].message.command_text == "delayed follow"
    finally:
        if client is not None:
            client.close()
        server.close()


def test_zmq_bus_multi_agent_fanout_and_replay() -> None:
    pytest.importorskip("zmq")
    port = _free_tcp_port()
    control_endpoint = f"tcp://127.0.0.1:{port}"
    telemetry_endpoint = f"tcp://127.0.0.1:{port + 1}"
    server = ZmqBus(control_endpoint=control_endpoint, telemetry_endpoint=telemetry_endpoint, role="bridge")
    client_a = ZmqBus(control_endpoint=control_endpoint, telemetry_endpoint=telemetry_endpoint, role="agent", identity="agent-a")
    client_b = None
    try:
        client_a.publish("isaac.capability", CapabilityReport(component="agent_a", status="ready"))
        deadline = time.time() + 2.0
        while time.time() < deadline:
            records = server.poll("isaac.capability", max_items=4)
            if records:
                break
            time.sleep(0.01)

        server.publish("isaac.task", TaskRequest(command_text="shared follow"))

        deadline = time.time() + 2.0
        task_records_a = []
        while time.time() < deadline:
            task_records_a = client_a.poll("isaac.task", max_items=4)
            if task_records_a:
                break
            time.sleep(0.01)
        assert len(task_records_a) == 1
        assert task_records_a[0].message.command_text == "shared follow"

        client_b = ZmqBus(control_endpoint=control_endpoint, telemetry_endpoint=telemetry_endpoint, role="agent", identity="agent-b")
        client_b.publish("isaac.capability", CapabilityReport(component="agent_b", status="ready"))
        deadline = time.time() + 2.0
        task_records_b = []
        while time.time() < deadline:
            server.poll("isaac.capability", max_items=8)
            task_records_b = client_b.poll("isaac.task", max_items=4)
            if task_records_b:
                break
            time.sleep(0.01)
        assert len(task_records_b) == 1
        assert task_records_b[0].message.command_text == "shared follow"
        assert server.health.control.peer_count == 2
    finally:
        if client_b is not None:
            client_b.close()
        client_a.close()
        server.close()
