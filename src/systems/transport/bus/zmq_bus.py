from __future__ import annotations

from collections import defaultdict, deque

from .base import BusRecord, MessageBus
from ..codec import decode_envelope, encode_envelope
from ..health import TransportHealthTracker
from ..messages import MessagePayload


class ZmqBus(MessageBus):
    CONTROL_TOPICS = {"isaac.task", "isaac.command", "isaac.notice", "isaac.capability"}
    TELEMETRY_TOPICS = {"isaac.observation", "isaac.status", "isaac.health"}
    RETAINED_CONTROL_TOPICS = {"isaac.task", "isaac.notice", "isaac.capability"}

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        bind: bool | None = None,
        control_endpoint: str = "",
        telemetry_endpoint: str = "",
        role: str = "",
        identity: str = "",
        control_timeout_ms: int = 250,
        telemetry_timeout_ms: int = 250,
        retained_control_history: int = 8,
    ) -> None:
        try:
            import zmq
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("pyzmq is required for ZmqBus.") from exc

        resolved_control = control_endpoint or str(endpoint or "")
        if resolved_control == "":
            resolved_control = "tcp://127.0.0.1:5560"
        resolved_telemetry = telemetry_endpoint or self._derive_telemetry_endpoint(resolved_control)
        resolved_role = str(role).strip().lower()
        if resolved_role == "":
            resolved_role = "bridge" if bool(bind) else "agent"
        if resolved_role not in {"bridge", "agent"}:
            raise ValueError(f"Unsupported ZMQ role: {role!r}")

        self._zmq = zmq
        self._context = zmq.Context.instance()
        self._role = resolved_role
        self._control_endpoint = resolved_control
        self._telemetry_endpoint = resolved_telemetry
        self._buffer: dict[str, deque[BusRecord]] = defaultdict(deque)
        self._pending_control: deque[tuple[bytes, bytes]] = deque()
        self._known_peers: list[bytes] = []
        self._retained_control_limit = max(int(retained_control_history), 0)
        self._retained_control: dict[str, deque[tuple[bytes, bytes]]] = defaultdict(
            lambda: deque(maxlen=self._retained_control_limit if self._retained_control_limit > 0 else None)
        )
        self._health = TransportHealthTracker()
        self._control_socket = None
        self._telemetry_socket = None
        try:
            self._control_socket = self._build_control_socket(timeout_ms=int(control_timeout_ms), identity=str(identity))
            self._telemetry_socket = self._build_telemetry_socket(timeout_ms=int(telemetry_timeout_ms))
        except Exception:
            if self._telemetry_socket is not None:
                self._telemetry_socket.close(linger=0)
                self._telemetry_socket = None
            if self._control_socket is not None:
                self._control_socket.close(linger=0)
                self._control_socket = None
            raise

        self._poller = zmq.Poller()
        self._poller.register(self._control_socket, zmq.POLLIN)
        if self._telemetry_socket is not None:
            self._poller.register(self._telemetry_socket, zmq.POLLIN)

    @property
    def health(self) -> TransportHealthTracker:
        return self._health

    def publish(self, topic: str, message: MessagePayload) -> None:
        raw_topic, raw_payload = encode_envelope(topic, message)
        plane = self._route_plane(topic)
        if plane == "telemetry" and self._role == "bridge":
            self._telemetry_socket.send_multipart([raw_topic, raw_payload])
            self._health.telemetry.on_send()
            return
        if plane == "control" and self._role == "bridge":
            self._retain_control(topic, raw_topic, raw_payload)
            self._flush_incoming()
            if not self._known_peers:
                self._pending_control.append((raw_topic, raw_payload))
                self._health.control.on_queue(len(self._pending_control))
                return
            for peer in list(self._known_peers):
                self._control_socket.send_multipart([peer, raw_topic, raw_payload])
                self._health.control.on_send()
            return
        self._control_socket.send_multipart([raw_topic, raw_payload])
        self._health.control.on_send()

    def poll(self, topic: str, *, max_items: int | None = None) -> list[BusRecord]:
        self._flush_incoming()
        requested_topic = str(topic)
        queue = self._buffer[requested_topic]
        if max_items is None or max_items < 0:
            limit = len(queue)
        else:
            limit = min(len(queue), int(max_items))
        return [queue.popleft() for _ in range(limit)]

    def close(self) -> None:
        if self._telemetry_socket is not None:
            self._telemetry_socket.close(linger=0)
            self._telemetry_socket = None
        if self._control_socket is not None:
            self._control_socket.close(linger=0)
            self._control_socket = None

    def _build_control_socket(self, *, timeout_ms: int, identity: str):
        socket_type = self._zmq.ROUTER if self._role == "bridge" else self._zmq.DEALER
        socket = self._context.socket(socket_type)
        socket.setsockopt(self._zmq.LINGER, 0)
        socket.setsockopt(self._zmq.RCVTIMEO, int(timeout_ms))
        socket.setsockopt(self._zmq.SNDTIMEO, int(timeout_ms))
        if self._role != "bridge" and identity.strip() != "":
            socket.setsockopt(self._zmq.IDENTITY, identity.encode("utf-8", errors="ignore"))
        if self._role == "bridge":
            socket.bind(self._control_endpoint)
        else:
            socket.connect(self._control_endpoint)
        return socket

    def _build_telemetry_socket(self, *, timeout_ms: int):
        if self._role == "bridge":
            socket = self._context.socket(self._zmq.PUB)
            socket.setsockopt(self._zmq.LINGER, 0)
            socket.setsockopt(self._zmq.SNDTIMEO, int(timeout_ms))
            socket.bind(self._telemetry_endpoint)
            return socket
        socket = self._context.socket(self._zmq.SUB)
        socket.setsockopt(self._zmq.LINGER, 0)
        socket.setsockopt(self._zmq.RCVTIMEO, int(timeout_ms))
        socket.setsockopt(self._zmq.SUBSCRIBE, b"")
        socket.connect(self._telemetry_endpoint)
        return socket

    def _flush_incoming(self) -> None:
        while True:
            events = dict(self._poller.poll(timeout=0))
            if not events:
                break
            if self._control_socket in events:
                self._drain_control_socket()
            if self._telemetry_socket is not None and self._telemetry_socket in events:
                self._drain_telemetry_socket()
            self._flush_pending_control()

    def _drain_control_socket(self) -> None:
        while True:
            try:
                parts = self._control_socket.recv_multipart(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
            if self._role == "bridge":
                if len(parts) < 3:
                    self._health.control.on_drop(error="invalid ROUTER multipart payload")
                    continue
                peer, raw_topic, raw_payload = parts[0], parts[-2], parts[-1]
                if peer not in self._known_peers:
                    self._known_peers.append(peer)
                    self._health.control.on_retry()
                    self._health.control.on_peers(len(self._known_peers))
                    if not self._pending_control:
                        self._replay_retained_control(peer)
                topic, message = decode_envelope(raw_topic, raw_payload)
            else:
                if len(parts) < 2:
                    self._health.control.on_drop(error="invalid DEALER multipart payload")
                    continue
                topic, message = decode_envelope(parts[-2], parts[-1])
            self._buffer[topic].append(BusRecord(topic=topic, message=message))
            self._health.control.on_recv()

    def _drain_telemetry_socket(self) -> None:
        assert self._telemetry_socket is not None
        while True:
            try:
                raw_topic, raw_payload = self._telemetry_socket.recv_multipart(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
            topic, message = decode_envelope(raw_topic, raw_payload)
            self._buffer[topic].append(BusRecord(topic=topic, message=message))
            self._health.telemetry.on_recv()

    def _flush_pending_control(self) -> None:
        if self._role != "bridge" or not self._known_peers:
            return
        while self._pending_control:
            raw_topic, raw_payload = self._pending_control.popleft()
            for peer in list(self._known_peers):
                self._control_socket.send_multipart([peer, raw_topic, raw_payload])
                self._health.control.on_send()
        self._health.control.on_queue(0)

    def _retain_control(self, topic: str, raw_topic: bytes, raw_payload: bytes) -> None:
        if self._retained_control_limit <= 0:
            return
        normalized = str(topic)
        if normalized not in self.RETAINED_CONTROL_TOPICS:
            return
        self._retained_control[normalized].append((raw_topic, raw_payload))

    def _replay_retained_control(self, peer: bytes) -> None:
        if self._retained_control_limit <= 0:
            return
        for topic in sorted(self._retained_control):
            for raw_topic, raw_payload in self._retained_control[topic]:
                self._control_socket.send_multipart([peer, raw_topic, raw_payload])
                self._health.control.on_send()

    @classmethod
    def _route_plane(cls, topic: str) -> str:
        normalized = str(topic)
        if normalized in cls.TELEMETRY_TOPICS:
            return "telemetry"
        return "control"

    @staticmethod
    def _derive_telemetry_endpoint(control_endpoint: str) -> str:
        endpoint = str(control_endpoint).strip()
        prefix, sep, port_str = endpoint.rpartition(":")
        if sep != "" and port_str.isdigit():
            return f"{prefix}:{int(port_str) + 1}"
        return f"{endpoint}.telemetry"
