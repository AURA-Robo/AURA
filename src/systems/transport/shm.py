from __future__ import annotations

import os
import struct
import uuid
from dataclasses import dataclass
from multiprocessing import shared_memory
from threading import Lock


_HEADER = struct.Struct("<QQ")


@dataclass(frozen=True)
class ShmSlotRef:
    name: str
    slot_index: int
    payload_size: int
    sequence: int


class SharedMemoryRing:
    def __init__(self, name: str, slot_size: int, capacity: int, *, create: bool = True) -> None:
        if slot_size <= 0:
            raise ValueError("slot_size must be positive")
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.name = str(name)
        self.slot_size = int(slot_size)
        self.capacity = int(capacity)
        self._bytes_per_slot = _HEADER.size + self.slot_size
        size = self.capacity * self._bytes_per_slot
        self._shm = self._open_shared_memory(create=create, size=size)
        self._lock = Lock()
        self._write_count = 0

    def _open_shared_memory(self, *, create: bool, size: int) -> shared_memory.SharedMemory:
        if not create:
            return shared_memory.SharedMemory(name=self.name, create=False, size=size)
        try:
            return shared_memory.SharedMemory(name=self.name, create=True, size=size)
        except FileExistsError:
            stale = shared_memory.SharedMemory(name=self.name, create=False, size=size)
            try:
                stale.unlink()
            finally:
                stale.close()
            try:
                return shared_memory.SharedMemory(name=self.name, create=True, size=size)
            except FileExistsError:
                self.name = f"{self.name}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
                return shared_memory.SharedMemory(name=self.name, create=True, size=size)

    def close(self, *, unlink: bool = False) -> None:
        self._shm.close()
        if unlink:
            try:
                self._shm.unlink()
            except FileNotFoundError:
                pass

    def write(self, payload: bytes | bytearray | memoryview) -> ShmSlotRef:
        raw = bytes(payload)
        if len(raw) > self.slot_size:
            raise ValueError(f"payload exceeds slot size: {len(raw)} > {self.slot_size}")
        with self._lock:
            slot_index = self._write_count % self.capacity
            offset = slot_index * self._bytes_per_slot
            sequence = self._write_count + 1
            _HEADER.pack_into(self._shm.buf, offset, sequence, len(raw))
            start = offset + _HEADER.size
            self._shm.buf[start : start + len(raw)] = raw
            self._write_count = sequence
            return ShmSlotRef(
                name=self.name,
                slot_index=slot_index,
                payload_size=len(raw),
                sequence=sequence,
            )

    def read(self, ref: ShmSlotRef) -> bytes:
        offset = int(ref.slot_index) * self._bytes_per_slot
        sequence, payload_size = _HEADER.unpack_from(self._shm.buf, offset)
        if sequence != int(ref.sequence):
            raise RuntimeError("Shared memory slot was overwritten before it was read.")
        start = offset + _HEADER.size
        return bytes(self._shm.buf[start : start + payload_size])

    def read_latest(self) -> bytes | None:
        with self._lock:
            if self._write_count == 0:
                return None
            slot_index = (self._write_count - 1) % self.capacity
            ref = ShmSlotRef(
                name=self.name,
                slot_index=slot_index,
                payload_size=0,
                sequence=self._write_count,
            )
        return self.read(ref)
