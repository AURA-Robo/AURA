# Transport Subsystem Context

## Scope

`systems.transport` is the runtime communication substrate: message contracts, JSON codecs, ndarray/shared-memory codecs, shared-memory ring buffers, in-process bus, ZMQ bus, and transport health.

## Read First

- `messages.py`
- `codec.py`
- `frame_codec.py`
- `shm.py`
- `health.py`
- `bus/base.py`
- `bus/inproc_bus.py`
- `bus/zmq_bus.py`

## Contract Surfaces

- Message dataclasses serialize with a `__type__` discriminator.
- Shared-memory refs carry array metadata plus slot identity.
- ZMQ bus routes telemetry/control topics and can replay retained control messages to late subscribers.

## Boundary Rules

- Keep transport domain-neutral. Do not add navigation, perception, reasoning, or dashboard business logic here.
- Preserve message compatibility unless all producers/consumers and tests are migrated together.
- Keep pyzmq-specific behavior isolated to the ZMQ bus implementation.

## State And Side Effects

Shared-memory rings mutate circular buffers and can unlink segments on close. ZMQ bus instances track peers, pending control messages, retained control history, and health counters.

## Cautions

- `SharedMemoryRing.read()` can raise when a slot has been overwritten before consumption.
- `SharedMemoryRing(create=True)` tries to reclaim a stale same-name segment; if the OS still keeps the name reserved, it
  creates a unique suffixed ring name. Consumers should follow the `ShmSlotRef.name` carried with frame metadata.
- Invalid multipart ZMQ payloads may be dropped and surfaced through health counters rather than exceptions.
- If ZMQ bus startup fails partway through socket creation, close any sockets that were already opened before raising.
- Always close/unlink shared-memory resources deliberately in tests and runtime teardown.

## Tests

- `tests/transport/test_messages.py`
- `tests/transport/test_transports.py`
