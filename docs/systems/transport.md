# Transport Subsystem

- Scope: runtime message contracts, bus abstraction, in-process and ZMQ
  transports, shared-memory frame transport, frame/message codecs, and
  transport health.
- Package root: `src/systems/transport`

## Components

- `messages.py`
- `codec.py`
- `frame_codec.py`
- `shm.py`
- `health.py`
- `bus/base.py`
- `bus/inproc_bus.py`
- `bus/zmq_bus.py`

## Notes

Transport is runtime substrate, not domain logic. It is shared by runtime,
server processes, dashboard/backend integration, WebRTC/media adapters, and
bridge code.
