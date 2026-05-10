# AURA System Documentation

This directory contains the canonical documentation for the AURA system
runtime, backend, and subsystem boundaries.

## Documents

- [architecture.md](./architecture.md)
  - System architecture, process boundaries, runtime data flow, and dashboard
    frontend/backend integration.
- [systems/README.md](./systems/README.md)
  - Subsystem catalog moved from the former `sub/` directory and updated for
    the current `src/systems` layout.

## Notes

- Source of truth for runtime code is `src/`.
- Source of truth for subsystem ownership is `src/systems`.
- The dashboard frontend is a sibling project at
  `C:\Users\mango\project\AURA\dashboard`.
- The system backend is in this repository under `src/backend`.
