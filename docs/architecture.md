# AURA System Architecture

This document describes the current system architecture around
`C:\Users\mango\project\AURA\system` and the dashboard frontend at
`C:\Users\mango\project\AURA\dashboard`.

## High-Level Shape

AURA is split into three runtime layers:

1. Dashboard frontend
   - React/Vite/Tauri app in the sibling `dashboard` project.
   - Reads dashboard state, sends operator commands, and displays live WebRTC
     video/telemetry.
2. Dashboard backend
   - `aiohttp` service in `src/backend`.
   - Owns or proxies runtime lifecycle, aggregates subsystem health, exposes
     dashboard APIs, and terminates WebRTC signaling.
3. Runtime subsystems
   - `src/systems/*` packages for control, inference, navigation, reasoning,
     memory, perception, transport, world state, and shared contracts.
   - `src/simulation` isolates Isaac Sim specific runtime assembly.

## Process Boundaries

Default local ports:

- Dashboard Vite dev server: `127.0.0.1:5173`
- Dashboard backend: `127.0.0.1:18095`
- Optional standalone runtime: `127.0.0.1:18096`
- Inference system: `127.0.0.1:15880`
- Reasoning system: `127.0.0.1:17881`
- Navigation system: `127.0.0.1:17882`
- Control runtime: `127.0.0.1:8892`

The normal dashboard bring-up path is:

1. Start `scripts/run_system/run_dashboard_windows.ps1`.
2. Reuse an already reachable backend at `/api/bootstrap`, or start
   `scripts/run_system/backend_windows.ps1`.
3. Export `AURA_DASHBOARD_API_BASE_URL` for the dashboard process.
4. Start the Tauri dashboard through `npm run tauri:dev` in the sibling
   `dashboard` project.

## Dashboard Frontend To Backend

The dashboard frontend resolves its backend base URL in
`dashboard/src/app/network.ts`:

- `VITE_AURA_API_BASE` wins when set.
- In Vite dev mode, the base is empty, so relative `/api/*` URLs are used.
- In packaged/desktop mode, the default backend is
  `http://127.0.0.1:18095`.

In development, `dashboard/vite.config.ts` proxies `/api` to
`AURA_DASHBOARD_PROXY_TARGET` when set, otherwise to
`http://127.0.0.1:18095`. The system dashboard launcher also sets
`AURA_DASHBOARD_API_BASE_URL` for backend readiness/process coordination.

Frontend state bootstrap in `dashboard/src/app/state.tsx`:

- `GET /api/bootstrap` returns API metadata, dev origin, and the WebRTC base
  path.
- `GET /api/state` returns the initial dashboard state snapshot.
- `GET /api/events` opens an `EventSource` stream. Backend SSE `state` events
  keep the UI live after initial bootstrap.

Operator actions:

- `POST /api/session/start` starts a runtime session.
- `POST /api/session/stop` stops it.
- `POST /api/runtime/task` sends task-like instructions to the reasoning
  service.
- `POST /api/runtime/reason` sends general utterances to the reasoning service.
- `POST /api/runtime/cancel` cancels the active reasoning/runtime request.

Knowledge and planner catalog UI:

- `GET/POST/PUT /api/knowledge/*` manage knowledge documents and status.
- `GET/POST/DELETE /api/planner/*` manage planner catalog intents and
  subgoals.

Memory inspection UI:

- `GET /api/memory/status`
- `GET /api/memory/blocks`
- `PUT /api/memory/blocks/{label}`
- `GET /api/memory/passages`
- `POST /api/memory/passages`

## Backend To Runtime Subsystems

The backend is created by `backend.api.serve_backend`, which calls
`backend.app.create_app`.

`DashboardSessionManager` in `src/backend/session_manager.py` builds dashboard
state by combining:

- Backend-owned runtime state or an external runtime state endpoint.
- Control runtime status from `/runtime/status`.
- Reasoning status from `/reasoning/status`.
- Navigation status from `/navigation/status`.
- WebRTC health from the backend WebRTC service.
- Object memory, agent memory, and knowledge runtime status.
- Logs, occupancy metadata, and runtime context summary data.

Backend proxy/control paths:

- `POST /api/runtime/reason` forwards to
  `{AURA_REASONING_SYSTEM_URL}/reasoning/respond`.
- `POST /api/runtime/task` uses the same reasoning route, but rejects pure
  dialogue responses for the legacy task endpoint.
- `POST /api/runtime/cancel` forwards to
  `{AURA_REASONING_SYSTEM_URL}/reasoning/cancel`.
- `POST /api/session/start` and `POST /api/session/stop` call either the
  backend-owned runtime service or the configured external runtime URL.

## WebRTC Viewer Path

The dashboard viewer uses `dashboard/src/app/hooks/useWebRTCViewer.ts`.

Flow:

1. Dashboard reads `webrtcBasePath` from `/api/bootstrap`.
2. Dashboard calls `GET {webrtcBasePath}/config`.
3. Dashboard creates a browser `RTCPeerConnection`.
4. Dashboard posts the SDP offer to `POST {webrtcBasePath}/offer`.
5. Backend returns the SDP answer from the in-process WebRTC service or an
   external WebRTC proxy.
6. Media tracks carry RGB/depth video.
7. Data channels carry frame state and telemetry messages.

The backend WebRTC service can also feed object-memory ingestion from live
viewer frames when object memory is configured and enabled.

## Subsystem Ownership

- `systems.control`
  - Runtime control, operator input, telemetry publishing, and Isaac Sim public
    entrypoint.
- `systems.inference`
  - Model stack supervision and model-serving surfaces for NavDP, System2,
    dialogue, and planner completion.
- `systems.navigation`
  - Navigation status, command/update/cancel APIs, geometry, goals, and follower
    logic. Common navigation state DTOs live in `systems.shared.contracts`.
- `systems.reasoning`
  - Natural-language routing, planner orchestration, task-frame validation, and
    planner catalog runtime. It compiles humanoid agent memory into
    dialogue/planning context on each reasoning response.
- `systems.memory`
  - STM, humanoid agent memory, object memory, knowledge documents,
    conversation memory, and Postgres repositories.
- `systems.perception`
  - Camera control, detector runtime, observation contracts, and perception
    telemetry.
- `systems.transport`
  - Runtime messages, in-process/ZMQ buses, codecs, shared-memory transport,
    and transport health.
- `systems.world_state`
  - Runtime state DTOs and status assembly helpers built on shared contracts.
- `systems.shared.contracts`
  - DTOs/contracts shared across subsystem API boundaries.
- `src/simulation`
  - Isaac Sim assembly, scene setup, policy session, and observation layout.

## Boundary Rules

- Cross-subsystem imports go through `systems.<subsystem>.api.*` or
  `systems.shared.contracts.*`.
- Dashboard frontend does not call subsystem services directly. It talks to the
  backend `/api/*` surface.
- Backend owns dashboard API compatibility and adapts subsystem/runtime details
  into stable dashboard payloads.
- Isaac Sim specific implementation stays under `src/simulation`; `systems`
  packages expose facades and contracts around it.
- Default local service endpoints live in
  `systems.shared.contracts.service_endpoints`; entrypoints and launchers should
  align with those contracts when adding or changing process boundaries.
