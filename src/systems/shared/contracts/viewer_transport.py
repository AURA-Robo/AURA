"""Shared viewer transport constants exposed as contracts."""

from __future__ import annotations


VIEWER_CONTROL_ENDPOINT = "tcp://127.0.0.1:18880"
VIEWER_TELEMETRY_ENDPOINT = "tcp://127.0.0.1:18881"
VIEWER_SHM_NAME = "aura_viewer_shm_01"
VIEWER_SHM_SLOT_SIZE = 8 * 1024 * 1024
VIEWER_SHM_CAPACITY = 8
VIEWER_OBSERVATION_TOPIC = "isaac.observation"
VIEWER_HEALTH_TOPIC = "isaac.health"


__all__ = [
    "VIEWER_CONTROL_ENDPOINT",
    "VIEWER_TELEMETRY_ENDPOINT",
    "VIEWER_SHM_NAME",
    "VIEWER_SHM_SLOT_SIZE",
    "VIEWER_SHM_CAPACITY",
    "VIEWER_OBSERVATION_TOPIC",
    "VIEWER_HEALTH_TOPIC",
]
