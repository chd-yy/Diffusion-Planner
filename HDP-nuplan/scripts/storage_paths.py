"""Shared storage policy for large HDP simulation outputs."""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_EXTERNAL_MOUNT = Path("/media/yanjun/拯救者PSSD")
DEFAULT_EXTERNAL_SIM_ROOT = (
    DEFAULT_EXTERNAL_MOUNT / "Diffusion-Planner-HDP/simulation_outputs"
)


def require_allowed_output(
    requested: Path,
    local_root: Path,
    external_category: str,
) -> tuple[Path, str]:
    """Allow a legacy local root or a mounted external-PSSD category.

    External outputs fail closed when the PSSD is not mounted or writable.  In
    particular, they never fall back to creating a similarly named directory
    on the system disk after an unplug/reboot.
    """
    output = requested.expanduser().resolve()
    local = local_root.expanduser().resolve()
    external_mount = Path(
        os.environ.get("HDP_EXTERNAL_STORAGE_MOUNT", str(DEFAULT_EXTERNAL_MOUNT))
    ).expanduser().resolve()
    external_base = Path(
        os.environ.get("HDP_EXTERNAL_SIM_ROOT", str(DEFAULT_EXTERNAL_SIM_ROOT))
    ).expanduser().resolve()
    external = (external_base / external_category).resolve()

    if local in output.parents:
        return output, "local"
    if external not in output.parents:
        raise ValueError(
            f"output must be a child of {local} or mounted external root {external}"
        )
    if not external_mount.is_mount():
        raise RuntimeError(
            f"external storage is not mounted at {external_mount}; refusing local fallback"
        )
    if not os.access(external_mount, os.W_OK):
        raise RuntimeError(f"external storage is not writable: {external_mount}")
    external.mkdir(parents=True, exist_ok=True)
    return output, "external_pssd"
