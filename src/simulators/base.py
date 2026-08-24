from __future__ import annotations

import shutil
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SimStatus:
    backend: str
    available: bool
    message: str


@dataclass(frozen=True)
class SimLaunch:
    backend: str
    launched: bool
    message: str
    command: list[str] | None = None


def check_backend(backend: str) -> SimStatus:
    backend = backend.lower().strip()
    if backend == "local":
        return SimStatus(backend=backend, available=True, message="Local python simulation")

    if backend == "jsbsim":
        try:
            import jsbsim  # type: ignore
        except Exception:
            return SimStatus(
                backend=backend,
                available=False,
                message="jsbsim package not found. Install with: pip install jsbsim",
            )
        return SimStatus(backend=backend, available=True, message="JSBSim available")

    if backend == "flightgear":
        fgfs = shutil.which("fgfs")
        if not fgfs:
            return SimStatus(
                backend=backend,
                available=False,
                message="FlightGear executable (fgfs) not found in PATH",
            )
        return SimStatus(backend=backend, available=True, message=f"FlightGear detected: {fgfs}")

    return SimStatus(backend=backend, available=False, message="Unknown backend")


def _load_command(config_path: str | Path) -> list[str]:
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    command = payload.get("command")
    args = payload.get("args") or []
    if not command or not isinstance(args, Iterable):
        raise ValueError("sim config must include 'command' and optional 'args' list")
    return [str(command), *[str(a) for a in args]]


def launch_backend(backend: str, config_path: str | Path | None, launch: bool = False) -> SimLaunch:
    backend = backend.lower().strip()
    if backend == "local":
        return SimLaunch(backend=backend, launched=True, message="Local python simulation", command=None)

    if not config_path:
        return SimLaunch(
            backend=backend,
            launched=False,
            message="External simulator selected but no sim config provided",
            command=None,
        )

    command = _load_command(config_path)
    if not launch:
        return SimLaunch(
            backend=backend,
            launched=False,
            message="External simulator config loaded (launch disabled)",
            command=command,
        )

    subprocess.Popen(command)
    return SimLaunch(
        backend=backend,
        launched=True,
        message="External simulator launched",
        command=command,
    )
