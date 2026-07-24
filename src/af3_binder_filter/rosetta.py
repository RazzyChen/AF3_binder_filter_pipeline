"""Rosetta InterfaceAnalyzer CLI adapter."""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from af3_binder_filter.config import RosettaSettings
from af3_binder_filter.io_utils import atomic_write_text

ROSETTA_OUTPUT_FIELDS = (
    "rosetta_dSASA_int",
    "rosetta_dSASA_polar",
    "rosetta_dSASA_hphobic",
    "rosetta_dG_separated",
    "rosetta_dG_separated_per_dSASA_x100",
    "rosetta_dG_cross",
    "rosetta_packstat",
    "rosetta_sc_value",
    "rosetta_hbonds_int",
    "rosetta_delta_unsat_hbonds",
    "rosetta_nres_int",
    "rosetta_per_residue_energy_int",
)


class InterfaceEnergyEngine(Protocol):
    def analyze(
        self,
        pdb_path: Path,
        *,
        output_dir: Path,
        log_dir: Path | None = None,
    ) -> dict[str, Any]: ...


def _bool(value: bool) -> str:
    return "true" if value else "false"


def build_rosetta_command(
    settings: RosettaSettings,
    *,
    pdb_path: Path,
    scorefile: Path,
) -> list[str]:
    command = [
        settings.binary,
        "-database",
        settings.database,
        "-s",
        str(pdb_path),
        "-interface",
        settings.interface,
        "-score:weights",
        settings.score_function,
        "-pack_input",
        _bool(settings.pack_input),
        "-pack_separated",
        _bool(settings.pack_separated),
        "-compute_packstat",
        _bool(settings.compute_packstat),
    ]
    if settings.constant_seed:
        command.extend(
            [
                "-constant_seed",
                "-jran",
                str(settings.random_seed),
            ]
        )
    command.extend(
        [
            "-out:file:score_only",
            str(scorefile),
            "-overwrite",
        ]
    )
    return command


def _parse_scorefile(path: Path) -> dict[str, float | str]:
    header: list[str] | None = None
    values: list[str] | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("SCORE:"):
            continue
        fields = line.split()[1:]
        if not fields:
            continue
        if "description" in fields:
            header = fields
        elif header is not None:
            values = fields
    if header is None or values is None:
        raise ValueError(f"Rosetta scorefile has no data row: {path}")
    row: dict[str, float | str] = {}
    for key, value in zip(header, values, strict=False):
        try:
            number = float(value)
            row[key] = number if math.isfinite(number) else value
        except ValueError:
            row[key] = value
    return row


ROSETTA_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "rosetta_dSASA_int": ("dSASA_int",),
    "rosetta_dSASA_polar": ("dSASA_polar",),
    "rosetta_dSASA_hphobic": ("dSASA_hphobic",),
    "rosetta_dG_separated": ("dG_separated",),
    "rosetta_dG_separated_per_dSASA_x100": (
        "dG_separated/dSASAx100",
        "dG_separated_per_dSASA_x100",
    ),
    "rosetta_dG_cross": ("dG_cross",),
    "rosetta_packstat": ("packstat",),
    "rosetta_sc_value": ("sc_value",),
    "rosetta_hbonds_int": ("hbonds_int",),
    "rosetta_delta_unsat_hbonds": ("delta_unsatHbonds", "delta_unsat_hbonds"),
    "rosetta_nres_int": ("nres_int",),
    "rosetta_per_residue_energy_int": ("per_residue_energy_int",),
}


@dataclass
class RosettaCliEngine:
    settings: RosettaSettings
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run

    def analyze(
        self,
        pdb_path: Path,
        *,
        output_dir: Path,
        log_dir: Path | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {field: None for field in ROSETTA_OUTPUT_FIELDS}
        result.update({"rosetta_status": "error", "rosetta_error": ""})
        if not pdb_path.is_file():
            result["rosetta_error"] = f"Rosetta input PDB does not exist: {pdb_path}"
            return result
        output_dir.mkdir(parents=True, exist_ok=True)
        effective_log_dir = log_dir or output_dir
        effective_log_dir.mkdir(parents=True, exist_ok=True)
        scorefile = output_dir / f"{pdb_path.stem}.score.sc"
        command = build_rosetta_command(self.settings, pdb_path=pdb_path, scorefile=scorefile)
        stdout_path = effective_log_dir / f"{pdb_path.stem}.stdout.log"
        stderr_path = effective_log_dir / f"{pdb_path.stem}.stderr.log"
        atomic_write_text(
            effective_log_dir / f"{pdb_path.stem}.command.txt",
            " ".join(command) + "\n",
        )
        try:
            completed = self.runner(
                command,
                capture_output=True,
                text=True,
                timeout=self.settings.timeout_seconds,
                check=False,
            )
            atomic_write_text(stdout_path, completed.stdout or "")
            atomic_write_text(stderr_path, completed.stderr or "")
            if completed.returncode != 0:
                result["rosetta_error"] = (
                    f"InterfaceAnalyzer returned {completed.returncode}: "
                    f"{(completed.stderr or '').strip()}"
                )
                return result
            raw = _parse_scorefile(scorefile)
            for output_field, aliases in ROSETTA_FIELD_ALIASES.items():
                result[output_field] = next(
                    (raw[alias] for alias in aliases if alias in raw),
                    None,
                )
            if (
                result["rosetta_dG_separated_per_dSASA_x100"] is None
                and isinstance(result["rosetta_dG_separated"], (int, float))
                and isinstance(result["rosetta_dSASA_int"], (int, float))
                and result["rosetta_dSASA_int"] != 0
            ):
                result["rosetta_dG_separated_per_dSASA_x100"] = (
                    result["rosetta_dG_separated"] / result["rosetta_dSASA_int"] * 100
                )
            if (
                result["rosetta_per_residue_energy_int"] is None
                and isinstance(result["rosetta_dG_separated"], (int, float))
                and isinstance(result["rosetta_nres_int"], (int, float))
                and result["rosetta_nres_int"] != 0
            ):
                result["rosetta_per_residue_energy_int"] = (
                    result["rosetta_dG_separated"] / result["rosetta_nres_int"]
                )
            result["rosetta_status"] = "success"
            return result
        except subprocess.TimeoutExpired:
            result["rosetta_status"] = "timeout"
            result["rosetta_error"] = (
                f"InterfaceAnalyzer exceeded {self.settings.timeout_seconds} seconds"
            )
            return result
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            result["rosetta_error"] = str(exc)
            return result
