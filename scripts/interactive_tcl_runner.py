#!/usr/bin/env python3
"""Interactive / multi-run direct-Tcl runner backed by the Strut Mojo solver."""

from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import tkinter
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence, TextIO

import run_strut_case
import tcl_to_strut

QUERY_COMMANDS = (
    "analyze",
    "nodeDisp",
    "nodeVel",
    "nodeAccel",
    "nodeReaction",
    "eleResponse",
    "eleForce",
    "getTime",
    "eigen",
)

# These callbacks already have the command result when they return to Tcl.
# Capture it directly on the hot live path instead of adding a Tcl trace.
_LIVE_QUERY_HANDLERS = {
    "_cmd_analyze": "analyze",
    "_cmd_get_time": "getTime",
    "_cmd_node_disp": "nodeDisp",
}


def _load_last_row(path: Path) -> list[float]:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not lines:
        raise RuntimeError(f"solver output is empty: {path}")
    return [float(tok) for tok in lines[-1].split()]


def _load_all_rows(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append([float(tok) for tok in stripped.split()])
    return rows


def _stage_step_count(stage: dict[str, Any]) -> int:
    analysis = stage.get("analysis", {})
    if not isinstance(analysis, dict):
        return 1
    return max(1, int(analysis.get("steps", 1)))


class _SessionUnavailable(RuntimeError):
    pass


class _SolverSession:
    def __init__(self, solver: Path, work_dir: Path) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="strut_tcl_session_")
        directory = Path(self.directory.name)
        self.output = directory / "out"
        self.diagnostics = (directory / "stderr").open("w+", encoding="utf-8")
        self.process = None
        self.active = False
        try:
            self.process = subprocess.Popen(
                [str(solver), "--session"],
                cwd=work_dir,
                env=run_strut_case._solver_environment(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.diagnostics,
            )
        except BaseException:
            self.close()
            raise

    def open_domain(self, case: dict[str, Any], entry: Path) -> None:
        self.active = True
        reply = self._submit_case("open", case, entry)
        self.compact_repeat = reply.get("compact_repeat", False)
        self._repeat_reply: struct.Struct | None = None
        self._repeat_reply_dimensions = (0, 0)

    def solve(self, case: dict[str, Any], entry: Path) -> Path:
        self._submit_case("solve", case, entry)
        return self.output

    def _submit_case(
        self, op: str, case: dict[str, Any], entry: Path
    ) -> dict[str, Any]:
        if self.output.exists():
            shutil.rmtree(self.output)
        return self.request(
            {
                **case,
                "op": op,
                "source_path": str(entry),
                "source_dir": str(entry.parent),
                "output_path": str(self.output),
            }
        )

    def _receive(self) -> dict[str, Any]:
        assert self.process is not None and self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            self.process.wait()
            self.diagnostics.seek(0)
            raise _SessionUnavailable(
                self.diagnostics.read().strip() or "solver session exited"
            )
        try:
            reply = json.loads(line)
        except ValueError as exc:
            raise _SessionUnavailable(
                f"invalid solver session response {line.rstrip()!r}"
            ) from exc
        if "error" in reply:
            raise _SessionUnavailable(reply["error"])
        return reply

    def _send(self, request: bytes) -> None:
        assert self.process is not None and self.process.stdin is not None
        try:
            if request == b"r\n":
                os.write(self.process.stdin.fileno(), request)
            else:
                self.process.stdin.write(request)
                self.process.stdin.flush()
        except BrokenPipeError as exc:
            self.diagnostics.seek(0)
            raise _SessionUnavailable(
                self.diagnostics.read().strip() or "solver session exited"
            ) from exc

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._send((json.dumps(payload, allow_nan=False) + "\n").encode())
        return self._receive()

    def repeat_one(self, node_count: int, ndf: int) -> dict[str, Any]:
        assert self.process is not None and self.process.stdout is not None
        layout = self._repeat_reply
        if layout is None or self._repeat_reply_dimensions != (node_count, ndf):
            layout = self._repeat_reply = struct.Struct(f"={2 + 3 * ndf * node_count}d")
            self._repeat_reply_dimensions = (node_count, ndf)
        self._send(b"r\n")
        data = self.process.stdout.read(layout.size)
        if len(data) != layout.size:
            self.process.wait()
            self.diagnostics.seek(0)
            raise _SessionUnavailable(
                self.diagnostics.read().strip() or "truncated solver session reply"
            )
        words = layout.unpack(data)
        nodes = [
            (
                words[i : i + ndf],
                words[i + ndf : i + 2 * ndf],
                words[i + 2 * ndf : i + 3 * ndf],
            )
            for i in range(2, len(words), 3 * ndf)
        ]
        return {
            "status": int(words[0]),
            "time": words[1],
            "steps": int(words[0] == 0),
            "nodes": nodes,
        }

    def finish(self) -> Path:
        self.request({"op": "close"})
        self.active = False
        return self.output

    def discard(self) -> None:
        self.request({"op": "discard"})
        self.active = False

    def close(self) -> None:
        if self.process is not None:
            if self.active and self.process.poll() is None:
                self.process.kill()
            with suppress(BrokenPipeError):
                self.process.stdin.close()
            self.process.wait()
            self.process.stdout.close()
        self.diagnostics.close()
        self.directory.cleanup()


class InteractiveTclRunner(tcl_to_strut.TclStrutBuilder):
    """Executes multi-run / interactive OpenSees Tcl scripts via Strut's Mojo solver."""

    def __init__(
        self,
        entry_path: Path,
        repo_root: Path,
        solver_path: Path,
        work_dir: Path,
        replay_mode: str | None = None,
    ) -> None:
        self._solver_path = solver_path.resolve()
        self._work_dir = work_dir.resolve()
        self._envelope_specs: list[dict[str, Any]] = []
        self._cached_solve_dir: Path | None = None
        self._cached_solve_key: tuple[int, tuple[int, ...]] | None = None
        self._solve_generation = 0
        self._temp_solve_dirs: list[Path] = []
        self._replay_mode = replay_mode
        self._session_setting = os.getenv("STRUT_TCL_SESSION", "1")
        self._solver_process: _SolverSession | None = None
        self._session: _SolverSession | None = None
        self._session_disabled = replay_mode != "live" or self._session_setting == "0"
        self._session_constraints: str | None = None
        self._session_recorders: list[dict[str, Any]] | None = []
        self._session_series: dict[int, dict[str, Any]] = {}
        self._session_series_source: list[dict[str, Any]] | None = None
        self._session_static_integrator: dict[str, Any] | None = None
        self._session_hs_load: tuple[Any, Any] | None = None
        self._session_can_repeat = False
        self._session_transient_dt: str | None = None
        self._session_next_nodes: set[int] = set()
        self._session_node_values: dict[int, Sequence[Sequence[float]]] = {}
        self._session_observed_nodes: list[int] = []
        self._session_ndf = 0
        self._domain_number = 0
        self.output_recorders: dict[str, dict[str, Any]] = {}
        self._step_dir: tempfile.TemporaryDirectory | None = None
        self._step_result: Path | None = None
        self._step_replay: tuple[dict[str, Any], int] | None = None
        self._pending_hs_retry: tuple[dict[str, Any], int, Path] | None = None
        self._failed_load_control: tuple[dict[str, Any], float, int] | None = None
        self._applying_load_control_retry = False
        self._capture_node_state = False
        self._state_step = -1
        self._state_rows: dict[Path, list[list[float]]] = {}
        self._query_streams: dict[str, TextIO] = {}
        self._reported_transient_time: float | None = None
        self._reaction_values: dict[int, list[float]] = {}
        super().__init__(entry_path, repo_root)
        self._setup_interactive_commands()

    def _wrap_command(self, fn, *, read_only: bool = False):
        wrapped = super()._wrap_command(fn)
        if self._replay_mode == "live" and (
            query := _LIVE_QUERY_HANDLERS.get(fn.__name__)
        ):

            def capture(*args: str):
                result = wrapped(*args)
                self._write_query_result(query, result)
                return result

            return capture
        # By default, commands may change the analysis, domain or recorders.
        # Read-only queries can run while a transient driver waits for more steps.
        if read_only or fn.__name__ in {
            "_cmd_analyze",
            "_cmd_get_time",
            "_cmd_node_disp",
            "_cmd_node_reaction",
            "_cmd_reactions",
            "_cmd_ele_response",
            "_cmd_ele_force",
            "_cmd_clock",
        }:
            return wrapped

        def invalidate(*args: str):
            self._session_transient_dt = None
            return wrapped(*args)

        return invalidate

    def _restore_tcl_builtins(self, names: tuple[str, ...]) -> None:
        for name in names:
            disabled = f"::strut::_disabled_{name}"
            exists = str(
                self.interp.tk.call("::strut::_info_builtin", "commands", disabled)
            )
            if exists:
                self.interp.tk.call("::strut::_rename_builtin", disabled, name)

    def _setup_interactive_commands(self) -> None:
        self._restore_tcl_builtins(
            (
                "regexp",
                "regsub",
                "linsert",
                "lreplace",
                "lsearch",
                "lrange",
                "upvar",
            )
        )
        self.interp.eval("proc rename {old new} { ::strut::_rename_builtin $old $new }")
        self.interp.createcommand("getPID", self._wrap_command(self._cmd_get_pid))
        self.interp.createcommand("getNP", self._wrap_command(self._cmd_get_np))
        self.interp.createcommand(
            "eleResponse", self._wrap_command(self._cmd_ele_response)
        )
        self.interp.createcommand(
            "setParameter", self._wrap_command(self._cmd_set_parameter)
        )
        self.interp.createcommand(
            "file", self._wrap_command(self._cmd_interactive_file)
        )
        if self._replay_mode:
            commands = {
                "nodeVel": lambda *args: self._cmd_node_response("velocity", args),
                "nodeAccel": lambda *args: self._cmd_node_response(
                    "acceleration", args
                ),
                "eleForce": self._cmd_ele_force,
                "reactions": self._cmd_reactions,
            }
            for name, command in commands.items():
                self.interp.createcommand(
                    name, self._wrap_command(command, read_only=True)
                )
            self.interp.createcommand("::strut::_record_query", self._record_query)
            self.interp.createcommand(
                "::strut::_domain_changed", self._wrap_command(self._domain_changed)
            )
            for name in (
                "model",
                "node",
                "mass",
                "fix",
                "fixX",
                "fixY",
                "fixZ",
                "element",
                "geomTransf",
                "section",
                "uniaxialMaterial",
                "nDMaterial",
                "equalDOF",
                "rigidDiaphragm",
                "timeSeries",
                "remove",
                "setParameter",
                "updateParameter",
                "sp",
            ):
                self.interp.tk.call(
                    "::strut::_disabled_trace",
                    "add",
                    "execution",
                    name,
                    "enter",
                    "::strut::_domain_changed",
                )
            for name in QUERY_COMMANDS:
                if (
                    self._replay_mode == "live"
                    and name in _LIVE_QUERY_HANDLERS.values()
                ) or (name == "analyze" and self._replay_mode != "live"):
                    continue
                self.interp.tk.call(
                    "::strut::_disabled_trace",
                    "add",
                    "execution",
                    name,
                    "leave",
                    ("::strut::_record_query", name),
                )

    def _record_query(
        self, name: str, command: str, code: str, result: str, op: str
    ) -> None:
        if code == "0":
            self._write_query_result(name, result)

    def _write_query_result(self, name: str, result: str) -> None:
        path = f"__strut_query_{name}.out"
        if path not in self._query_streams:
            self._query_streams[path] = (self._work_dir / path).open(
                "w", encoding="utf-8"
            )
            self.output_recorders[path] = {"category": "raw_table"}
        self._query_streams[path].write(" ".join(result.split()) + "\n")

    def _domain_changed(self, command: str, op: str) -> str:
        name, *args = self.interp.tk.splitlist(command)
        if self.analysis_stages and not (
            name == "timeSeries"
            or (name == "remove" and args and args[0] in {"recorders", "loadPattern"})
        ):
            raise self._error(
                "changing this model after analyze requires a new domain",
                name,
                tuple(args),
            )
        self._step_result = None
        self._step_replay = None
        return ""

    def _invalidate_solution(self) -> None:
        self._solve_generation += 1
        self._cached_solve_key = None

    def _cmd_get_pid(self, *args: str) -> str:
        if args:
            raise self._error("getPID takes no arguments", "getPID", args)
        return "0"

    def _cmd_get_np(self, *args: str) -> str:
        if args:
            raise self._error("getNP takes no arguments", "getNP", args)
        return "1"

    def _cmd_interactive_file(self, *args: str) -> str:
        if len(args) >= 2 and args[0] == "delete":
            for raw_target in args[1:]:
                if raw_target == "-force":
                    continue
                resolved = self._resolve_io_path(raw_target, "file", args)
                if resolved.exists():
                    if resolved.is_dir():
                        shutil.rmtree(resolved)
                    else:
                        resolved.unlink()
            return ""
        if len(args) == 2 and args[0] == "exists":
            resolved = self._resolve_io_path(args[1], "file", args)
            return "1" if resolved.exists() else "0"
        return super()._cmd_file(*args)

    def _cmd_recorder(self, *args: str) -> str:
        if self._replay_mode:
            start = len(self.recorders)
            result = super()._cmd_recorder(*args)
            options, _ = (
                self._parse_recorder_flags(args[1:])
                if start < len(self.recorders)
                else ({}, [])
            )
            # Recorder stems alone can collide across directories or lifetimes.
            for i in range(start, len(self.recorders)):
                self.recorders[i]["output"] = f"__tcl_recorder_{i}"
                if "timeSeries" in options:
                    self.recorders[i]["time_series"] = int(options["timeSeries"])
            return result
        if args and args[0] == "EnvelopeNode":
            options, positional = self._parse_recorder_flags(args[1:])
            if not positional:
                raise self._error(
                    "EnvelopeNode recorder response is required", "recorder", args
                )
            raw_file = options.get("file")
            nodes = options.get("node", [])
            dofs = options.get("dof", [])
            kind = positional[0]
            if not raw_file or not nodes or not dofs:
                raise self._error(
                    "EnvelopeNode recorder requires -file, -node, and -dof",
                    "recorder",
                    args,
                )
            if kind not in {"disp", "accel"}:
                raise self._error(
                    f"unsupported EnvelopeNode response `{kind}`", "recorder", args
                )
            self._envelope_specs.append(
                {
                    "file": str(raw_file),
                    "nodes": [int(n) for n in nodes],
                    "dofs": [int(d) for d in dofs],
                    "kind": kind,
                    "include_time": bool(options.get("time", False)),
                    "start_stage": len(self.analysis_stages),
                }
            )
            self._invalidate_solution()
            return ""
        res = super()._cmd_recorder(*args)
        self._invalidate_solution()
        return res

    def _cmd_analyze(self, *args: str) -> str:
        if self._replay_mode:
            steps = int(args[0]) if args else -1
            if steps == 0:
                return "0"
            if self._replay_mode == "live":
                return self._run_live_analyze(args, steps)
            self._step_result = None
            return self._append_replay_steps(args)
        res = super()._cmd_analyze(*args)
        self._invalidate_solution()
        return res

    def _run_live_analyze(self, args: tuple[str, ...], steps: int) -> str:
        if (
            len(args) == 2
            and args[1] == self._session_transient_dt
            and self._session_can_repeat
            and steps > 0
        ):
            response = self._session_analyze_request(
                {"op": "repeat", "steps": steps}, compact=steps == 1
            )
            if response is not None:
                self.last_stage["analysis"]["steps"] += response["steps"]
                self._commit_session_time(args, response)
                return str(response["status"])

        # An unsuccessful trial must not enter the replay history. Other solver
        # errors (e.g. an unsupported model) still surface as errors to Tcl.
        fields = (
            "current_time",
            "node_displacements",
            "pending_time_reset",
            "pending_initialize",
            "load_control_has_run",
            "pattern_removed",
            "uniform_excitation_count_since_analyze",
            "permeability_updates",
            "has_complex_transient_excitation",
        )
        previous = deepcopy({name: getattr(self, name) for name in fields})
        previous_stages = len(self.analysis_stages)
        previous_last_steps = (
            self.last_stage["analysis"]["steps"] if self.last_stage else 0
        )
        failed_retry = self._failed_load_control
        self._append_replay_steps(args)
        merged = len(self.analysis_stages) == previous_stages
        if self.last_stage is None or (
            merged and self.last_stage["analysis"]["steps"] == previous_last_steps
        ):
            return "0"
        session_result = self._analyze_in_session(
            args, merged, previous["current_time"], previous["load_control_has_run"]
        )
        if session_result is not None:
            status = session_result["status"]
            committed_steps = session_result["steps"]
            if status == -3:
                self._restore_analysis(previous, previous_stages, previous_last_steps)
                if committed_steps:
                    self._failed_load_control = failed_retry
                    self._append_replay_steps((str(committed_steps), *args[1:]))
                if session_result["increment"]:
                    self._failed_load_control = (
                        self.integrator,
                        session_result["increment"],
                        session_result["iterations"],
                    )
            if (
                status == 0
                and len(args) == 2
                and not self.last_stage.get("initialize")
                and self.last_stage.get("pattern", {}).get("type") != "None"
            ):
                self._session_transient_dt = args[1]
            else:
                self._session_transient_dt = None
            self._commit_session_time(args, session_result, previous["current_time"])
            return str(status)
        if self._replay_transient_stage(args, previous["current_time"]):
            return "0"
        if self._accept_hs_retry(args, previous_stages):
            return "0"
        hs_result = self._replay_hs_constraint_stage(args)
        if hs_result == "0":
            return "0"
        if hs_result is None:
            out, result = self._solve_replay(self._node_state_recorders())
            if result.returncode == 0:
                self._step_result = out
                self._state_step = -1
                self._update_committed_time(out)
                self._advance_reported_time(args, previous["current_time"])
                return "0"
            if "did not converge" not in result.stdout + result.stderr:
                self._raise_solver_error(result)
        committed_steps = 0
        if steps > 1:
            out, failed = self._solve_replay([], progress=True)
            if "did not converge" not in failed.stdout + failed.stderr:
                self._raise_solver_error(failed)
            progress = json.loads((out / "run_progress.json").read_text())
            failed_stage = progress["stage_number"] - 1
            steps_before = (
                sum(map(_stage_step_count, self.analysis_stages[: previous_stages - 1]))
                + previous_last_steps
                if previous_stages
                else 0
            )
            committed_steps = (
                sum(map(_stage_step_count, self.analysis_stages[:failed_stage]))
                + progress["step_number"]
                - 1
                - steps_before
            )
            if not 0 <= committed_steps < int(args[0]):
                self._raise_solver_error(result)
        failed_load_control = None
        if hs_result is None:
            failure_path = out / "__strut_load_control_failure.out"
            if failure_path.exists():
                increment, iterations = _load_last_row(failure_path)
                failed_load_control = (self.integrator, increment, int(iterations))
        self._restore_analysis(previous, previous_stages, previous_last_steps)
        if committed_steps:
            self._append_replay_steps((str(committed_steps), *args[1:]))
            out, committed = self._solve_replay(self._node_state_recorders())
            if committed.returncode:
                self._raise_solver_error(committed)
            self._step_result = out
            self._update_committed_time(out)
            self._advance_reported_time(
                (str(committed_steps), *args[1:]), previous["current_time"]
            )
        self._failed_load_control = failed_load_control
        return "-3"

    def _commit_session_time(
        self,
        args: tuple[str, ...],
        response: dict[str, Any],
        previous_time: float | None = None,
    ) -> None:
        if response["steps"]:
            self._advance_reported_time(
                (str(response["steps"]), *args[1:]),
                self.current_time if previous_time is None else previous_time,
            )
        self.current_time = response["time"]

    def _restore_analysis(
        self, previous: dict[str, Any], stage_count: int, last_steps: int
    ) -> None:
        for name, value in previous.items():
            setattr(self, name, value)
        del self.analysis_stages[stage_count:]
        self.last_stage = self.analysis_stages[-1] if self.analysis_stages else None
        if self.last_stage is not None:
            self.last_stage["analysis"]["steps"] = last_steps

    def _close_session(self) -> None:
        if self._solver_process is not None:
            self._solver_process.close()
            self._solver_process = None
        self._session = None

    def _session_worker(self) -> _SolverSession:
        if self._solver_process is None:
            self._solver_process = _SolverSession(self._solver_path, self._work_dir)
        return self._solver_process

    def _disable_session(self, reason: str, *, broken: bool = False) -> None:
        if broken:
            self._close_session()
        elif self._session is not None:
            try:
                self._session.discard()
            except _SessionUnavailable:
                self._close_session()
            self._session = None
        self._session_disabled = True
        self._session_can_repeat = False
        self._session_transient_dt = None
        self._session_node_values.clear()
        if self._session_setting == "required":
            raise _SessionUnavailable(reason)
        if os.getenv("STRUT_VERBOSE") == "1":
            print(f"Strut live Tcl replay fallback: {reason}", file=sys.stderr)

    def _session_request(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        assert self._session is not None
        try:
            return self._session.request(payload)
        except _SessionUnavailable as exc:
            self._disable_session(str(exc), broken=True)
            return None

    def _session_recorder_payload(
        self, recorders: list[dict[str, Any]]
    ) -> dict[str, Any]:
        assert self.model is not None
        return {
            "model": self.model,
            "nodes": [],
            "elements": [],
            "recorders": recorders,
        }

    def _session_analyze_request(
        self, payload: dict[str, Any], *, compact: bool = False
    ) -> dict[str, Any] | None:
        observed = sorted(self._session_next_nodes)
        self._session_next_nodes.clear()
        session = self._session
        assert session is not None
        if (
            compact
            and session.compact_repeat
            and observed == self._session_observed_nodes
        ):
            try:
                response = session.repeat_one(len(observed), self._session_ndf)
            except (OSError, _SessionUnavailable) as exc:
                self._disable_session(str(exc), broken=True)
                response = None
        else:
            if observed:
                payload["nodes_to_observe"] = observed
            response = self._session_request(payload)
        self._session_observed_nodes = observed
        self._session_can_repeat = response is not None and response["status"] == 0
        if response is not None:
            self._session_node_values = dict(
                zip(observed, response.get("nodes", []), strict=True)
            )
        return response

    def _analyze_in_session(
        self, args: tuple[str, ...], merged: bool, start_time: float, resume: bool
    ) -> dict[str, Any] | None:
        if self._session_disabled:
            return None
        integrator = self.integrator["type"]
        static_control = integrator in {"DisplacementControl", "HSConstraint"}
        resume_integrator = self.integrator is self._session_static_integrator
        if (
            merged
            and self._session_can_repeat
            and self.time_series is self._session_series_source
            and (not static_control or resume_integrator)
        ):
            return self._session_analyze_request(
                {
                    "op": "repeat",
                    "steps": int(args[0]),
                    "time": start_time,
                    "resume": resume or self.integrator["type"] != "LoadControl",
                }
            )
        assert self.last_stage is not None
        stage = dict(self.last_stage)
        stage["analysis"] = analysis = {
            **stage["analysis"],
            "steps": int(args[0]),
            "start_time": start_time,
        }
        stage_idx = len(self.analysis_stages) - 1
        active_recorders = [
            {k: v for k, v in recorder.items() if k not in {"start_stage", "end_stage"}}
            for recorder in self.recorders
            if recorder.get("start_stage", 0)
            <= stage_idx
            < recorder.get("end_stage", math.inf)
        ]
        constraints = self._analysis_constraints_name()
        pattern = stage.get("pattern", {})
        if (
            integrator
            not in {"LoadControl", "DisplacementControl", "HSConstraint", "Newmark"}
            or constraints == "Penalty"
            or (
                self._session_constraints is not None
                and self._session_constraints != constraints
            )
            or analysis.get("support_motions")
            or analysis.get("modal_damping")
            or analysis.get("solver_chain")
            or pattern.get("sp_constraints")
            or pattern.get("vel0")
            or any(
                excitation.get("vel0") for excitation in pattern.get("excitations", [])
            )
            or any(
                stage.get(key)
                for key in (
                    "sample_before",
                    "material_stages",
                    "permeability_updates",
                    "prescribed_pressures",
                    "remove_nodes",
                    "remove_elements",
                )
            )
            or "initial_state_reset" in stage
            or "contact_friction" in stage
            or any(
                rec.get("delta_t") and integrator != "Newmark"
                for rec in active_recorders
            )
        ):
            self._disable_session(f"unsupported stage settings for {integrator}")
            return None

        if (
            self._session_recorders is not None
            and active_recorders != self._session_recorders
            and any(rec.get("delta_t") for rec in self._session_recorders)
        ):
            self._disable_session("changed sampled recorder set")
            return None
        if any(
            self.time_series_by_tag.get(tag) is not original
            for tag, original in self._session_series.items()
        ):
            self._disable_session("time-series tag reassigned")
            return None

        if self._session is None:
            case = self._build_current_case_dict([])
            run_strut_case._normalize_time_series_paths(
                case, self.entry_path, self.repo_root
            )
            run_strut_case._lower_one_dimensional_case(case)
            try:
                worker = self._session_worker()
                worker.open_domain(case, self.entry_path)
                self._session = worker
            except (OSError, _SessionUnavailable) as exc:
                self._disable_session(str(exc), broken=True)
                return None
            self._session_constraints = constraints
            self._session_series = dict(self.time_series_by_tag)
            self._session_ndf = case["model"]["ndf"]

        if merged and integrator == "LoadControl":
            stage["resume_load_control"] = resume
        new_series = [
            deepcopy(series)
            for series in self.time_series
            if series["tag"] not in self._session_series
        ]
        if new_series:
            stage["time_series"] = new_series
            run_strut_case._normalize_time_series_paths(
                {"time_series": new_series}, self.entry_path, self.repo_root
            )
        payload = {"op": "analyze", "stage": stage}
        if static_control:
            payload["resume_integrator"] = resume_integrator
        hs_load = None
        if integrator == "HSConstraint":
            hs_load = (stage.get("pattern"), stage.get("loads"))
            if resume_integrator and hs_load != self._session_hs_load:
                payload["refresh_hs_reference"] = True
        if active_recorders != self._session_recorders:
            payload.update(self._session_recorder_payload(active_recorders))
        response = self._session_analyze_request(payload)
        if response is not None:
            self._session_static_integrator = (
                self.integrator if static_control else None
            )
            self._session_hs_load = hs_load
            self._session_recorders = active_recorders
            self._session_series = dict(self.time_series_by_tag)
            self._session_series_source = self.time_series
        return response

    def _replay_transient_stage(self, args: tuple[str, ...], start_time: float) -> bool:
        stage = self.analysis_stages[-1]
        analysis = stage["analysis"]
        if not analysis["type"].startswith("transient") or analysis.get("solver_chain"):
            return False
        steps = analysis["steps"]
        max_samples = (
            2_000_000 // (3 * sum(self.node_ndfs.values()))
            if self._capture_node_state
            else 500_000
        )
        if steps >= max_samples:
            return False
        if self._step_replay is not None:
            cached_stage, horizon = self._step_replay
            if stage is cached_stage and steps <= horizon:
                self._state_step = steps - 1
                self._advance_reported_time(args, start_time)
                return True

        horizon = min(max_samples, max(128, 2 * steps))
        analysis["steps"] = horizon
        try:
            out, result = self._solve_replay(self._node_state_recorders())
        finally:
            analysis["steps"] = steps
        if result.returncode:
            # An error in a speculative future step cannot affect this call.
            return False
        self._step_replay = (stage, horizon)
        self._step_result = out
        self._state_step = steps - 1
        self._advance_reported_time(args, start_time)
        return True

    def _hs_solver_attempt(self, analysis: dict[str, Any]) -> dict[str, Any]:
        attempt = self._solver_chain_primary_attempt(analysis)
        if attempt["algorithm"] == "ModifiedNewtonInitial" and attempt.get(
            "algorithm_options"
        ) == {"raw_args": ["-initial"], "initial": True}:
            del attempt["algorithm_options"]
        return attempt

    def _accept_hs_retry(self, args: tuple[str, ...], previous_stages: int) -> bool:
        pending, self._pending_hs_retry = self._pending_hs_retry, None
        if pending is None or args != ("1",) or self._step_replay is None:
            return False
        stage, steps, out = pending
        candidate = self.analysis_stages[-1]
        if (
            self._step_replay[0] is not stage
            or len(self.analysis_stages) != previous_stages + 1
            or candidate["analysis"].get("integrator")
            != stage["analysis"]["integrator"]
            or self._hs_solver_attempt(candidate["analysis"])
            != stage["live_hs_attempts"][1]
        ):
            return False
        # Accept a cached retry only if Tcl selected the same model and settings.
        ignored = {"analysis", "live_hs_attempts"}
        if {k: v for k, v in stage.items() if k not in ignored} != {
            k: v for k, v in candidate.items() if k not in ignored
        }:
            return False
        solver_keys = {"system", "system_options", "constraints", "numberer"}
        if any(
            stage["analysis"].get(k) != candidate["analysis"].get(k)
            for k in solver_keys
        ):
            return False
        self.analysis_stages.pop()
        if not self.analysis_stages or self.analysis_stages[-1] is not stage:
            self.analysis_stages.append(stage)
        self.last_stage = stage
        stage["analysis"]["steps"] = steps
        self._step_result = out
        self._state_step = steps - 1
        self._update_committed_time(out)
        return True

    def _replay_hs_constraint_stage(self, args: tuple[str, ...]) -> str | None:
        stage = self.analysis_stages[-1]
        analysis = stage["analysis"]
        if (
            args != ("1",)
            or analysis.get("integrator", {}).get("type") != "HSConstraint"
            or not self._capture_node_state
        ):
            return None

        steps = analysis["steps"]
        if self._step_replay is not None:
            cached_stage, horizon = self._step_replay
            if stage is cached_stage and steps <= horizon:
                assert self._step_result is not None
                return self._read_hs_step(stage, steps, self._step_result)

        if steps == 1 and "live_hs_attempts" not in stage:
            policy = super()._active_step_retry_policy(args)
            if (
                policy
                and len(policy.attempts) == 1
                and not policy.continue_after_failure
            ):
                primary = self._hs_solver_attempt(analysis)
                retry = self._solver_chain_retry_attempt(analysis, policy.attempts[0])
                if primary != retry:
                    stage["live_hs_attempts"] = [primary, retry]

        # Run a window using the detected retry settings. Tcl still receives a
        # failure when the primary algorithm fails and chooses whether to retry.
        horizon = max(128, 2 * steps)
        recorders = self._node_state_recorders()
        try:
            analysis["steps"] = horizon
            out, result = self._solve_replay(recorders, hs_window=True)
            if result.returncode:
                if (
                    "did not converge (HSConstraint)"
                    not in result.stdout + result.stderr
                ):
                    return None
                progress = json.loads((out / "run_progress.json").read_text())
                if progress["stage_number"] != len(self.analysis_stages):
                    return None
                horizon = progress["step_number"] - 1
                if steps > horizon:
                    return None
                analysis["steps"] = horizon
                out, result = self._solve_replay(recorders, hs_window=True)
                if result.returncode:
                    return None
        finally:
            analysis["steps"] = steps
        self._step_replay = (stage, horizon)
        return self._read_hs_step(stage, steps, out)

    def _read_hs_step(self, stage: dict[str, Any], steps: int, out: Path) -> str:
        path = out / "__strut_hs_window_times.out"
        if path not in self._state_rows:
            self._state_rows[path] = _load_all_rows(path)
        if self._state_rows[path][steps - 1][1] != 0:
            self._pending_hs_retry = (stage, steps, out)
            self._step_result = out if steps > 1 else None
            self._state_step = steps - 2
            return "-3"
        self._step_result = out
        self._state_step = steps - 1
        self._update_committed_time(out)
        return "0"

    def _advance_reported_time(
        self, args: tuple[str, ...], previous_time: float
    ) -> None:
        if len(args) != 2:
            self._reported_transient_time = None
            return
        value = self._reported_transient_time
        if value is None:
            value = previous_time
        for _ in range(int(args[0])):
            value += float(args[1])
        self._reported_transient_time = value

    def _cmd_get_time(self, *args: str) -> str:
        if not self._replay_mode or len(args) > 1:
            return super()._cmd_get_time(*args)
        value = self._reported_transient_time
        if value is None:
            value = self.current_time
        if args:
            return str(self.interp.tk.call("format", args[0], value))
        return f"{value:f}"

    def _cmd_wipe_analysis(self, *args: str) -> str:
        self._session_can_repeat = False
        return super()._cmd_wipe_analysis(*args)

    def _cmd_set_time(self, *args: str) -> str:
        self._session_can_repeat = False
        self._reported_transient_time = None
        self._step_replay = None
        self._pending_hs_retry = None
        return super()._cmd_set_time(*args)

    def _cmd_load_const(self, *args: str) -> str:
        self._reported_transient_time = None
        self._step_replay = None
        if self._replay_mode == "live" and not args:
            args = ("-time", str(self.current_time))
        result = super()._cmd_load_const(*args)
        if self._session is not None and "load_const" in self.last_stage:
            self._session_can_repeat = False
            self._session_request({"op": "loadConst", "time": self.current_time})
        return result

    def _can_merge_stage(self, stage: dict[str, Any]) -> bool:
        analysis = stage["analysis"]
        if self._replay_mode == "live" and analysis["type"] == "static_nonlinear":
            # A live Tcl script chooses its own retry algorithm and test.
            analysis.setdefault("solver_chain", [])
        if self._applying_load_control_retry or (
            self._replay_mode
            and self.last_stage is not None
            and self.last_stage.get("load_const")
        ):
            return False
        return super()._can_merge_stage(stage)

    def _active_step_retry_policy(
        self, analyze_args: tuple[str, ...]
    ) -> tcl_to_strut.StepRetryPolicy | None:
        if self._replay_mode == "live":
            return None
        return super()._active_step_retry_policy(analyze_args)

    def _append_replay_steps(self, args: tuple[str, ...]) -> str:
        previous_time = self.current_time
        previous_stages = len(self.analysis_stages)
        failed, self._failed_load_control = self._failed_load_control, None
        if failed is not None and self.integrator is failed[0]:
            integrator, increment, iterations = failed
            target_iters = integrator.get("num_iter", 1)
            retry_increment = increment * target_iters / max(1, iterations)
            self.integrator = {
                **integrator,
                "step": retry_increment,
                "min_step": integrator.get("min_step", integrator["step"]),
                "max_step": integrator.get("max_step", integrator["step"]),
            }
            self.load_control_has_run = False
            self._applying_load_control_retry = True
            try:
                result = super()._cmd_analyze(*args)
            finally:
                self.integrator = integrator
                self._applying_load_control_retry = False
        else:
            result = super()._cmd_analyze(*args)
        if len(self.analysis_stages) > previous_stages:
            analysis = self.analysis_stages[-1]["analysis"]
            integrator = analysis.get("integrator", {}).get("type")
            if integrator in {"LoadControl", "DisplacementControl", "HSConstraint"}:
                analysis.setdefault("start_time", previous_time)
        if len(args) == 2:
            analysis = self.analysis_stages[-1]["analysis"]
            self.current_time = (
                analysis["start_time"] + analysis["steps"] * analysis["dt"]
            )
        return result

    def _node_state_recorders(self) -> list[dict[str, Any]]:
        if not self._capture_node_state:
            return []
        assert self.model is not None
        kinds = ("displacement",)
        if self.analysis_stages[-1]["analysis"]["type"].startswith("transient"):
            kinds += ("velocity", "acceleration")
        return [
            {
                "type": f"node_{kind}",
                "nodes": sorted(self.nodes),
                "dofs": list(range(1, self.model["ndf"] + 1)),
                "output": f"__tcl_{kind}",
                "start_stage": len(self.analysis_stages) - 1,
            }
            for kind in kinds
        ]

    def _solve_replay(
        self,
        recorders: list[dict[str, Any]],
        *,
        progress: bool = False,
        hs_window: bool = False,
    ) -> tuple[Path, subprocess.CompletedProcess[str]]:
        if self._step_dir is None:
            self._step_dir = tempfile.TemporaryDirectory(prefix="strut_tcl_replay_")
        solve_dir = Path(self._step_dir.name)
        if self._step_result == solve_dir / "out":
            self._step_result = None
        self._step_replay = None
        self._pending_hs_retry = None
        self._state_step = -1
        self._state_rows.clear()
        out = solve_dir / "out"
        shutil.rmtree(out, ignore_errors=True)
        case = self._build_current_case_dict(recorders)
        run_strut_case._normalize_time_series_paths(
            case, self.entry_path, self.repo_root
        )
        run_strut_case._lower_one_dimensional_case(case)
        input_path = solve_dir / "case.json"
        input_path.write_text(json.dumps(case), encoding="utf-8")
        mode = "hs-window" if hs_window else "progress" if progress else "1"
        result = subprocess.run(
            [str(self._solver_path), "--input", str(input_path), "--output", str(out)],
            cwd=self._work_dir,
            env={
                **run_strut_case._solver_environment(),
                "STRUT_TCL_REPLAY": mode,
                "STRUT_TCL_LOAD_CONTROL_FAILURE": str(
                    out / "__strut_load_control_failure.out"
                ),
            },
            capture_output=True,
            check=False,
            text=True,
        )
        return out, result

    @staticmethod
    def _raise_solver_error(result: subprocess.CompletedProcess[str]) -> None:
        details = (result.stdout + "\n" + result.stderr).strip()
        raise RuntimeError(f"Strut replay failed ({result.returncode}): {details}")

    def _update_committed_time(self, out: Path) -> None:
        if self._state_step >= 0:
            time_path = out / "__strut_hs_window_times.out"
            if time_path not in self._state_rows:
                self._state_rows[time_path] = _load_all_rows(time_path)
            self.current_time = self._state_rows[time_path][self._state_step][0]
            return
        time_path = out / "__strut_domain_time"
        if (
            not self.analysis_stages[-1]["analysis"]["type"].startswith("transient")
            and time_path.exists()
        ):
            self.current_time = float(time_path.read_text())

    def _cmd_set_parameter(self, *args: str) -> str:
        val: float | None = None
        ele_tags: list[int] | None = None
        param_name: str | None = None
        idx = 0
        while idx < len(args):
            tok = args[idx]
            if tok == "-value":
                val = float(args[idx + 1])
                idx += 2
            elif tok == "-ele":
                idx += 1
                tags: list[int] = []
                while idx < len(args):
                    try:
                        tags.append(int(args[idx]))
                        idx += 1
                    except ValueError:
                        break
                ele_tags = tags if tags else None
                if idx < len(args) and not args[idx].startswith("-"):
                    param_name = args[idx]
                    idx += 1
            elif tok == "-eleRange":
                start = int(args[idx + 1])
                end = int(args[idx + 2])
                ele_tags = list(range(start, end + 1))
                idx += 3
                if idx < len(args) and not args[idx].startswith("-"):
                    param_name = args[idx]
                    idx += 1
            else:
                param_name = tok
                idx += 1

        if val is None or param_name is None:
            raise self._error(
                "setParameter requires -value and parameter name", "setParameter", args
            )
        updated = 0
        for elem in self.elements:
            if ele_tags is not None and int(elem["id"]) not in ele_tags:
                continue
            if elem.get("type") in {"truss", "corotTruss"}:
                if param_name == "A":
                    elem["area"] = float(val)
                    updated += 1
                elif param_name == "rho":
                    elem["rho"] = float(val)
                    updated += 1
        if updated == 0:
            raise self._error(
                f"setParameter matched no elements for `{param_name}`",
                "setParameter",
                args,
            )
        # For linear static constant-load re-solves (e.g., CallingMatlab.tcl),
        # replace the previous linear static stage so the next `analyze 1`
        # solves K(param_new) * u = P cleanly.
        if self.analysis_stages and all(
            stage.get("analysis", {}).get("type") == "static_linear"
            and not stage.get("load_const")
            for stage in self.analysis_stages
        ):
            self.analysis_stages.clear()
            self.last_stage = None
            self.current_time = 0.0
        self._invalidate_solution()
        return ""

    def _build_current_case_dict(
        self, extra_recorders: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("model must be defined before solving")
        if not self.analysis_stages:
            raise RuntimeError("no analysis stages defined before solving")
        stages = [
            {
                **{k: v for k, v in stage.items() if k != "live_hs_attempts"},
                "analysis": {
                    **stage["analysis"],
                    "solver_chain": stage["live_hs_attempts"],
                },
            }
            if "live_hs_attempts" in stage
            else stage
            for stage in self.analysis_stages
        ]
        case_dict = self._build_case(
            stages,
            self._analysis_constraints_name(),
            extra_recorders,
            equal_dof_as_mp=True,
        ).to_json_dict()
        if self.current_rayleigh is not None:
            case_dict["rayleigh"] = dict(self.current_rayleigh)
        return case_dict

    def _ensure_solved(
        self,
        *,
        query_elements: tuple[int, ...] = (),
        include_envelopes: bool = False,
    ) -> Path:
        key = (self._solve_generation, tuple(sorted(set(query_elements))))
        if (
            not include_envelopes
            and self._cached_solve_dir is not None
            and self._cached_solve_key == key
        ):
            return self._cached_solve_dir

        assert self.model is not None
        ndf = int(self.model["ndf"])
        all_nodes = sorted(self.nodes)
        all_dofs = list(range(1, ndf + 1))

        extra_recorders: list[dict[str, Any]] = [
            {
                "type": "node_displacement",
                "nodes": all_nodes,
                "dofs": all_dofs,
                "output": "__interactive_disp",
            }
        ]
        if any(spec["kind"] == "accel" for spec in self._envelope_specs):
            extra_recorders.append(
                {
                    "type": "node_acceleration",
                    "nodes": all_nodes,
                    "dofs": all_dofs,
                    "output": "__interactive_accel",
                }
            )
        if query_elements:
            extra_recorders.append(
                {
                    "type": "element_force",
                    "elements": sorted(set(query_elements)),
                    "output": "__interactive_eforce",
                }
            )
            deformable = [
                int(elem["id"])
                for elem in self.elements
                if int(elem["id"]) in query_elements
                and elem["type"] in {"forceBeamColumn2d", "dispBeamColumn2d"}
            ]
            if deformable:
                extra_recorders.append(
                    {
                        "type": "element_deformation",
                        "elements": deformable,
                        "output": "__interactive_def",
                    }
                )

        case_dict = self._build_current_case_dict(extra_recorders)
        solve_dir = Path(tempfile.mkdtemp(prefix="strut_interactive_solve_"))
        self._temp_solve_dirs.append(solve_dir)
        case_json = solve_dir / "case.json"
        out_dir = solve_dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        run_strut_case._normalize_time_series_paths(
            case_dict, self.entry_path, self.repo_root
        )
        case_json.write_text(json.dumps(case_dict), encoding="utf-8")

        subprocess.run(
            [
                str(self._solver_path),
                "--input",
                str(case_json),
                "--output",
                str(out_dir),
            ],
            cwd=self._work_dir,
            env=run_strut_case._solver_environment(),
            check=True,
        )
        self._cached_solve_dir = out_dir
        self._cached_solve_key = key
        return out_dir

    def _flush_envelope_recorders(self) -> None:
        if not self._envelope_specs or not self.analysis_stages or self.model is None:
            return
        out_dir = self._ensure_solved(include_envelopes=True)
        ndf = int(self.model["ndf"])
        stage_steps = [_stage_step_count(stage) for stage in self.analysis_stages]

        for spec in self._envelope_specs:
            start_stage = int(spec["start_stage"])
            skip_rows = sum(stage_steps[:start_stage])
            prefix = (
                "__interactive_disp"
                if spec["kind"] == "disp"
                else "__interactive_accel"
            )
            nodes: list[int] = spec["nodes"]
            dofs: list[int] = spec["dofs"]

            per_node_rows: list[list[list[float]]] = []
            for node_id in nodes:
                node_file = out_dir / f"{prefix}_node{node_id}.out"
                all_rows = _load_all_rows(node_file)
                sliced = all_rows[skip_rows:]
                if not sliced:
                    raise RuntimeError(
                        f"EnvelopeNode recorder {spec['file']} has no steps after stage {start_stage}"
                    )
                per_node_rows.append(sliced)

            num_steps = min(len(rows) for rows in per_node_rows)
            total_cols = len(nodes) * len(dofs)
            mins = [math.inf] * total_cols
            maxs = [-math.inf] * total_cols
            abss = [0.0] * total_cols

            for step_idx in range(num_steps):
                col = 0
                for node_rows in per_node_rows:
                    row = node_rows[step_idx]
                    for dof in dofs:
                        if dof < 1 or dof > ndf:
                            val = 0.0
                        else:
                            val = row[dof - 1]
                        if val < mins[col]:
                            mins[col] = val
                        if val > maxs[col]:
                            maxs[col] = val
                        aval = abs(val)
                        if aval > abss[col]:
                            abss[col] = aval
                        col += 1

            out_path = self._resolve_io_path(spec["file"], "recorder", ())
            out_path.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                " ".join(f"{v:.15g}" for v in mins),
                " ".join(f"{v:.15g}" for v in maxs),
                " ".join(f"{v:.15g}" for v in abss),
            ]
            out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _cmd_node_disp(self, *args: str) -> str:
        if self._replay_mode:
            return self._cmd_node_response("displacement", args)
        if not args:
            raise self._error("nodeDisp expects a node id", "nodeDisp", args)
        node_id = int(args[0])
        model = self._require_model("nodeDisp", args)
        if not self.analysis_stages:
            return super()._cmd_node_disp(*args)
        out_dir = self._ensure_solved()
        disp = _load_last_row(out_dir / f"__interactive_disp_node{node_id}.out")
        if len(args) == 1:
            return " ".join(f"{disp[i]:.15g}" for i in range(int(model["ndf"])))
        dof = int(args[1])
        return f"{disp[dof - 1]:.15g}"

    def _cmd_node_response(self, kind: str, args: tuple[str, ...]) -> str:
        if not 1 <= len(args) <= 2:
            raise ValueError("node query requires an existing node and optional DOF")
        node = int(args[0])
        if node not in self.nodes:
            raise ValueError("node query requires an existing node and optional DOF")
        dof = int(args[1]) if len(args) == 2 else 0
        ndf = self.node_ndfs[node]
        if dof > ndf:
            raise ValueError(f"node {node} has {ndf} DOFs")
        if kind != "reaction" and self._session is not None:
            self._capture_node_state = True
            self._session_next_nodes.add(node)
            node_values = self._session_node_values.get(node)
            if node_values is None:
                response = self._session_request({"op": "node", "node": node})
                if response is not None:
                    node_values = self._session_node_values[node] = response["values"]
            if node_values is not None:
                component = ("displacement", "velocity", "acceleration").index(kind)
                values = node_values[component]
                if dof > 0:
                    return f"{values[dof - 1]:.17g}"
                return " ".join(f"{value:.17g}" for value in values[:ndf])
        values = [0.0] * ndf
        if kind == "reaction":
            values = self._reaction_values.get(node, values)
        elif self.analysis_stages:
            if not self._capture_node_state:
                self._capture_node_state = True
                self._step_result = None
            if self._step_result is None:
                out, result = self._solve_replay(self._node_state_recorders())
                if result.returncode:
                    self._raise_solver_error(result)
                self._step_result = out
                if not self.pending_time_reset and not self.analysis_stages[-1].get(
                    "load_const"
                ):
                    self._update_committed_time(out)
            path = self._step_result / f"__tcl_{kind}_node{node}.out"
            if path.exists():
                if self._state_step == -1:
                    values = _load_last_row(path)[:ndf]
                else:
                    if path not in self._state_rows:
                        self._state_rows[path] = _load_all_rows(path)
                    values = self._state_rows[path][self._state_step][:ndf]
        elif kind == "displacement":
            values = [
                self.node_displacements.get((node, d), 0.0) for d in range(1, ndf + 1)
            ]
        return " ".join(
            f"{value:.17g}" for value in (values[dof - 1 : dof] if dof > 0 else values)
        )

    def _cmd_node_reaction(self, *args: str) -> str:
        if self._replay_mode:
            return self._cmd_node_response("reaction", args)
        return super()._cmd_node_reaction(*args)

    def _cmd_reactions(self, *args: str) -> str:
        if args:
            raise ValueError(
                "reactions with inertia or Rayleigh damping is unsupported"
            )
        self._reaction_values.clear()
        if self._session is not None:
            response = self._session_request({"op": "reactions"})
            if response is not None:
                forces = response["values"]
                self._reaction_values = {
                    node: forces[
                        i * self._session_ndf : i * self._session_ndf
                        + self.node_ndfs[node]
                    ]
                    for i, node in enumerate(sorted(self.nodes))
                }
                return ""
        if self.analysis_stages:
            recorder = {
                "type": "node_reaction",
                "nodes": sorted(self.nodes),
                "dofs": list(range(1, self.model["ndf"] + 1)),
                "output": "__tcl_reaction",
                "start_stage": len(self.analysis_stages) - 1,
            }
            out, result = self._solve_replay([recorder])
            if result.returncode:
                self._raise_solver_error(result)
            self._reaction_values = {
                node: _load_last_row(out / f"__tcl_reaction_node{node}.out")
                for node in self.nodes
            }
        return ""

    def _cmd_ele_force(self, *args: str) -> str:
        if not 1 <= len(args) <= 2:
            raise ValueError("eleForce expects an element and optional DOF")
        values = self._query_element(int(args[0]), ("forces",))
        if len(args) == 2:
            dof = int(args[1])
            if dof > len(values):
                raise ValueError("eleForce DOF out of range")
            if dof > 0:
                values = [values[dof - 1]]
        return " ".join(f"{value:.17g}" for value in values)

    def _query_element(self, tag: int, response: tuple[str, ...]) -> list[float]:
        from run_case import _normalized_recorder_outputs

        kind = response[0]
        element = next(e for e in self.elements if e["id"] == tag)
        recorder = {"elements": [tag], "output": "__tcl_element"}
        if kind in {"forces", "force", "globalForce", "globalForces"}:
            recorder["type"] = "element_force"
        elif kind in {"localForce", "localForces"}:
            recorder["type"] = "element_local_force"
        elif (kind == "axialForce" or response == ("material", "stress")) and element[
            "type"
        ] in {"truss", "corotTruss"}:
            recorder.update(type="element_basic_force", truss_response=kind)
        elif kind in {"basicForce", "basicForces"}:
            recorder["type"] = "element_basic_force"
        elif kind in {
            "basicDeformation",
            "basicDeformations",
            "basicDisp",
            "basicDisplacement",
        }:
            recorder["type"] = "element_deformation"
        elif kind == "section" and len(response) == 3:
            section_response = response[2].rstrip("s")
            if section_response not in {"force", "deformation"}:
                raise ValueError(f"unsupported section response {response[2]}")
            recorder.update(
                type=f"section_{section_response}", section=int(response[1])
            )
        else:
            raise ValueError(f"unsupported eleResponse {response}")

        if not self.analysis_stages:
            if recorder.get("truss_response"):
                return [0.0]
            if recorder["type"] in {"element_force", "element_local_force"}:
                return [0.0] * (2 * self.model["ndf"])
            if element["type"] in {
                "elasticBeamColumn2d",
                "forceBeamColumn2d",
                "dispBeamColumn2d",
            }:
                return [0.0] * (2 if kind == "section" else 3)
            raise ValueError(
                f"unsupported initial {kind} response for {element['type']}"
            )
        if self._session is not None:
            result = self._session_request(
                {
                    "op": "element",
                    "element": tag,
                    "kind": recorder["type"],
                    "section": recorder.get("section", 1),
                    "material_stress": recorder.get("truss_response") == "material",
                }
            )
            if result is not None:
                values = result["values"]
                if self.model["ndm"] == 1 and recorder["type"] == "element_force":
                    values = values[::2]
                return values
        recorder["start_stage"] = len(self.analysis_stages) - 1
        out, result = self._solve_replay([recorder])
        if result.returncode:
            self._raise_solver_error(result)
        (filename,) = _normalized_recorder_outputs(recorder)
        values = _load_last_row(out / filename)
        if self.model["ndm"] == 1 and recorder["type"] == "element_force":
            values = values[::2]
        return values

    def _cmd_ele_response(self, *args: str) -> str:
        if len(args) < 2:
            raise self._error(
                "eleResponse expects `eleTag arg1 ...`", "eleResponse", args
            )
        ele_id = int(args[0])
        if self._replay_mode:
            return " ".join(
                f"{value:.17g}" for value in self._query_element(ele_id, args[1:])
            )
        kind = args[1]
        elem = next((e for e in self.elements if int(e["id"]) == ele_id), None)
        if elem is None:
            raise self._error(f"element {ele_id} not found", "eleResponse", args)
        out_dir = self._ensure_solved(query_elements=(ele_id,))

        if kind in {"forces", "force", "globalForce", "globalForces"}:
            vals = _load_last_row(out_dir / f"__interactive_eforce_ele{ele_id}.out")
            return " ".join(f"{v:.15g}" for v in vals)

        if elem["type"] in {"forceBeamColumn2d", "dispBeamColumn2d"} and kind in {
            "basicDeformation",
            "basicDeformations",
            "basicDisp",
            "basicDisplacement",
        }:
            vals = _load_last_row(out_dir / f"__interactive_def_ele{ele_id}.out")
            return " ".join(f"{v:.15g}" for v in vals)

        raise self._error(f"unsupported eleResponse `{kind}`", "eleResponse", args)

    def _solve_eigenvalues(
        self, num_modes: int, *, largest: bool = False, standard: bool = False
    ) -> list[float]:
        if not self._replay_mode or not self.analysis_stages:
            return super()._solve_eigenvalues(
                num_modes, largest=largest, standard=standard
            )
        if largest and not standard:
            raise ValueError("largest eigenvalues after analyze are unsupported")
        if not standard and any(elem.get("cMass") for elem in self.elements):
            raise ValueError("consistent element mass after analyze is unsupported")
        node = min(self.nodes)
        recorder = {
            "type": "modal_eigen",
            "nodes": [node],
            "dofs": [1],
            "modes": [1],
            "output": "__tcl_modal",
            "start_stage": len(self.analysis_stages),
        }
        stage = {
            "analysis": {
                "type": "modal_eigen",
                "steps": 1,
                "num_modes": num_modes,
                "standard_eigen": standard,
                "constraints": self._analysis_constraints_name(),
            }
        }
        if self._session is not None:
            response = self._session_request(
                {
                    "op": "eigen",
                    "stage": stage,
                    **self._session_recorder_payload(
                        [
                            {
                                **recorder,
                                "output": "__strut_session_eigen",
                                "start_stage": 0,
                            }
                        ]
                    ),
                }
            )
            if response is not None:
                self._session_can_repeat = False
                if self._session_recorders and any(
                    rec.get("delta_t") for rec in self._session_recorders
                ):
                    self._disable_session("eigen with sampled recorders")
                self._session_recorders = None
                return response["values"]
        self.analysis_stages.append(stage)
        try:
            out, result = self._solve_replay([recorder])
            if result.returncode:
                self._raise_solver_error(result)
            return [
                float(value)
                for value in (out / "__tcl_modal_eigenvalues.out").read_text().split()
            ]
        finally:
            self.analysis_stages.pop()

    def _run_mojo_eigen_case(self, case: dict[str, Any]) -> list[float]:
        if self._session_disabled:
            return super()._run_mojo_eigen_case(case)
        # Eigen before analyze uses its own setup because subsequent Tcl
        # commands can still change the model before the first live step.
        run_strut_case._normalize_time_series_paths(
            case, self.entry_path, self.repo_root
        )
        run_strut_case._lower_one_dimensional_case(case)
        try:
            out = self._session_worker().solve(case, self.entry_path)
        except (OSError, _SessionUnavailable) as exc:
            self._disable_session(str(exc), broken=True)
            return super()._run_mojo_eigen_case(case)
        return [
            float(value)
            for value in (out / "modal_eigenvalues.out").read_text().split()
        ]

    def _cmd_wipe(self, *args: str) -> str:
        if args and not (len(args) == 1 and args[0].lower() == "all"):
            raise self._error("wipe takes no arguments", "wipe", args)
        if self._replay_mode:
            self._finish_domain()
            self._session_disabled = (
                self._replay_mode != "live" or self._session_setting == "0"
            )
            self._session_constraints = None
            self._session_recorders = []
            self._session_series.clear()
            self._session_series_source = None
            self._session_static_integrator = None
            self._session_hs_load = None
            self._session_can_repeat = False
            self._session_next_nodes.clear()
            self._session_observed_nodes.clear()
            self._session_node_values.clear()
            self._step_result = None
            self._step_replay = None
            self._capture_node_state = False
            self._failed_load_control = None
            self._pending_hs_retry = None
            self._reported_transient_time = None
            self._reaction_values.clear()
            self.last_eigenvalues.clear()
        else:
            self._flush_envelope_recorders()
        # Reset full model/analysis state so multi-run loops start fresh.
        self.analysis_stages.clear()
        self.last_stage = None
        self._envelope_specs.clear()
        self.uniform_excitation_count_since_analyze = 0
        self.pattern_removed = False
        self.has_complex_transient_excitation = False
        self._invalidate_solution()
        return super()._cmd_wipe(*args)

    def _finish_domain(self) -> None:
        self._domain_number += 1
        out = None
        if self._session is not None:
            try:
                out = self._session.finish()
                self._session = None
            except _SessionUnavailable as exc:
                self._disable_session(str(exc), broken=True)
        if not self.analysis_stages or not self.recorders:
            return
        if out is None:
            out, result = self._solve_replay(self.recorders)
            if result.returncode:
                self._raise_solver_error(result)
        from run_case import _normalized_recorder_outputs

        grouped = {}
        for recorder in self.recorders:
            grouped.setdefault(recorder["raw_path"], []).append(recorder)
        for raw_path, recorders in grouped.items():
            columns = []
            for recorder in recorders:
                for name in _normalized_recorder_outputs(recorder):
                    path = out / name
                    rows = _load_all_rows(path) if path.exists() else []
                    if self.model["ndm"] == 1 and recorder["type"] == "element_force":
                        rows = [row[::2] for row in rows]
                    columns.append(rows)
            destination = self._resolve_io_path(raw_path, "recorder", ())
            relative = destination.relative_to(self._work_dir)
            snapshot = Path("__strut_domains") / str(self._domain_number) / relative
            snapshot_path = self._work_dir / snapshot
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for row in zip(*columns, strict=True):
                lines.append(
                    " ".join(f"{value:.17g}" for values in row for value in values)
                )
            snapshot_path.write_text(
                "".join(line + "\n" for line in lines), encoding="utf-8"
            )
            self.output_recorders[str(snapshot)] = {
                "category": recorders[0]["type"],
                "include_time": recorders[0]["include_time"],
            }

    def run_script(self) -> None:
        try:
            try:
                self.interp.eval(f"source {{{self.entry_path}}}")
            except tcl_to_strut.TclToStrutError:
                raise
            except tkinter.TclError as exc:
                if self.last_error is not None:
                    raise self.last_error from exc
                location = self._current_location("eval", [])
                raise tcl_to_strut.TclToStrutError(str(exc), location) from exc
            if self._replay_mode:
                self._finish_domain()
            else:
                self._flush_envelope_recorders()
        finally:
            self._close_session()
            for stream in self._query_streams.values():
                stream.close()
            if self._step_dir is not None:
                self._step_dir.cleanup()
            for d in self._temp_solve_dirs:
                shutil.rmtree(d, ignore_errors=True)
            self._temp_solve_dirs.clear()


def run_interactive_tcl_case(
    entry_tcl: Path,
    output_dir: Path,
    repo_root: Path,
    solver_path: Path,
    replay_mode: str | None = None,
) -> None:
    entry_tcl = entry_tcl.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    work_root = Path(tempfile.mkdtemp(prefix="strut_interactive_work_"))
    try:
        work_dir = work_root / entry_tcl.parent.name
        shutil.copytree(entry_tcl.parent, work_dir)
        preexisting = {
            p.relative_to(work_dir): p.stat().st_mtime_ns
            for p in work_dir.rglob("*.out")
            if p.is_file()
        }

        runner = InteractiveTclRunner(
            work_dir / entry_tcl.name,
            repo_root=repo_root,
            solver_path=solver_path,
            work_dir=work_dir,
            replay_mode=replay_mode,
        )
        runner.run_script()
        if replay_mode:
            for name in runner.output_recorders:
                dest = output_dir / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(work_dir / name, dest)
            (output_dir / ".direct_tcl_recorders.json").write_text(
                json.dumps(runner.output_recorders, indent=2) + "\n", encoding="utf-8"
            )
            return

        for produced in sorted(work_dir.rglob("*.out")):
            if not produced.is_file():
                continue
            rel = produced.relative_to(work_dir)
            mtime_ns = produced.stat().st_mtime_ns
            if rel in preexisting and mtime_ns == preexisting[rel]:
                continue
            dest = output_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(produced, dest)
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
