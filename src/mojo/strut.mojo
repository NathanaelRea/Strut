from os import abort
from pathlib import Path
from time import perf_counter_ns

from json_native import JsonValueTag
from solver.run_case import run_case_from_native_source
from solver.run_case.main import run_case_session
from solver.run_case.session import SessionProtocol
from solver.run_case.input_types import _json_get_string
from strut_io import (
    load_case_document,
    load_json_native,
    NativeCaseDocument,
    parse_args,
)


def main():
    var (
        input_path,
        input_bin_path,
        write_input_bin_path,
        output_path,
        batch_path,
        profile_path,
        compute_only,
        session,
    ) = parse_args()
    if session and (batch_path != "" or compute_only or profile_path != ""):
        abort("--session does not support batch, profiling or compute-only")
    if session and input_path == "" and input_bin_path == "":
        if output_path != "" or write_input_bin_path != "":
            abort("session outputs are set by each open command")
        run_case_session()
        return
    if batch_path != "":
        var batch = load_json_native(batch_path)
        var cases = batch.object_find(batch.root_index, "cases")
        if cases < 0:
            return
        if batch.node_tag(cases) != JsonValueTag.Array:
            abort("batch cases must be an array")
        for i in range(batch.node_len(cases)):
            var entry = batch.array_item(cases, i)
            var entry_input = _json_get_string(batch, entry, "input", "")
            var entry_input_bin = _json_get_string(batch, entry, "input_bin", "")
            var entry_output = _json_get_string(batch, entry, "output", "")
            var entry_profile = _json_get_string(batch, entry, "profile", "")
            if (entry_input == "" and entry_input_bin == "") or entry_output == "":
                abort("batch entry missing input/output")
            var t_case_start = Int(perf_counter_ns())
            var loaded = load_case_document(entry_input, entry_input_bin, "")
            var case_load_us = (Int(perf_counter_ns()) - t_case_start) // 1000
            run_case_from_native_source(
                loaded.doc,
                loaded.source_info,
                entry_output,
                entry_profile,
                case_load_us,
                not compute_only,
            )
            var elapsed_us = (Int(perf_counter_ns()) - t_case_start) // 1000
            (Path(entry_output) / "case_time_us.txt").write_text(String(elapsed_us) + "\n")
        return

    if output_path == "":
        if write_input_bin_path != "":
            _ = load_case_document(input_path, input_bin_path, write_input_bin_path)
            return
        abort(
            "usage: strut.mojo -- (--input <case.json> | --input-bin <case.bin>) --output <dir>"
            " [--batch <manifest.json>] [--profile <speedscope.json>] [--compute-only]"
            " [--write-input-bin <case.bin>] [--session]"
        )
    var t_load_start = Int(perf_counter_ns())
    var loaded: NativeCaseDocument
    if input_path == "-":
        if session or input_bin_path != "" or write_input_bin_path != "":
            abort("stdin input does not support session or cache options")
        var protocol = SessionProtocol(True)
        var line = protocol.read_line()
        if line == "":
            abort("missing JSON input on stdin")
        loaded = NativeCaseDocument()
        loaded.doc = protocol.decode(line)
    else:
        loaded = load_case_document(input_path, input_bin_path, write_input_bin_path)
    var case_load_us = (Int(perf_counter_ns()) - t_load_start) // 1000
    run_case_from_native_source(
        loaded.doc,
        loaded.source_info,
        output_path,
        profile_path,
        case_load_us,
        not compute_only,
        session,
    )
