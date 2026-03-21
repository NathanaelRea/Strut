from os import abort
from python import Python, PythonObject

from solver.run_case import run_case_from_native_source
from strut_io import (
    load_case_document,
    load_json,
    parse_args,
    py_len,
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
    ) = parse_args()
    if batch_path != "":
        var batch = load_json(batch_path)
        var cases = batch.get("cases", [])
        var time = Python.import_module("time")
        var pathlib = Python.import_module("pathlib")
        for i in range(py_len(cases)):
            var entry = cases[i]
            var entry_input = String(entry.get("input", ""))
            var entry_input_bin = String(entry.get("input_bin", ""))
            var entry_output = String(entry.get("output", ""))
            var entry_profile = String(entry.get("profile", ""))
            if (entry_input == "" and entry_input_bin == "") or entry_output == "":
                abort("batch entry missing input/output")
            var t_case_start = Int(time.perf_counter_ns())
            var t_load_start = Int(time.perf_counter_ns())
            var loaded = load_case_document(entry_input, entry_input_bin, "")
            var t_load_end = Int(time.perf_counter_ns())
            var case_load_us = (t_load_end - t_load_start) // 1000
            run_case_from_native_source(
                loaded.doc,
                loaded.source_info,
                entry_output,
                entry_profile,
                case_load_us,
                not compute_only,
            )
            var t_case_end = Int(time.perf_counter_ns())
            var elapsed_us = (t_case_end - t_case_start) // 1000
            var out_dir = pathlib.Path(entry_output)
            out_dir.mkdir(parents=True, exist_ok=True)
            var file_path = out_dir.joinpath("case_time_us.txt")
            var line = String(elapsed_us) + "\n"
            file_path.write_text(PythonObject(line))
        return

    if output_path == "":
        if write_input_bin_path != "":
            _ = load_case_document(input_path, input_bin_path, write_input_bin_path)
            return
        abort(
            "usage: strut.mojo -- "
            "(--input <case.json> | --input-bin <case.bin>) "
            "--output <dir> [--batch <manifest.json>] [--profile <speedscope.json>] [--compute-only] "
            "[--write-input-bin <case.bin>]"
        )
    var time = Python.import_module("time")
    var t_load_start = Int(time.perf_counter_ns())
    var loaded = load_case_document(
        input_path, input_bin_path, write_input_bin_path
    )
    var t_load_end = Int(time.perf_counter_ns())
    var case_load_us = (t_load_end - t_load_start) // 1000
    run_case_from_native_source(
        loaded.doc,
        loaded.source_info,
        output_path,
        profile_path,
        case_load_us,
        not compute_only,
    )
