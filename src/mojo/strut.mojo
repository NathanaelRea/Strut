from os import abort
from python import Python, PythonObject

from solver.run_case import run_case
from strut_io import load_json, parse_args, py_len


def main():
    var (input_path, output_path, batch_path, profile_path) = parse_args()
    if batch_path != "":
        var batch = load_json(batch_path)
        var cases = batch.get("cases", [])
        var time = Python.import_module("time")
        var pathlib = Python.import_module("pathlib")
        for i in range(py_len(cases)):
            var entry = cases[i]
            var entry_input = String(entry.get("input", ""))
            var entry_output = String(entry.get("output", ""))
            var entry_profile = String(entry.get("profile", ""))
            if entry_input == "" or entry_output == "":
                abort("batch entry missing input/output")
            var t_case_start = Int(time.perf_counter_ns())
            var t_load_start = Int(time.perf_counter_ns())
            var data = load_json(entry_input)
            var t_load_end = Int(time.perf_counter_ns())
            var case_load_us = (t_load_end - t_load_start) // 1000
            run_case(data, entry_output, entry_profile, case_load_us)
            var t_case_end = Int(time.perf_counter_ns())
            var elapsed_us = (t_case_end - t_case_start) // 1000
            var out_dir = pathlib.Path(entry_output)
            out_dir.mkdir(parents=True, exist_ok=True)
            var file_path = out_dir.joinpath("case_time_us.txt")
            var line = String(elapsed_us) + "\n"
            file_path.write_text(PythonObject(line))
        return

    if input_path == "" or output_path == "":
        abort(
            "usage: strut.mojo -- --input <case.json> --output <dir> "
            "[--batch <manifest.json>] [--profile <speedscope.json>]"
        )

    var time = Python.import_module("time")
    var t_load_start = Int(time.perf_counter_ns())
    var data = load_json(input_path)
    var t_load_end = Int(time.perf_counter_ns())
    var case_load_us = (t_load_end - t_load_start) // 1000
    run_case(data, output_path, profile_path, case_load_us)
