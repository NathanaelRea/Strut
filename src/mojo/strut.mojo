from os import abort
from python import Python, PythonObject

from solver.run_case import run_case
from strut_io import load_json, load_pickle, parse_args, py_len


fn _strip_recorders_if_needed(mut data: PythonObject, compute_only: Bool) raises:
    if not compute_only:
        return
    var builtins = Python.import_module("builtins")
    data["recorders"] = builtins.list()


fn _load_case_data(
    input_path: String,
    input_pickle_path: String,
    compute_only: Bool,
) raises -> PythonObject:
    var sources = 0
    if input_path != "":
        sources += 1
    if input_pickle_path != "":
        sources += 1
    if sources > 1:
        abort("use only one of --input or --input-pickle")
    if input_pickle_path != "":
        var data = load_pickle(input_pickle_path)
        _strip_recorders_if_needed(data, compute_only)
        return data
    if input_path == "":
        abort("missing --input or --input-pickle")
    var data = load_json(input_path)
    _strip_recorders_if_needed(data, compute_only)
    return data


def main():
    var (
        input_path,
        input_pickle_path,
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
            var entry_input_pickle = String(entry.get("input_pickle", ""))
            var entry_output = String(entry.get("output", ""))
            var entry_profile = String(entry.get("profile", ""))
            if (entry_input == "" and entry_input_pickle == "") or entry_output == "":
                abort("batch entry missing input/output")
            var t_case_start = Int(time.perf_counter_ns())
            var t_load_start = Int(time.perf_counter_ns())
            var data = _load_case_data(
                entry_input,
                entry_input_pickle,
                compute_only,
            )
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

    if (
        output_path == ""
        or (input_path == "" and input_pickle_path == "")
    ):
        abort(
            "usage: strut.mojo -- "
            "(--input <case.json> | --input-pickle <case.pkl>) "
            "--output <dir> [--batch <manifest.json>] [--profile <speedscope.json>] [--compute-only]"
        )

    var time = Python.import_module("time")
    var t_load_start = Int(time.perf_counter_ns())
    var data = _load_case_data(input_path, input_pickle_path, compute_only)
    var t_load_end = Int(time.perf_counter_ns())
    var case_load_us = (t_load_end - t_load_start) // 1000
    run_case(data, output_path, profile_path, case_load_us)
