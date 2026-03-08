from python import Python, PythonObject
from sys import is_defined

alias PROFILE_FRAME_TOTAL: Int = 0
alias PROFILE_FRAME_ASSEMBLE: Int = 1
alias PROFILE_FRAME_SOLVE: Int = 2
alias PROFILE_FRAME_OUTPUT: Int = 3
alias PROFILE_FRAME_ASSEMBLE_STIFFNESS: Int = 4
alias PROFILE_FRAME_KFF_EXTRACT: Int = 5
alias PROFILE_FRAME_SOLVE_LINEAR: Int = 6
alias PROFILE_FRAME_SOLVE_NONLINEAR: Int = 7
alias PROFILE_FRAME_NONLINEAR_STEP: Int = 8
alias PROFILE_FRAME_NONLINEAR_ITER: Int = 9
alias PROFILE_FRAME_TIME_SERIES_EVAL: Int = 10
alias PROFILE_FRAME_CONSTRAINTS: Int = 11
alias PROFILE_FRAME_RECORDERS: Int = 12
alias PROFILE_FRAME_FACTORIZE: Int = 13
alias PROFILE_FRAME_TRANSIENT_STEP: Int = 14
alias PROFILE_FRAME_CASE_LOAD_PARSE: Int = 15
alias PROFILE_FRAME_MODEL_BUILD_DOF_MAP: Int = 16
alias PROFILE_FRAME_ASSEMBLE_UNIAXIAL: Int = 17
alias PROFILE_FRAME_ASSEMBLE_FIBER: Int = 18
alias PROFILE_FRAME_UNIAXIAL_REVERT_ALL: Int = 19
alias PROFILE_FRAME_UNIAXIAL_COMMIT_ALL: Int = 20
alias PROFILE_FRAME_ASSEMBLE_FIBER_GEOMETRY: Int = 21
alias PROFILE_FRAME_ASSEMBLE_FIBER_SECTION_RESPONSE: Int = 22
alias PROFILE_FRAME_ASSEMBLE_FIBER_MATRIX_SCATTER: Int = 23
alias PROFILE_FRAME_ASSEMBLE_FIBER_INTERNAL_FORCE: Int = 24
alias PROFILE_FRAME_UNIAXIAL_TRIAL_UPDATE: Int = 25
alias PROFILE_FRAME_UNIAXIAL_COPY_RESET: Int = 26


fn _profile_enabled(profile_path: String) -> Bool:
    @parameter
    if is_defined["STRUT_PROFILE"]():
        return profile_path != ""
    else:
        return False


fn _append_frame(mut frames: String, mut need_comma: Bool, name: String):
    if need_comma:
        frames += ","
    frames += "{\"name\":\"" + name + "\"}"
    need_comma = True


fn _append_event(
    mut events: String, mut need_comma: Bool, event_type: String, frame: Int, at_us: Int
):
    if need_comma:
        events += ","
    events += (
        "{\"type\":\""
        + event_type
        + "\",\"frame\":"
        + String(frame)
        + ",\"at\":"
        + String(at_us)
        + "}"
    )
    need_comma = True


def _write_speedscope(profile_path: String, frames: String, events: String, total_us: Int):
    var json = String()
    json += "{"
    json += "\"$schema\":\"https://www.speedscope.app/file-format-schema.json\","
    json += "\"shared\":{\"frames\":[" + frames + "]},"
    json += "\"profiles\":[{"
    json += "\"type\":\"evented\","
    json += "\"name\":\"strut\","
    json += "\"unit\":\"microseconds\","
    json += "\"startValue\":0,"
    json += "\"endValue\":" + String(total_us) + ","
    json += "\"events\":[" + events + "]"
    json += "}]}" + "\n"

    var pathlib = Python.import_module("pathlib")
    var out_path = pathlib.Path(profile_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(PythonObject(json))
