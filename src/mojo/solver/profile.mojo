from python import Python, PythonObject
from sys import is_defined


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
