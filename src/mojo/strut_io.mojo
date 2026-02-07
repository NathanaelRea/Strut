from python import Python, PythonObject
from sys.arg import argv


fn arg_value(
    args: VariadicList[StringSlice[StaticConstantOrigin]], name: String
) -> String:
    for i in range(len(args) - 1):
        if String(args[i]) == name:
            return String(args[i + 1])
    return ""


fn py_len(obj: PythonObject) raises -> Int:
    return Int(obj.__len__())


fn load_json(path: String) raises -> PythonObject:
    var json = Python.import_module("json")
    var pathlib = Python.import_module("pathlib")
    var path_obj = pathlib.Path(path)
    var text = path_obj.read_text()
    return json.loads(text)


fn parse_args() -> (String, String, String, String):
    var args = argv()
    var input_path = arg_value(args, "--input")
    var output_path = arg_value(args, "--output")
    var batch_path = arg_value(args, "--batch")
    var profile_path = arg_value(args, "--profile")
    return (input_path, output_path, batch_path, profile_path)
