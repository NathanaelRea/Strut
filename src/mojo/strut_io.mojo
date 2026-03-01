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
    var builtins = Python.import_module("builtins")
    var path_obj = pathlib.Path(path)
    var resolved = path_obj.resolve()
    var text = path_obj.read_text()
    var data = json.loads(text)
    if Bool(builtins.isinstance(data, builtins.dict)):
        data["__strut_case_json_path"] = PythonObject(String(resolved))
        data["__strut_case_dir"] = PythonObject(String(resolved.parent))
    return data


fn load_pickle(path: String) raises -> PythonObject:
    var pickle = Python.import_module("pickle")
    var pathlib = Python.import_module("pathlib")
    var builtins = Python.import_module("builtins")
    var path_obj = pathlib.Path(path)
    var resolved = path_obj.resolve()
    var file_obj = path_obj.open("rb")
    try:
        var data = pickle.load(file_obj)
        if Bool(builtins.isinstance(data, builtins.dict)):
            if not data.__contains__("__strut_case_json_path"):
                data["__strut_case_json_path"] = PythonObject(String(resolved))
            if not data.__contains__("__strut_case_dir"):
                data["__strut_case_dir"] = PythonObject(String(resolved.parent))
            return data
    finally:
        file_obj.close()
    return builtins.dict()


fn parse_args() -> (String, String, String, String, String, Bool):
    var args = argv()
    var input_path = arg_value(args, "--input")
    var input_pickle_path = arg_value(args, "--input-pickle")
    var output_path = arg_value(args, "--output")
    var batch_path = arg_value(args, "--batch")
    var profile_path = arg_value(args, "--profile")
    var compute_only = False
    for i in range(len(args)):
        if String(args[i]) == "--compute-only":
            compute_only = True
            break
    return (
        input_path,
        input_pickle_path,
        output_path,
        batch_path,
        profile_path,
        compute_only,
    )
