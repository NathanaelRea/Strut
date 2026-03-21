from os import abort
from pathlib import Path
from python import Python, PythonObject
from sys.arg import argv

from json_native import (
    JsonDocument,
    JsonNode,
    JsonObjectEntry,
    JsonValueTag,
    load_json_native as load_json_document_native,
)


struct CaseSourceInfo(Movable, ImplicitlyCopyable):
    var json_path: String
    var case_dir: String

    fn __init__(out self):
        self.json_path = ""
        self.case_dir = ""

    fn __init__(out self, json_path: String, case_dir: String):
        self.json_path = json_path
        self.case_dir = case_dir


struct NativeCaseDocument(Movable):
    var doc: JsonDocument
    var source_info: CaseSourceInfo

    fn __init__(out self):
        self.doc = JsonDocument()
        self.source_info = CaseSourceInfo()


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


fn read_text_native(path: String) raises -> String:
    return Path(path).read_text()


fn load_json_native(path: String) raises -> JsonDocument:
    return load_json_document_native(path)


fn _append_cache_block(mut out: String, value: String):
    out += String(len(value.as_bytes()))
    out += "\n"
    out += value
    out += "\n"


fn _read_cache_line(text: String, position: Int) -> (String, Int):
    var bytes = text.as_bytes()
    var start = position
    var end = position
    while end < len(bytes) and bytes[end] != Byte(ord("\n")):
        end += 1
    if end >= len(bytes):
        abort("invalid native input cache: truncated line")
    return (
        String(StringSlice(unsafe_from_utf8=bytes[start:end])),
        end + 1,
    )


fn _read_cache_block(text: String, position: Int, block_len: Int) -> (String, Int):
    var bytes = text.as_bytes()
    if block_len < 0:
        abort("invalid native input cache: negative block length")
    var block_start = position
    var block_end = block_start + block_len
    if block_end > len(bytes):
        abort("invalid native input cache: truncated block")
    if block_end >= len(bytes) or bytes[block_end] != Byte(ord("\n")):
        abort("invalid native input cache: missing block terminator")
    return (
        String(StringSlice(unsafe_from_utf8=bytes[block_start:block_end])),
        block_end + 1,
    )


fn _next_tab_field(line: String, field_start: Int) -> (String, Int):
    var bytes = line.as_bytes()
    var start = field_start
    var end = field_start
    while end < len(bytes) and bytes[end] != Byte(ord("\t")):
        end += 1
    var next = end
    if next < len(bytes) and bytes[next] == Byte(ord("\t")):
        next += 1
    return (
        String(StringSlice(unsafe_from_utf8=bytes[start:end])),
        next,
    )


fn _case_source_from_path(path: String) raises -> CaseSourceInfo:
    var pathlib = Python.import_module("pathlib")
    var path_obj = pathlib.Path(path)
    var resolved = path_obj.resolve()
    return CaseSourceInfo(String(resolved), String(resolved.parent))


fn case_source_from_path(path: String) raises -> CaseSourceInfo:
    return _case_source_from_path(path)


fn write_json_cache(
    path: String, doc: JsonDocument, source_info: CaseSourceInfo
) raises:
    var serialized = String("STRUT_JSON_CACHE_V1\n")
    _append_cache_block(serialized, source_info.json_path)
    _append_cache_block(serialized, source_info.case_dir)
    serialized += String(doc.root_index) + "\n"
    serialized += String(len(doc.nodes)) + "\n"
    for i in range(len(doc.nodes)):
        var node = doc.nodes[i]
        serialized += (
            String(node.tag)
            + "\t"
            + String(node.start)
            + "\t"
            + String(node.end)
            + "\t"
            + String(Int(node.bool_value))
            + "\t"
            + String(Int(node.number_is_integer))
            + "\t"
            + String(node.child_offset)
            + "\t"
            + String(node.child_count)
            + "\t"
            + String(len(node.text_value.as_bytes()))
            + "\n"
        )
        serialized += node.text_value
        serialized += "\n"
    serialized += String(len(doc.object_entries)) + "\n"
    for i in range(len(doc.object_entries)):
        var entry = doc.object_entries[i]
        serialized += (
            String(entry.value_index)
            + "\t"
            + String(entry.key_offset)
            + "\t"
            + String(len(entry.key.as_bytes()))
            + "\n"
        )
        serialized += entry.key
        serialized += "\n"
    serialized += String(len(doc.array_items)) + "\n"
    for i in range(len(doc.array_items)):
        serialized += String(doc.array_items[i]) + "\n"
    Path(path).write_text(serialized)


fn load_json_cache(path: String) raises -> NativeCaseDocument:
    var serialized = Path(path).read_text()
    var position = 0
    var (magic, next_position) = _read_cache_line(serialized, position)
    position = next_position
    if magic != "STRUT_JSON_CACHE_V1":
        abort("invalid native input cache magic")

    var (json_path_len_line, json_path_len_position) = _read_cache_line(
        serialized, position
    )
    position = json_path_len_position
    var (json_path, json_path_position) = _read_cache_block(
        serialized, position, Int(json_path_len_line)
    )
    position = json_path_position
    var (case_dir_len_line, case_dir_len_position) = _read_cache_line(
        serialized, position
    )
    position = case_dir_len_position
    var (case_dir, case_dir_position) = _read_cache_block(
        serialized, position, Int(case_dir_len_line)
    )
    position = case_dir_position

    var source_info = CaseSourceInfo(json_path, case_dir)
    if source_info.json_path == "":
        source_info = _case_source_from_path(path)
    elif source_info.case_dir == "":
        source_info.case_dir = _case_source_from_path(source_info.json_path).case_dir

    var doc = JsonDocument()
    var (root_line, root_position) = _read_cache_line(serialized, position)
    position = root_position
    doc.root_index = Int(root_line)
    var (node_count_line, node_count_position) = _read_cache_line(
        serialized, position
    )
    position = node_count_position
    for _ in range(Int(node_count_line)):
        var (node_line, node_position) = _read_cache_line(serialized, position)
        position = node_position
        var field_position = 0
        var (tag_field, tag_position) = _next_tab_field(node_line, field_position)
        field_position = tag_position
        var (start_field, start_position) = _next_tab_field(
            node_line, field_position
        )
        field_position = start_position
        var (end_field, end_position) = _next_tab_field(node_line, field_position)
        field_position = end_position
        var (bool_field, bool_position) = _next_tab_field(node_line, field_position)
        field_position = bool_position
        var (number_integer_field, number_integer_position) = _next_tab_field(
            node_line, field_position
        )
        field_position = number_integer_position
        var (child_offset_field, child_offset_position) = _next_tab_field(
            node_line, field_position
        )
        field_position = child_offset_position
        var (child_count_field, child_count_position) = _next_tab_field(
            node_line, field_position
        )
        field_position = child_count_position
        var (text_len_field, _) = _next_tab_field(node_line, field_position)
        var (text_value, text_position) = _read_cache_block(
            serialized, position, Int(text_len_field)
        )
        position = text_position

        var node = JsonNode()
        node.tag = Int(tag_field)
        node.start = Int(start_field)
        node.end = Int(end_field)
        node.bool_value = Int(bool_field) != 0
        node.number_is_integer = Int(number_integer_field) != 0
        node.child_offset = Int(child_offset_field)
        node.child_count = Int(child_count_field)
        node.text_value = text_value
        if node.tag == JsonValueTag.Number and len(node.text_value) > 0:
            node.number_value = atof(StringSlice(node.text_value))
        doc.nodes.append(node)
    var (entry_count_line, entry_count_position) = _read_cache_line(
        serialized, position
    )
    position = entry_count_position
    for _ in range(Int(entry_count_line)):
        var (entry_line, entry_position) = _read_cache_line(serialized, position)
        position = entry_position
        var field_position = 0
        var (value_index_field, value_index_position) = _next_tab_field(
            entry_line, field_position
        )
        field_position = value_index_position
        var (key_offset_field, key_offset_position) = _next_tab_field(
            entry_line, field_position
        )
        field_position = key_offset_position
        var (key_len_field, _) = _next_tab_field(entry_line, field_position)
        var (key, key_position) = _read_cache_block(
            serialized, position, Int(key_len_field)
        )
        position = key_position
        doc.object_entries.append(
            JsonObjectEntry(key, Int(value_index_field), Int(key_offset_field))
        )
    var (array_count_line, array_count_position) = _read_cache_line(
        serialized, position
    )
    position = array_count_position
    for _ in range(Int(array_count_line)):
        var (item_line, item_position) = _read_cache_line(serialized, position)
        position = item_position
        doc.array_items.append(Int(item_line))
    var loaded = NativeCaseDocument()
    loaded.doc = doc^
    loaded.source_info = source_info
    return loaded^


fn load_case_document(
    input_path: String, input_bin_path: String, write_input_bin_path: String
) raises -> NativeCaseDocument:
    var sources = 0
    if input_path != "":
        sources += 1
    if input_bin_path != "":
        sources += 1
    if sources > 1:
        abort("use only one of --input or --input-bin")
    if input_path == "" and input_bin_path == "":
        abort("missing --input or --input-bin")
    if input_bin_path != "":
        var loaded = load_json_cache(input_bin_path)
        if write_input_bin_path != "":
            write_json_cache(write_input_bin_path, loaded.doc, loaded.source_info)
        return loaded^
    var source_info = _case_source_from_path(input_path)
    var doc = load_json_document_native(input_path)
    if write_input_bin_path != "":
        write_json_cache(write_input_bin_path, doc, source_info)
    var loaded = NativeCaseDocument()
    loaded.doc = doc^
    loaded.source_info = source_info
    return loaded^


fn parse_args() -> (String, String, String, String, String, String, Bool):
    var args = argv()
    var input_path = arg_value(args, "--input")
    var input_bin_path = arg_value(args, "--input-bin")
    var write_input_bin_path = arg_value(args, "--write-input-bin")
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
        input_bin_path,
        write_input_bin_path,
        output_path,
        batch_path,
        profile_path,
        compute_only,
    )
