from testing import assert_equal, assert_raises, assert_true
from testing import TestSuite

from json_native import JsonValueTag, load_json_native, parse_json_native


def test_parse_basic_object_and_array():
    var doc = parse_json_native(
        "{\"flag\":true,\"count\":12,\"name\":\"beam\",\"items\":[null,false,1.25e3]}"
    )

    var root = doc.root_index
    assert_equal(doc.node_tag(root), JsonValueTag.Object)

    var flag_index = doc.object_find(root, "flag")
    var count_index = doc.object_find(root, "count")
    var name_index = doc.object_find(root, "name")
    var items_index = doc.object_find(root, "items")

    assert_true(flag_index >= 0)
    assert_true(count_index >= 0)
    assert_true(name_index >= 0)
    assert_true(items_index >= 0)

    assert_equal(doc.node_bool(flag_index), True)
    assert_equal(doc.node_number(count_index), 12.0)
    assert_equal(doc.node_text(name_index), "beam")
    assert_equal(doc.node_tag(items_index), JsonValueTag.Array)
    assert_equal(doc.node_len(items_index), 3)
    assert_equal(
        doc.node_tag(doc.array_item(items_index, 0)),
        JsonValueTag.Null,
    )
    assert_equal(doc.node_bool(doc.array_item(items_index, 1)), False)
    assert_equal(doc.node_number(doc.array_item(items_index, 2)), 1250.0)


def test_parse_string_escapes_and_unicode():
    var doc = parse_json_native(
        "{\"text\":\"line\\nquote: \\\"x\\\" slash: \\\\ greek: \\u03B1 fire: \\uD83D\\uDD25\"}"
    )

    var root = doc.root_index
    var text_index = doc.object_find(root, "text")
    assert_equal(
        doc.node_text(text_index),
        "line\nquote: \"x\" slash: \\ greek: α fire: 🔥",
    )


def test_parse_existing_case_file():
    var doc = load_json_native(
        "tests/validation/elastic_beam_cantilever/elastic_beam_cantilever.json"
    )
    var root = doc.root_index
    var model_index = doc.object_find(root, "model")
    var nodes_index = doc.object_find(root, "nodes")
    var elements_index = doc.object_find(root, "elements")
    var analysis_index = doc.object_find(root, "analysis")

    assert_equal(doc.node_tag(root), JsonValueTag.Object)
    assert_equal(doc.node_number(doc.object_find(model_index, "ndm")), 2.0)
    assert_equal(doc.node_len(nodes_index), 2)
    assert_equal(doc.node_len(elements_index), 1)
    assert_equal(
        doc.node_text(doc.object_find(analysis_index, "type")),
        "static_linear",
    )


def test_parse_error_reports_line_column_and_context():
    with assert_raises(contains="JSON parse error at line 2, column 18"):
        _ = parse_json_native("{\n  \"items\": [1, 2,],\n  \"ok\": true\n}")

    with assert_raises(contains="trailing comma in array"):
        _ = parse_json_native("{\"items\":[1,2,]}")

    with assert_raises(contains="numbers cannot have leading zeroes"):
        _ = parse_json_native("{\"id\":01}")

    with assert_raises(contains="invalid escape sequence"):
        _ = parse_json_native("{\"bad\":\"\\q\"}")

    with assert_raises(contains="unexpected trailing content"):
        _ = parse_json_native("{\"ok\":true} []")


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
