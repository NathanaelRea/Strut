from testing import assert_equal
from testing import TestSuite

from json_native import JsonValueTag, parse_json_native
from strut_io import CaseSourceInfo, load_json_cache, write_json_cache


def test_json_cache_round_trips_document_and_source_info():
    var doc = parse_json_native(
        "{\"model\":{\"ndm\":2,\"ndf\":3},\"items\":[true,1.25,\"beam\"]}"
    )
    var cache_path = "/tmp/strut_test_case_cache.bin"
    var source_info = CaseSourceInfo("/tmp/example_case.json", "/tmp")

    write_json_cache(cache_path, doc, source_info)
    var loaded = load_json_cache(cache_path)

    assert_equal(loaded.source_info.json_path, source_info.json_path)
    assert_equal(loaded.source_info.case_dir, source_info.case_dir)
    assert_equal(loaded.doc.root_index, doc.root_index)
    assert_equal(len(loaded.doc.nodes), len(doc.nodes))
    assert_equal(len(loaded.doc.object_entries), len(doc.object_entries))
    assert_equal(len(loaded.doc.array_items), len(doc.array_items))

    var root = loaded.doc.root_index
    var items = loaded.doc.object_find(root, "items")
    assert_equal(loaded.doc.node_tag(items), JsonValueTag.Array)
    assert_equal(loaded.doc.node_bool(loaded.doc.array_item(items, 0)), True)
    assert_equal(loaded.doc.node_number(loaded.doc.array_item(items, 1)), 1.25)
    assert_equal(loaded.doc.node_text(loaded.doc.array_item(items, 2)), "beam")


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
