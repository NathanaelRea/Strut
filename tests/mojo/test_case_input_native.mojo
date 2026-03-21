from testing import assert_equal
from testing import TestSuite

from json_native import parse_json_native
from solver.run_case.input_types import parse_case_input_native_from_source
from solver.run_case.precheck import precheck_case_input_native
from strut_io import case_source_from_path, load_json_native


def test_native_decode_loads_basic_case():
    var path = "tests/validation/elastic_beam_cantilever/elastic_beam_cantilever.json"
    var source_info = case_source_from_path(path)
    var native_doc = load_json_native(path)
    var native_case = parse_case_input_native_from_source(native_doc, source_info, True)

    assert_equal(native_case.model.ndm, 2)
    assert_equal(native_case.model.ndf, 3)
    assert_equal(len(native_case.nodes), 2)
    assert_equal(len(native_case.elements), 1)
    assert_equal(len(native_case.sections), 1)
    assert_equal(len(native_case.materials), 1)
    assert_equal(native_case.analysis.type, "static_linear")
    assert_equal(native_case.analysis.steps, 1)
    assert_equal(len(native_case.recorders), 1)
    assert_equal(native_case.recorders[0].node_count, 2)
    assert_equal(native_case.recorders[0].dof_count, 3)


def test_native_decode_loads_inline_path_time_series_case():
    var path = "tests/validation/elastic_time_series_path/elastic_time_series_path.json"
    var source_info = case_source_from_path(path)
    var native_doc = load_json_native(path)
    var native_case = parse_case_input_native_from_source(native_doc, source_info, True)

    assert_equal(len(native_case.time_series), 1)
    assert_equal(len(native_case.time_series_values), 3)
    assert_equal(len(native_case.time_series_times), 3)
    assert_equal(native_case.time_series[0].values_count, 3)
    assert_equal(native_case.time_series[0].time_count, 3)
    assert_equal(native_case.time_series_values[1], 1.0)
    assert_equal(native_case.time_series_times[2], 1.0)
    assert_equal(native_case.analysis.type, "static_linear")


def test_native_precheck_allows_layered_shell_surrogate_material_path():
    var doc = parse_json_native(
        "{\"model\":{\"ndm\":3,\"ndf\":6},\"nodes\":[],\"materials\":[{\"id\":1,\"type\":\"PlaneStressUserMaterial\",\"params\":{\"rho\":1.0,\"props\":[1.0,0.3]}},{\"id\":2,\"type\":\"PlateFromPlaneStress\",\"params\":{\"material\":1,\"gmod\":1.0}}],\"sections\":[{\"id\":10,\"type\":\"LayeredShellSection\",\"params\":{\"layers\":[{\"material\":2,\"thickness\":0.5}]}}],\"elements\":[],\"analysis\":{\"type\":\"static_linear\"}}"
    )

    precheck_case_input_native(doc, True)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
