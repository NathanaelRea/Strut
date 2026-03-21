from testing import assert_equal
from testing import TestSuite

from json_native import parse_json_native
from solver.run_case.input_types import parse_case_input_native
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


def test_native_decode_preserves_equal_dof_and_rigid_diaphragm_payloads():
    var doc = parse_json_native(
        "{\"model\":{\"ndm\":3,\"ndf\":6},\"nodes\":[{\"id\":10,\"x\":0.0,\"y\":0.0,\"z\":0.0},{\"id\":11,\"x\":4.0,\"y\":2.0,\"z\":0.5}],\"elements\":[],\"analysis\":{\"type\":\"static_linear\",\"constraints\":\"Transformation\"},\"mp_constraints\":[{\"type\":\"equalDOF\",\"retained_node\":10,\"constrained_node\":11,\"dofs\":[1,3]},{\"type\":\"rigidDiaphragm\",\"retained_node\":10,\"constrained_node\":11,\"perp_dirn\":3,\"constrained_dofs\":[1,2,6],\"retained_dofs\":[1,2,6],\"matrix\":[[1.0,0.0,-2.0],[0.0,1.0,4.0],[0.0,0.0,1.0]],\"dx\":4.0,\"dy\":2.0,\"dz\":0.5}]}"
    )
    var native_case = parse_case_input_native(doc)

    assert_equal(len(native_case.mp_constraints), 2)

    var equal_dof = native_case.mp_constraints[0]
    assert_equal(equal_dof.type, "equalDOF")
    assert_equal(equal_dof.dof_count, 2)
    assert_equal(equal_dof.dof_1, 1)
    assert_equal(equal_dof.dof_2, 3)
    assert_equal(equal_dof.rigid_matrix_row_count, 0)

    var rigid = native_case.mp_constraints[1]
    assert_equal(rigid.type, "rigidDiaphragm")
    assert_equal(rigid.rigid_perp_dirn, 3)
    assert_equal(rigid.rigid_constrained_dof_count, 3)
    assert_equal(rigid.rigid_constrained_dof_1, 1)
    assert_equal(rigid.rigid_constrained_dof_2, 2)
    assert_equal(rigid.rigid_constrained_dof_3, 6)
    assert_equal(rigid.rigid_retained_dof_count, 3)
    assert_equal(rigid.rigid_retained_dof_1, 1)
    assert_equal(rigid.rigid_retained_dof_2, 2)
    assert_equal(rigid.rigid_retained_dof_3, 6)
    assert_equal(rigid.rigid_matrix_row_count, 3)
    assert_equal(rigid.rigid_matrix_col_count, 3)
    assert_equal(rigid.rigid_matrix_11, 1.0)
    assert_equal(rigid.rigid_matrix_12, 0.0)
    assert_equal(rigid.rigid_matrix_13, -2.0)
    assert_equal(rigid.rigid_matrix_21, 0.0)
    assert_equal(rigid.rigid_matrix_22, 1.0)
    assert_equal(rigid.rigid_matrix_23, 4.0)
    assert_equal(rigid.rigid_matrix_31, 0.0)
    assert_equal(rigid.rigid_matrix_32, 0.0)
    assert_equal(rigid.rigid_matrix_33, 1.0)
    assert_equal(rigid.rigid_dx, 4.0)
    assert_equal(rigid.rigid_dy, 2.0)
    assert_equal(rigid.rigid_dz, 0.5)


def test_native_decode_preserves_2d_rigid_diaphragm_single_dof_payload():
    var doc = parse_json_native(
        "{\"model\":{\"ndm\":2,\"ndf\":3},\"nodes\":[{\"id\":1,\"x\":0.0,\"y\":0.0},{\"id\":2,\"x\":3.0,\"y\":0.0}],\"elements\":[],\"analysis\":{\"type\":\"static_linear\",\"constraints\":\"Transformation\"},\"mp_constraints\":[{\"type\":\"rigidDiaphragm\",\"retained_node\":1,\"constrained_node\":2,\"perp_dirn\":1,\"constrained_dofs\":[1],\"retained_dofs\":[1],\"matrix\":[[1.0]],\"dx\":3.0,\"dy\":0.0,\"dz\":0.0}]}"
    )
    var native_case = parse_case_input_native(doc)
    var rigid = native_case.mp_constraints[0]

    assert_equal(rigid.rigid_perp_dirn, 1)
    assert_equal(rigid.rigid_constrained_dof_count, 1)
    assert_equal(rigid.rigid_retained_dof_count, 1)
    assert_equal(rigid.rigid_matrix_row_count, 1)
    assert_equal(rigid.rigid_matrix_col_count, 1)
    assert_equal(rigid.rigid_constrained_dof_1, 1)
    assert_equal(rigid.rigid_retained_dof_1, 1)
    assert_equal(rigid.rigid_matrix_11, 1.0)
    assert_equal(rigid.rigid_matrix_12, 0.0)
    assert_equal(rigid.rigid_matrix_21, 0.0)
    assert_equal(rigid.rigid_dx, 3.0)
    assert_equal(rigid.rigid_dy, 0.0)
    assert_equal(rigid.rigid_dz, 0.0)


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
