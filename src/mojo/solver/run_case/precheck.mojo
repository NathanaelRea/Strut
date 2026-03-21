from collections import List
from os import abort

from json_native import JsonDocument, JsonValueTag
from solver.run_case.input_types import (
    analysis_type_tag,
    beam_integration_tag,
    canonical_analysis_system_name,
    constraint_handler_tag,
    element_load_type_tag,
    element_type_tag,
    force_beam_mode_tag,
    geom_transf_tag,
    integrator_type_tag,
    numberer_tag,
    pattern_type_tag,
    recorder_type_tag,
)
from tag_types import (
    AnalysisTypeTag,
    ConstraintHandlerTag,
    ElementTypeTag,
    IntegratorTypeTag,
    NumbererTag,
    PatternTypeTag,
    RecorderTypeTag,
)


fn _json_has_value(doc: JsonDocument, node_index: Int) -> Bool:
    return node_index >= 0 and doc.node_tag(node_index) != JsonValueTag.Null


fn _json_key(doc: JsonDocument, object_index: Int, key: StringSlice) raises -> Int:
    if object_index < 0:
        return -1
    if doc.node_tag(object_index) != JsonValueTag.Object:
        return -1
    return doc.object_find(object_index, key)


fn _json_string_key(
    doc: JsonDocument, object_index: Int, key: StringSlice, default: String
) raises -> String:
    var node_index = _json_key(doc, object_index, key)
    if node_index < 0 or doc.node_tag(node_index) != JsonValueTag.String:
        return default
    return doc.node_text(node_index)


fn _json_int_key(
    doc: JsonDocument, object_index: Int, key: StringSlice, default: Int
) raises -> Int:
    var node_index = _json_key(doc, object_index, key)
    if node_index < 0 or doc.node_tag(node_index) != JsonValueTag.Number:
        return default
    return Int(doc.node_number(node_index))


fn _is_supported_material_type(type_name: String) -> Bool:
    return (
        type_name == "Elastic"
        or type_name == "Steel01"
        or type_name == "Concrete01"
        or type_name == "Steel02"
        or type_name == "Concrete02"
        or type_name == "ElasticIsotropic"
        or type_name == "PlateFromPlaneStress"
        or type_name == "PlateRebar"
        or type_name == "PlaneStressUserMaterial"
    )


fn _is_supported_section_type(type_name: String) -> Bool:
    return (
        type_name == "ElasticSection2d"
        or type_name == "ElasticSection3d"
        or type_name == "FiberSection2d"
        or type_name == "FiberSection3d"
        or type_name == "AggregatorSection2d"
        or type_name == "ElasticMembranePlateSection"
        or type_name == "LayeredShellSection"
    )


fn _is_supported_uniaxial_material(type_name: String) -> Bool:
    return (
        type_name == "Elastic"
        or type_name == "Steel01"
        or type_name == "Concrete01"
        or type_name == "Steel02"
        or type_name == "Concrete02"
    )


fn _precheck_abort(message: String) -> None:
    abort("[precheck-fail] " + message)


fn _precheck_analysis(doc: JsonDocument, analysis_index: Int) raises:
    if not _json_has_value(doc, analysis_index):
        return

    var analysis_type = _json_string_key(doc, analysis_index, "type", "static_linear")
    if analysis_type_tag(analysis_type) == AnalysisTypeTag.Unknown:
        _precheck_abort("unsupported analysis type: " + analysis_type)

    var constraints = _json_string_key(doc, analysis_index, "constraints", "Plain")
    if constraint_handler_tag(constraints) == ConstraintHandlerTag.Unknown:
        _precheck_abort("unsupported constraints handler: " + constraints)

    var numberer = _json_string_key(doc, analysis_index, "numberer", "")
    if len(numberer) > 0 and numberer_tag(numberer) == NumbererTag.Unknown:
        _precheck_abort("unsupported analysis numberer: " + numberer)

    var force_beam_mode = _json_string_key(doc, analysis_index, "force_beam_mode", "auto")
    if len(force_beam_mode) > 0 and force_beam_mode_tag(force_beam_mode) == 0:
        _precheck_abort("unsupported force_beam_mode: " + force_beam_mode)

    var system_name = ""
    var system_index = _json_key(doc, analysis_index, "system")
    if system_index >= 0:
        system_name = _json_string_key(doc, analysis_index, "system", "")
    else:
        var solver_index = _json_key(doc, analysis_index, "solver")
        if solver_index >= 0:
            system_name = _json_string_key(doc, analysis_index, "solver", "")
    _ = canonical_analysis_system_name(system_name)

    var integrator_index = _json_key(doc, analysis_index, "integrator")
    var integrator_type = ""
    if analysis_type == "static_nonlinear":
        integrator_type = "LoadControl"
    elif (
        analysis_type == "transient_linear"
        or analysis_type == "transient_nonlinear"
    ):
        integrator_type = "Newmark"
    integrator_type = _json_string_key(doc, integrator_index, "type", integrator_type)
    if (
        len(integrator_type) > 0
        and integrator_type_tag(integrator_type) == IntegratorTypeTag.Unknown
    ):
        _precheck_abort("unsupported analysis integrator: " + integrator_type)

    if analysis_type == "staged":
        var stages_index = _json_key(doc, analysis_index, "stages")
        if stages_index < 0:
            return
        if doc.node_tag(stages_index) != JsonValueTag.Array:
            return
        for i in range(doc.node_len(stages_index)):
            var stage_index = doc.array_item(stages_index, i)
            var stage_analysis_index = _json_key(doc, stage_index, "analysis")
            if stage_analysis_index < 0:
                stage_analysis_index = stage_index
            _precheck_analysis(doc, stage_analysis_index)
            _precheck_pattern(doc, stage_index)
            _precheck_time_series(doc, stage_index)
            _precheck_element_loads(doc, stage_index)


fn _precheck_pattern(doc: JsonDocument, owner_index: Int) raises:
    var pattern_index = _json_key(doc, owner_index, "pattern")
    if not _json_has_value(doc, pattern_index):
        return
    var pattern_type = _json_string_key(doc, pattern_index, "type", "Plain")
    if pattern_type_tag(pattern_type) == PatternTypeTag.Unknown:
        _precheck_abort("unsupported pattern type: " + pattern_type)


fn _precheck_time_series_entry(doc: JsonDocument, ts_index: Int) raises:
    var typ = _json_string_key(doc, ts_index, "type", "")
    if typ == "PathFile":
        typ = "Path"
    if (
        typ != "Constant"
        and typ != "Linear"
        and typ != "Path"
        and typ != "Trig"
    ):
        _precheck_abort("unsupported time_series type: " + typ)


fn _precheck_time_series(doc: JsonDocument, owner_index: Int) raises:
    var ts_index = _json_key(doc, owner_index, "time_series")
    if not _json_has_value(doc, ts_index):
        return
    if doc.node_tag(ts_index) == JsonValueTag.Array:
        for i in range(doc.node_len(ts_index)):
            _precheck_time_series_entry(doc, doc.array_item(ts_index, i))
        return
    if doc.node_tag(ts_index) == JsonValueTag.Object:
        _precheck_time_series_entry(doc, ts_index)


fn _precheck_element_loads(doc: JsonDocument, owner_index: Int) raises:
    var loads_index = _json_key(doc, owner_index, "element_loads")
    if loads_index < 0:
        return
    if doc.node_tag(loads_index) != JsonValueTag.Array:
        return
    for i in range(doc.node_len(loads_index)):
        var load_index = doc.array_item(loads_index, i)
        var load_type = _json_string_key(doc, load_index, "type", "")
        if element_load_type_tag(load_type) == 0:
            _precheck_abort("unsupported element load type")


fn _precheck_dampings(doc: JsonDocument, root_index: Int) raises:
    var dampings_index = _json_key(doc, root_index, "dampings")
    if not _json_has_value(doc, dampings_index):
        return

    if doc.node_tag(dampings_index) == JsonValueTag.Object:
        var damping_type = _json_string_key(doc, dampings_index, "type", "")
        if damping_type == "SecStiff":
            damping_type = "SecStif"
        if damping_type != "SecStif":
            _precheck_abort("unsupported damping type: " + damping_type)
        return

    if doc.node_tag(dampings_index) != JsonValueTag.Array:
        return

    for i in range(doc.node_len(dampings_index)):
        var damping_index = doc.array_item(dampings_index, i)
        var damping_type = _json_string_key(doc, damping_index, "type", "")
        if damping_type == "SecStiff":
            damping_type = "SecStif"
        if damping_type != "SecStif":
            _precheck_abort("unsupported damping type: " + damping_type)


fn _build_material_maps(
    doc: JsonDocument,
    root_index: Int,
    mut material_types_by_id: List[String],
    mut material_base_by_id: List[Int],
) raises:
    var materials_index = _json_key(doc, root_index, "materials")
    if materials_index < 0:
        return
    if doc.node_tag(materials_index) != JsonValueTag.Array:
        return

    for i in range(doc.node_len(materials_index)):
        var material_index = doc.array_item(materials_index, i)
        var material_id = _json_int_key(doc, material_index, "id", -1)
        if material_id < 0:
            continue
        if material_id >= len(material_types_by_id):
            material_types_by_id.resize(material_id + 1, "")
            material_base_by_id.resize(material_id + 1, -1)
        var material_type = _json_string_key(doc, material_index, "type", "")
        material_types_by_id[material_id] = material_type
        var params_index = _json_key(doc, material_index, "params")
        material_base_by_id[material_id] = _json_int_key(
            doc, params_index, "material", -1
        )
    return


fn _precheck_layered_shell_sections(
    doc: JsonDocument,
    root_index: Int,
    material_types_by_id: List[String],
    material_base_by_id: List[Int],
) raises:
    var sections_index = _json_key(doc, root_index, "sections")
    if sections_index < 0:
        return
    if doc.node_tag(sections_index) != JsonValueTag.Array:
        return

    for i in range(doc.node_len(sections_index)):
        var section_index = doc.array_item(sections_index, i)
        if _json_string_key(doc, section_index, "type", "") != "LayeredShellSection":
            continue
        var params_index = _json_key(doc, section_index, "params")
        var layers_index = _json_key(doc, params_index, "layers")
        if layers_index < 0:
            continue
        if doc.node_tag(layers_index) != JsonValueTag.Array:
            continue
        for j in range(doc.node_len(layers_index)):
            var layer_index = doc.array_item(layers_index, j)
            var material_id = _json_int_key(doc, layer_index, "material", -1)
            if material_id < 0 or material_id >= len(material_types_by_id):
                continue
            var material_type = material_types_by_id[material_id]
            if material_type == "PlaneStressUserMaterial":
                _precheck_abort(
                    "LayeredShellSection runtime does not support PlaneStressUserMaterial because the OpenSees PSUMAT implementation is not present in the reference repo"
                )
            if material_type == "PlateFromPlaneStress":
                var base_id = material_base_by_id[material_id]
                if base_id < 0 or base_id >= len(material_types_by_id):
                    continue
                var base_type = material_types_by_id[base_id]
                if (
                    base_type != "ElasticIsotropic"
                    and base_type != "PlaneStressUserMaterial"
                ):
                    _precheck_abort(
                        "LayeredShellSection currently supports PlateFromPlaneStress only over ElasticIsotropic"
                    )
                continue
            if material_type == "PlateRebar":
                var base_id = material_base_by_id[material_id]
                if base_id < 0 or base_id >= len(material_types_by_id):
                    continue
                if not _is_supported_uniaxial_material(material_types_by_id[base_id]):
                    _precheck_abort(
                        "PlateRebar base material must be a supported uniaxial material"
                    )
                continue
            if material_type == "ElasticIsotropic":
                continue
            if len(material_type) > 0:
                _precheck_abort(
                    "unsupported LayeredShellSection material: " + material_type
                )


fn precheck_case_input_native(doc: JsonDocument, include_recorders: Bool) raises:
    var root_index = doc.root_index
    if root_index < 0 or doc.node_tag(root_index) != JsonValueTag.Object:
        return

    var material_types_by_id: List[String] = []
    var material_base_by_id: List[Int] = []

    var materials_index = _json_key(doc, root_index, "materials")
    if materials_index >= 0 and doc.node_tag(materials_index) == JsonValueTag.Array:
        for i in range(doc.node_len(materials_index)):
            var material_index = doc.array_item(materials_index, i)
            var material_type = _json_string_key(doc, material_index, "type", "")
            if len(material_type) > 0 and not _is_supported_material_type(material_type):
                _precheck_abort("unsupported material type: " + material_type)
        _build_material_maps(
            doc,
            root_index,
            material_types_by_id,
            material_base_by_id,
        )

    var sections_index = _json_key(doc, root_index, "sections")
    if sections_index >= 0 and doc.node_tag(sections_index) == JsonValueTag.Array:
        for i in range(doc.node_len(sections_index)):
            var section_index = doc.array_item(sections_index, i)
            var section_type = _json_string_key(doc, section_index, "type", "")
            if len(section_type) > 0 and not _is_supported_section_type(section_type):
                _precheck_abort("unsupported section type: " + section_type)
        _precheck_layered_shell_sections(
            doc,
            root_index,
            material_types_by_id,
            material_base_by_id,
        )

    var elements_index = _json_key(doc, root_index, "elements")
    if elements_index >= 0 and doc.node_tag(elements_index) == JsonValueTag.Array:
        for i in range(doc.node_len(elements_index)):
            var element_index = doc.array_item(elements_index, i)
            var element_type = _json_string_key(doc, element_index, "type", "")
            if element_type_tag(element_type) == ElementTypeTag.Unknown:
                _precheck_abort("unsupported element type: " + element_type)
            var geom_name = _json_string_key(doc, element_index, "geomTransf", "")
            if len(geom_name) > 0 and geom_transf_tag(geom_name) == 0:
                _precheck_abort("unsupported geomTransf: " + geom_name)
            var integration_name = _json_string_key(doc, element_index, "integration", "")
            if len(integration_name) > 0 and beam_integration_tag(integration_name) == 0:
                _precheck_abort("unsupported beam integration: " + integration_name)

    _precheck_analysis(doc, _json_key(doc, root_index, "analysis"))
    _precheck_pattern(doc, root_index)
    _precheck_time_series(doc, root_index)
    _precheck_element_loads(doc, root_index)
    _precheck_dampings(doc, root_index)

    var analysis_index = _json_key(doc, root_index, "analysis")
    var constraints = _json_string_key(doc, analysis_index, "constraints", "Plain")
    var constraints_tag = constraint_handler_tag(constraints)
    var mpc_index = _json_key(doc, root_index, "mp_constraints")
    if mpc_index >= 0 and doc.node_tag(mpc_index) == JsonValueTag.Array:
        if (
            constraints_tag != ConstraintHandlerTag.Transformation
            and constraints_tag != ConstraintHandlerTag.Lagrange
        ):
            _precheck_abort(
                "mp_constraints require analysis.constraints=Transformation or Lagrange"
            )
        for i in range(doc.node_len(mpc_index)):
            var constraint_index = doc.array_item(mpc_index, i)
            var constraint_type = _json_string_key(doc, constraint_index, "type", "")
            if constraint_type != "equalDOF":
                _precheck_abort("unsupported mp constraint type: " + constraint_type)

    if not include_recorders:
        return

    var recorders_index = _json_key(doc, root_index, "recorders")
    if recorders_index < 0:
        return
    if doc.node_tag(recorders_index) != JsonValueTag.Array:
        return
    for i in range(doc.node_len(recorders_index)):
        var recorder_index = doc.array_item(recorders_index, i)
        var recorder_type = _json_string_key(doc, recorder_index, "type", "")
        if recorder_type_tag(recorder_type) == RecorderTypeTag.Unknown:
            _precheck_abort("unsupported recorder type")
