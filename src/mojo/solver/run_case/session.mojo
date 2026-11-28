from json_native import JsonDocument, parse_json_native
from sections import FiberSection2dDef, FiberSection3dDef
from materials import UniMaterialState
from solver.assembly import assemble_internal_forces_typed_soa
from solver.assembly.surface_cache import SurfaceAssemblyCache
from solver.run_case.helpers import (
    _element_basic_force_for_recorder,
    _element_deformation_for_recorder,
    _element_force_global_for_recorder,
    _element_local_force_for_recorder,
    _section_response_for_recorder,
)
from solver.run_case.input_types import (
    ElementLoadInput,
    UniformExcitationInput,
    _json_get_bool,
    _json_get_float,
    _json_get_int,
    _json_get_string,
)
from solver.run_case.loader import RunCaseState
from solver.run_case.load_state import (
    add_uniform_excitation_load,
    build_active_element_load_state,
    build_active_nodal_load,
)
from solver.time_series import TimeSeriesInput, eval_time_series_input
from tag_types import PatternTypeTag
from sys.ffi import external_call


alias SessionBorrow[T: AnyType] = UnsafePointer[T, mut=False, origin=ImmutableAnyOrigin]


struct SessionProtocol(Movable):
    var input: UnsafePointer[NoneType]
    var line: UnsafePointer[UInt8]
    var line_capacity: Int

    fn __init__(out self, enabled: Bool) raises:
        self.input = UnsafePointer[NoneType]()
        self.line = UnsafePointer[UInt8]()
        self.line_capacity = 0
        if enabled:
            var mode = String("r")
            var fd = external_call["dup", Int32](Int32(0))
            if fd < 0:
                raise Error("cannot duplicate solver session stdin")
            self.input = external_call["fdopen", UnsafePointer[NoneType]](
                fd, mode.unsafe_cstr_ptr()
            )
            if not self.input:
                _ = external_call["close", Int32](Int(fd))
                raise Error("cannot open solver session stdin")

    fn __del__(deinit self):
        if self.input:
            _ = external_call["fclose", Int32](self.input)
        if self.line:
            _ = external_call["free", NoneType](self.line)

    fn _receive(mut self) -> Int:
        return external_call["getline", Int](
            UnsafePointer(to=self.line), UnsafePointer(to=self.line_capacity), self.input
        )

    fn _line_text(self, size: Int) -> String:
        if size < 0:
            return ""
        return String(StringSlice(unsafe_from_utf8=Span(ptr=self.line, length=size)))

    fn read_line(mut self) -> String:
        var size = self._receive()
        return self._line_text(size)

    fn read_driver_command(mut self) -> (Bool, String):
        var size = self._receive()
        if size == 2 and self.line[0] == UInt8(ord("r")) and self.line[1] == UInt8(ord("\n")):
            return (True, "")
        return (False, self._line_text(size))

    fn decode(self, line: String) raises -> JsonDocument:
        if line == "":
            return parse_json_native('{"op":"close"}')
        return parse_json_native(line)

    fn reply(self, json: String) raises:
        var line = json + "\n"
        var bytes = line.as_bytes()
        self.write_reply(bytes.unsafe_ptr().bitcast[UInt8](), len(bytes))

    fn reply_binary(self, words: List[Float64]) raises:
        self.write_reply(words.unsafe_ptr().bitcast[UInt8](), len(words) * 8)

    fn write_reply(self, data: UnsafePointer[UInt8], size: Int) raises:
        var offset = 0
        while offset < size:
            var written = Int(external_call["write", Int32](1, data + offset, size - offset))
            if written <= 0:
                raise Error("cannot write solver session reply")
            offset += written


struct SessionProgress(Movable):
    var enabled: Bool
    var protocol: SessionProtocol
    var pending: JsonDocument
    var observed_nodes: List[Int]
    var binary_reply: Bool
    var reply_words: List[Float64]
    var requested_steps: Int
    var command_start_step: Int
    var replied: Bool
    var has_driver_time: Bool
    var failed: Bool
    var steps: Int
    var time: Float64
    var increment: Float64
    var iterations: Int
    var displacement_increment: Float64
    var displacement_iterations: Int
    var hs_increment: Float64
    var hs_reference: List[Float64]

    # The staged dispatcher binds these borrows only for the duration of a
    # command. Model fields the drivers mutate are supplied by the caller.
    var model: SessionBorrow[RunCaseState]
    var const_F: SessionBorrow[List[Float64]]
    var pattern_F: SessionBorrow[List[Float64]]
    var const_element_loads: SessionBorrow[List[ElementLoadInput]]
    var pattern_element_loads: SessionBorrow[List[ElementLoadInput]]
    var uniform_excitations: SessionBorrow[List[UniformExcitationInput]]
    var pattern_type_tag: Int

    fn __init__(out self, enabled: Bool) raises:
        self.enabled = enabled
        self.protocol = SessionProtocol(enabled)
        self.pending = JsonDocument()
        self.observed_nodes = []
        self.binary_reply = False
        self.reply_words = []
        self.requested_steps = 0
        self.command_start_step = 0
        self.replied = False
        self.has_driver_time = False
        self.failed = False
        self.steps = 0
        self.time = 0.0
        self.increment = 0.0
        self.iterations = 0
        self.displacement_increment = 0.0
        self.displacement_iterations = 0
        self.hs_increment = 0.0
        self.hs_reference = []
        self.model = SessionBorrow[RunCaseState]()
        self.const_F = SessionBorrow[List[Float64]]()
        self.pattern_F = SessionBorrow[List[Float64]]()
        self.const_element_loads = SessionBorrow[List[ElementLoadInput]]()
        self.pattern_element_loads = SessionBorrow[List[ElementLoadInput]]()
        self.uniform_excitations = SessionBorrow[List[UniformExcitationInput]]()
        self.pattern_type_tag = PatternTypeTag.Plain

    fn reset_static_integrator(mut self):
        self.displacement_increment = 0.0
        self.displacement_iterations = 0
        self.hs_increment = 0.0
        self.hs_reference.clear()

    fn bind_loads(
        mut self,
        const_F: List[Float64],
        pattern_F: List[Float64],
        const_element_loads: List[ElementLoadInput],
        pattern_element_loads: List[ElementLoadInput],
        uniform_excitations: List[UniformExcitationInput],
        pattern_type_tag: Int,
    ):
        self.const_F = UnsafePointer(to=const_F)
        self.pattern_F = UnsafePointer(to=pattern_F)
        self.const_element_loads = UnsafePointer(to=const_element_loads)
        self.pattern_element_loads = UnsafePointer(to=pattern_element_loads)
        self.uniform_excitations = UnsafePointer(to=uniform_excitations)
        self.pattern_type_tag = pattern_type_tag

    fn read(mut self) raises -> JsonDocument:
        if self.pending.root_index >= 0:
            var command = JsonDocument()
            swap(command, self.pending)
            return command^
        return self.protocol.decode(self.protocol.read_line())

    fn begin_command(mut self, steps: Int, start_step: Int, binary_reply: Bool):
        self.requested_steps = steps
        self.command_start_step = start_step
        self.replied = False
        self.has_driver_time = False
        self.failed = False
        self.binary_reply = binary_reply

    fn begin(mut self, command: JsonDocument, steps: Int, start_step: Int = 0) raises:
        self.begin_command(steps, start_step, False)
        self.observed_nodes.clear()
        var nodes = command.object_find(command.root_index, "nodes_to_observe")
        if nodes >= 0:
            ref ids = self.model[].id_to_index
            for i in range(command.node_len(nodes)):
                var tag = Int(command.node_number(command.array_item(nodes, i)))
                var node = -1
                if tag >= 0 and tag < len(ids):
                    node = ids[tag]
                self.observed_nodes.append(node)

    fn fail(mut self, steps: Int, time: Float64, increment: Float64 = 0.0, iterations: Int = 0):
        self.failed = True
        self.has_driver_time = True
        self.steps = steps - self.command_start_step
        self.time = time
        self.increment = increment
        self.iterations = iterations

    fn reply_result(
        mut self,
        time: Float64,
        u: List[Float64],
        v: List[Float64],
        a: List[Float64],
    ) raises:
        var ndf = self.model[].ndf
        if self.binary_reply:
            self.reply_words.resize(2 + len(self.observed_nodes) * 3 * ndf, 0.0)
            self.reply_words[0] = -3.0 if self.failed else 0.0
            self.reply_words[1] = time
            for i in range(len(self.observed_nodes)):
                var node = self.observed_nodes[i]
                if node < 0:
                    raise Error("compact repeat requires valid observed nodes")
                var source = node * ndf
                var target = 2 + i * 3 * ndf
                for j in range(ndf):
                    self.reply_words[target + j] = u[source + j]
                    self.reply_words[target + ndf + j] = v[source + j]
                    self.reply_words[target + 2 * ndf + j] = a[source + j]
            self.protocol.reply_binary(self.reply_words)
            return

        var observations = ""
        if len(self.observed_nodes) > 0:
            observations = ',"nodes":['
            for i in range(len(self.observed_nodes)):
                if i > 0:
                    observations += ","
                var node = self.observed_nodes[i]
                if node < 0:
                    observations += "null"
                else:
                    append_session_node(observations, node, ndf, u, v, a)
            observations += "]"
        self.protocol.reply(self.result(self.requested_steps, time, observations))

    fn continue_steps(
        mut self,
        completed_steps: Int,
        time: Float64,
        pattern_scale: Float64,
        u: List[Float64],
        v: List[Float64],
        a: List[Float64],
        mut uniaxial_states: List[UniMaterialState],
        mut force_basic_q: List[Float64],
        mut fiber_section_defs: List[FiberSection2dDef],
        mut fiber_section3d_defs: List[FiberSection3dDef],
        mut shell_cache: SurfaceAssemblyCache,
        can_continue: Bool = True,
    ) raises -> Int:
        self.reply_result(time, u, v, a)
        self.replied = True
        self.has_driver_time = True
        self.time = time
        while True:
            var compact, line = self.protocol.read_driver_command()
            if compact:
                if not can_continue:
                    raise Error("compact repeat requires an active solver driver")
                self.begin_command(1, completed_steps, True)
                return 1
            var command = self.protocol.decode(line)
            var root = command.root_index
            var op = _json_get_string(command, root, "op", "")
            if (
                can_continue
                and op == "repeat"
                and _json_get_float(command, root, "time", time) == time
                and _json_get_bool(command, root, "resume", True)
            ):
                var steps = _json_get_int(command, root, "steps", 1)
                if steps > 0:
                    self.begin(command, steps, completed_steps)
                    return steps
            if self.answer_query(
                command,
                op,
                time,
                pattern_scale,
                u,
                v,
                a,
                uniaxial_states,
                force_basic_q,
                fiber_section_defs,
                fiber_section3d_defs,
                shell_cache,
            ):
                continue
            self.pending = command^
            return 0

    fn answer_query(
        self,
        command: JsonDocument,
        op: String,
        replay_time: Float64,
        committed_pattern_scale: Float64,
        u: List[Float64],
        v: List[Float64],
        a: List[Float64],
        mut uniaxial_states: List[UniMaterialState],
        mut force_basic_q: List[Float64],
        mut fiber_section_defs: List[FiberSection2dDef],
        mut fiber_section3d_defs: List[FiberSection3dDef],
        mut shell_cache: SurfaceAssemblyCache,
    ) raises -> Bool:
        var root = command.root_index
        ref model = self.model[]
        if op == "node":
            var node = _json_get_int(command, root, "node", -1)
            if node < 0 or node >= len(model.id_to_index) or model.id_to_index[node] < 0:
                self.protocol.reply('{"error":"unknown node"}')
                return True
            var reply = '{"values":'
            append_session_node(
                reply,
                model.id_to_index[node],
                model.ndf,
                u,
                v,
                a,
            )
            self.protocol.reply(reply + "}")
            return True
        if op == "element":
            var tag = _json_get_int(command, root, "element", -1)
            if tag < 0 or tag >= len(model.elem_id_to_index) or model.elem_id_to_index[tag] < 0:
                self.protocol.reply('{"error":"unknown element"}')
                return True
            var elem_index = model.elem_id_to_index[tag]
            ref elem = model.typed_elements[elem_index]
            var kind = _json_get_string(command, root, "kind", "")
            var material_stress = _json_get_bool(command, root, "material_stress", False)
            var sec_no = _json_get_int(command, root, "section", 1)
            var want_defo = kind == "section_deformation"
            var active_loads = build_active_element_load_state(
                self.const_element_loads[],
                self.pattern_element_loads[],
                committed_pattern_scale,
                model.typed_elements,
                model.elem_id_to_index,
                model.ndm,
                model.ndf,
            )
            var values: List[Float64]
            if kind == "element_force":
                values = _element_force_global_for_recorder(
                    elem_index,
                    elem,
                    model.ndf,
                    u,
                    active_loads.element_loads,
                    active_loads.elem_load_offsets,
                    active_loads.elem_load_pool,
                    1.0,
                    model.typed_nodes,
                    model.typed_sections_by_id,
                    fiber_section_defs,
                    model.fiber_section_cells,
                    model.fiber_section_index_by_id,
                    fiber_section3d_defs,
                    model.fiber_section3d_cells,
                    model.fiber_section3d_index_by_id,
                    model.uniaxial_defs,
                    model.uniaxial_state_defs,
                    uniaxial_states,
                    model.elem_uniaxial_offsets,
                    model.elem_uniaxial_counts,
                    model.elem_uniaxial_state_ids,
                    model.force_basic_offsets,
                    model.force_basic_counts,
                    force_basic_q,
                    shell_cache.continuum,
                )
            elif kind == "element_local_force":
                values = _element_local_force_for_recorder(
                    elem_index,
                    elem,
                    model.ndf,
                    u,
                    active_loads.element_loads,
                    active_loads.elem_load_offsets,
                    active_loads.elem_load_pool,
                    model.typed_nodes,
                    model.typed_sections_by_id,
                    fiber_section_defs,
                    model.fiber_section_cells,
                    model.fiber_section_index_by_id,
                    fiber_section3d_defs,
                    model.fiber_section3d_cells,
                    model.fiber_section3d_index_by_id,
                    model.uniaxial_defs,
                    model.uniaxial_state_defs,
                    uniaxial_states,
                    model.elem_uniaxial_offsets,
                    model.elem_uniaxial_counts,
                    model.elem_uniaxial_state_ids,
                    model.force_basic_offsets,
                    model.force_basic_counts,
                    force_basic_q,
                )
            elif kind == "element_basic_force":
                values = _element_basic_force_for_recorder(
                    elem_index,
                    elem,
                    model.ndf,
                    u,
                    model.typed_nodes,
                    model.uniaxial_defs,
                    model.uniaxial_state_defs,
                    uniaxial_states,
                    model.elem_uniaxial_offsets,
                    model.elem_uniaxial_counts,
                    model.elem_uniaxial_state_ids,
                    material_stress,
                )
            elif kind == "element_deformation":
                values = _element_deformation_for_recorder(
                    elem_index,
                    elem,
                    model.ndf,
                    u,
                    model.typed_nodes,
                )
            elif kind == "section_force" or want_defo:
                values = _section_response_for_recorder(
                    elem_index,
                    elem,
                    sec_no,
                    model.ndf,
                    u,
                    active_loads.element_loads,
                    active_loads.elem_load_offsets,
                    active_loads.elem_load_pool,
                    1.0,
                    model.typed_nodes,
                    model.typed_sections_by_id,
                    fiber_section_defs,
                    model.fiber_section_cells,
                    model.fiber_section_index_by_id,
                    fiber_section3d_defs,
                    model.fiber_section3d_cells,
                    model.fiber_section3d_index_by_id,
                    model.uniaxial_defs,
                    uniaxial_states,
                    model.elem_uniaxial_offsets,
                    model.elem_uniaxial_counts,
                    model.elem_uniaxial_state_ids,
                    model.force_basic_offsets,
                    model.force_basic_counts,
                    force_basic_q,
                    want_defo,
                )
            else:
                self.protocol.reply('{"error":"unknown element response"}')
                return True
            var reply = '{"values":'
            append_session_values(reply, values, 0, len(values))
            self.protocol.reply(reply + "}")
            return True
        if op == "reactions":
            var active_loads = build_active_element_load_state(
                self.const_element_loads[],
                self.pattern_element_loads[],
                committed_pattern_scale,
                model.typed_elements,
                model.elem_id_to_index,
                model.ndm,
                model.ndf,
            )
            var external = build_active_nodal_load(
                self.const_F[], self.pattern_F[], committed_pattern_scale
            )
            if self.pattern_type_tag == PatternTypeTag.UniformExcitation:
                add_uniform_excitation_load(
                    self.uniform_excitations[],
                    replay_time,
                    model.time_series,
                    model.time_series_values,
                    model.time_series_times,
                    model.M_total,
                    model.ndf,
                    model.has_transformation_mpc,
                    model.mpc_row_offsets,
                    model.mpc_dof_pool,
                    model.mpc_coeff_pool,
                    external,
                )
            var forces = assemble_internal_forces_typed_soa(
                model.typed_nodes,
                model.typed_elements,
                model.node_x,
                model.node_y,
                model.node_z,
                model.elem_dof_offsets,
                model.elem_dof_pool,
                model.elem_node_offsets,
                model.elem_node_pool,
                model.elem_primary_material_ids,
                model.elem_type_tags,
                model.elem_geom_tags,
                model.elem_section_ids,
                model.elem_integration_tags,
                model.elem_num_int_pts,
                model.elem_area,
                model.elem_thickness,
                model.frame2d_elem_indices,
                model.frame3d_elem_indices,
                model.truss_elem_indices,
                model.zero_length_elem_indices,
                model.two_node_link_elem_indices,
                model.zero_length_section_elem_indices,
                model.quad_elem_indices,
                model.shell_elem_indices,
                active_loads.element_loads,
                active_loads.elem_load_offsets,
                active_loads.elem_load_pool,
                1.0,
                model.typed_sections_by_id,
                model.typed_materials_by_id,
                model.id_to_index,
                model.node_count,
                model.ndf,
                model.ndm,
                u,
                model.uniaxial_defs,
                model.uniaxial_state_defs,
                uniaxial_states,
                model.elem_uniaxial_offsets,
                model.elem_uniaxial_counts,
                model.elem_uniaxial_state_ids,
                model.force_basic_offsets,
                model.force_basic_counts,
                force_basic_q,
                fiber_section_defs,
                model.fiber_section_cells,
                model.fiber_section_index_by_id,
                fiber_section3d_defs,
                model.fiber_section3d_cells,
                model.fiber_section3d_index_by_id,
                model.layered_shell_section_defs,
                shell_cache,
            )
            for i in range(len(forces)):
                forces[i] -= external[i]
            var reply = '{"values":'
            append_session_values(reply, forces, 0, len(forces))
            self.protocol.reply(reply + "}")
            return True
        if op == "time":
            self.protocol.reply('{"time":' + String(replay_time) + "}")
            return True
        return False

    fn result(self, steps: Int, time: Float64, extra: String) -> String:
        if self.failed:
            return (
                '{"status":-3,"steps":'
                + String(self.steps)
                + ',"time":'
                + String(self.time)
                + ',"increment":'
                + String(self.increment)
                + ',"iterations":'
                + String(self.iterations)
                + extra
                + "}"
            )
        return '{"status":0,"steps":' + String(steps) + ',"time":' + String(time) + extra + "}"


fn append_session_values(mut json: String, values: List[Float64], start: Int, count: Int):
    json += "["
    for i in range(count):
        if i > 0:
            json += ","
        json += String(values[start + i])
    json += "]"


fn append_session_node(
    mut json: String,
    node: Int,
    ndf: Int,
    u: List[Float64],
    v: List[Float64],
    a: List[Float64],
):
    json += "["
    append_session_values(json, u, node * ndf, ndf)
    json += ","
    append_session_values(json, v, node * ndf, ndf)
    json += ","
    append_session_values(json, a, node * ndf, ndf)
    json += "]"


fn transient_pattern_scale(
    pattern_type_tag: Int,
    ts_index: Int,
    time: Float64,
    time_series: List[TimeSeriesInput],
    time_series_values: List[Float64],
    time_series_times: List[Float64],
) raises -> Float64:
    if pattern_type_tag != PatternTypeTag.Plain:
        return 0.0
    if ts_index < 0:
        return 1.0
    return eval_time_series_input(
        time_series[ts_index], time, time_series_values, time_series_times
    )
