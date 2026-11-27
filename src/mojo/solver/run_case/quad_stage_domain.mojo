from collections import List
from os import abort


struct QuadStageDomain(Movable):
    var elements: List[Int]
    var beams: List[Int]
    var free: List[Int]
    var free_index: List[Int]
    var constrained: List[Bool]
    var removed_nodes: List[Bool]
    var removed_elements: List[Bool]

    fn __init__(
        out self,
        elements: List[Int],
        beams: List[Int],
        free: List[Int],
        free_index: List[Int],
        constrained: List[Bool],
    ):
        self.elements = elements.copy()
        self.beams = beams.copy()
        self.free = free.copy()
        self.free_index = free_index.copy()
        self.constrained = constrained.copy()
        self.removed_nodes = []
        self.removed_elements = []

    fn apply_removals(
        mut self,
        nodes: List[Int],
        elements: List[Int],
        active_elements: List[Int],
        active_beams: List[Int],
        node_indices: List[Int],
        element_indices: List[Int],
        elem_node_offsets: List[Int],
        elem_node_pool: List[Int],
        base_free: List[Int],
        base_constrained: List[Bool],
        ndf: Int,
    ):
        var node_count = len(base_constrained) // ndf
        self.removed_nodes.resize(node_count, False)
        self.removed_elements.resize(len(elem_node_offsets) - 1, False)
        for id in elements:
            if id < 0 or id >= len(element_indices) or element_indices[id] < 0:
                abort("staged removal refers to an unknown element")
            self.removed_elements[element_indices[id]] = True
        for id in nodes:
            if id < 0 or id >= len(node_indices) or node_indices[id] < 0:
                abort("staged removal refers to an unknown node")
            self.removed_nodes[node_indices[id]] = True

        self.elements = self._surviving_elements(active_elements, elem_node_offsets, elem_node_pool)
        self.beams = self._surviving_elements(active_beams, elem_node_offsets, elem_node_pool)

        self.free.clear()
        self.free_index = []
        self.free_index.resize(len(base_constrained), -1)
        self.constrained = base_constrained.copy()
        for i in range(node_count):
            if self.removed_nodes[i]:
                for dof in range(ndf):
                    self.constrained[i * ndf + dof] = True
        for dof in base_free:
            if not self.removed_nodes[dof // ndf]:
                self.free_index[dof] = len(self.free)
                self.free.append(dof)

    fn _surviving_elements(
        self, active_elements: List[Int], offsets: List[Int], node_pool: List[Int]
    ) -> List[Int]:
        var surviving: List[Int] = []
        for element in active_elements:
            if self.removed_elements[element]:
                continue
            for i in range(offsets[element], offsets[element + 1]):
                if self.removed_nodes[node_pool[i]]:
                    abort("remove all attached elements before removing a node")
            surviving.append(element)
        return surviving^
