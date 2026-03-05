from collections import List
from os import abort

from solver.run_case.input_types import ElementInput


fn _append_unique(mut items: List[Int], value: Int):
    for i in range(len(items)):
        if items[i] == value:
            return
    items.append(value)


fn _append_unique_row(mut rows: List[List[Int]], row_index: Int, value: Int):
    for i in range(len(rows[row_index])):
        if rows[row_index][i] == value:
            return
    rows[row_index].append(value)


fn _elem_node_index(elem: ElementInput, idx: Int) -> Int:
    if idx == 0:
        return elem.node_index_1
    if idx == 1:
        return elem.node_index_2
    if idx == 2:
        return elem.node_index_3
    return elem.node_index_4


fn build_node_adjacency_typed(
    elements: List[ElementInput], node_count: Int
) -> List[List[Int]]:
    var adjacency: List[List[Int]] = []
    for _ in range(node_count):
        var row: List[Int] = []
        row.resize(0, 0)
        adjacency.append(row^)

    for e in range(len(elements)):
        var elem = elements[e]
        var node_len = elem.node_count
        if node_len < 2:
            continue
        var idxs: List[Int] = []
        idxs.resize(node_len, 0)
        for i in range(node_len):
            var idx = _elem_node_index(elem, i)
            if idx < 0 or idx >= node_count:
                abort("element node not found")
            idxs[i] = idx
        for i in range(node_len):
            var a = idxs[i]
            for j in range(i + 1, node_len):
                var b = idxs[j]
                if a == b:
                    continue
                _append_unique(adjacency[a], b)
                _append_unique(adjacency[b], a)

    return adjacency^


fn _sort_by_degree(mut neighbors: List[Int], degrees: List[Int]):
    var n = len(neighbors)
    for i in range(n):
        var min_idx = i
        var min_deg = degrees[neighbors[i]]
        for j in range(i + 1, n):
            var deg = degrees[neighbors[j]]
            if deg < min_deg:
                min_deg = deg
                min_idx = j
        if min_idx != i:
            var tmp = neighbors[i]
            neighbors[i] = neighbors[min_idx]
            neighbors[min_idx] = tmp


fn rcm_order(adjacency: List[List[Int]]) -> List[Int]:
    var n = len(adjacency)
    var degrees: List[Int] = []
    degrees.resize(n, 0)
    for i in range(n):
        degrees[i] = len(adjacency[i])

    var visited: List[Bool] = []
    visited.resize(n, False)
    var order: List[Int] = []

    for _ in range(n):
        var start = -1
        var min_deg = 0
        for i in range(n):
            if visited[i]:
                continue
            var deg = degrees[i]
            if start < 0 or deg < min_deg:
                start = i
                min_deg = deg
        if start < 0:
            break

        var queue: List[Int] = []
        queue.append(start)
        visited[start] = True
        var q_head = 0
        while q_head < len(queue):
            var v = queue[q_head]
            q_head += 1
            order.append(v)
            var neighbors = adjacency[v].copy()
            _sort_by_degree(neighbors, degrees)
            for k in range(len(neighbors)):
                var nb = neighbors[k]
                if not visited[nb]:
                    visited[nb] = True
                    queue.append(nb)

    var rcm: List[Int] = []
    rcm.resize(len(order), 0)
    if len(order) == 0:
        return rcm^
    var out_idx = 0
    var i = len(order) - 1
    while True:
        rcm[out_idx] = order[i]
        out_idx += 1
        if i == 0:
            break
        i -= 1

    return rcm^


fn min_degree_order(adjacency: List[List[Int]]) -> List[Int]:
    var n = len(adjacency)
    var active: List[Bool] = []
    active.resize(n, True)
    var graph: List[List[Int]] = []
    for i in range(n):
        graph.append(adjacency[i].copy())
    var order: List[Int] = []

    for _ in range(n):
        var chosen = -1
        var min_degree = 0
        for node in range(n):
            if not active[node]:
                continue
            var degree = 0
            for k in range(len(graph[node])):
                var nb = graph[node][k]
                if active[nb]:
                    degree += 1
            if chosen < 0 or degree < min_degree:
                chosen = node
                min_degree = degree
        if chosen < 0:
            break

        order.append(chosen)
        active[chosen] = False

        var active_neighbors: List[Int] = []
        for k in range(len(graph[chosen])):
            var nb = graph[chosen][k]
            if active[nb]:
                active_neighbors.append(nb)

        # Approximate minimum-degree fill-in by connecting active neighbors.
        for i in range(len(active_neighbors)):
            var a = active_neighbors[i]
            for j in range(i + 1, len(active_neighbors)):
                var b = active_neighbors[j]
                if a == b:
                    continue
                _append_unique_row(graph, a, b)
                _append_unique_row(graph, b, a)

    return order^
