from os import abort


fn node_dof_index(node_index: Int, dof: Int, ndf: Int) -> Int:
    return node_index * ndf + (dof - 1)


fn require_dof_in_range(dof: Int, ndf: Int, context: String):
    if dof < 1 or dof > ndf:
        abort(context + " dof out of range 1.." + String(ndf))
