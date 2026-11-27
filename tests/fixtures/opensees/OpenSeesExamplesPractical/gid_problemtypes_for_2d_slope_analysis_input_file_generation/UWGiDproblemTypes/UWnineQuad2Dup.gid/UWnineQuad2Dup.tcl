proc makeCornerLayer {} {
GiD_Process Layers New cornerNodes escape \
escape \
Layers ToUse cornerNodes escape \
escape escape
}
proc findCorners {n1 n2 n3 n4} {
GiD_Process Mescape Data Conditions AssignCond Nodal_DOF_on_nodes Change 3 \
$n1 $n2 $n3 $n4 \
escape escape \
Layers Entities cornerNodes LowerEntities Nodes \
$n1 $n2 $n3 $n4 \
escape escape
}
