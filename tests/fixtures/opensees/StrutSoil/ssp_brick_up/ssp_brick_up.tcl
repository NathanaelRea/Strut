model BasicBuilder -ndm 3 -ndf 4
node 1 0 0 0
node 2 1 0 0
node 3 1 1 0
node 4 0 1 0
node 5 0 0 1
node 6 1 0 1
node 7 1.1 1.05 1.15
node 8 0 1 1
foreach i {1 2 3 4} {fix $i 1 1 1 0}
foreach i {5 6 7 8} {fix $i 0 0 0 1}

nDMaterial ElasticIsotropic 1 10000 0.25 2
element SSPbrickUP 1 1 2 3 4 5 6 7 8 1 200000 1 0.001 0.002 0.003 0.5 1e-6 0 0 -9.81
recorder Node -file pore.out -node 1 2 3 4 -dof 4 vel
recorder Node -file displacement.out -node 5 6 7 8 -dof 1 2 3 disp
recorder Element -file stress.out -ele 1 stress

constraints Penalty 1e12 1e12
numberer RCM
system SparseGeneral
test NormDispIncr 1e-10 8
algorithm Newton
integrator Newmark 0.5 0.25
analysis Transient
analyze 5 1

setParameter -value 1e-4 -eleRange 1 1 zPerm
pattern Plain 1 Constant {
    load 7 2 0 -10 0
}
analyze 12 0.25
wipe
