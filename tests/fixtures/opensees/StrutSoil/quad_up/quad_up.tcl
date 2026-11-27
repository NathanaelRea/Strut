model BasicBuilder -ndm 2 -ndf 3
node 1 0 0
node 2 1 0
node 3 1 1
node 4 0 1
fix 1 1 1 0
fix 2 1 1 0
fix 3 1 0 1
fix 4 1 0 1

nDMaterial ElasticIsotropic 1 10000 0.25 2
element quadUP 1 1 2 3 4 1.5 1 200000 1 1e-3 1e-3 0 -9.81 2
recorder Node -file pore.out -node 1 2 -dof 3 vel
recorder Node -file displacement.out -node 3 4 -dof 2 disp
recorder Element -file stress.out -ele 1 material 1 stress

constraints Transformation
numberer RCM
system BandGeneral
test NormDispIncr 1e-10 8
algorithm Newton
integrator Newmark 0.5 0.25
analysis Transient
analyze 5 1

parameter 10 element 1 vPerm
updateParameter 10 1e-4
pattern Plain 1 Constant {
    load 3 0 -10 0
    load 4 0 -10 0
}
analyze 12 0.25

wipeAnalysis
pattern MultipleSupport 2 {
    groundMotion 1 Plain -vel "Constant -factor 3.0"
    imposedMotion 2 3 1
}
pattern UniformExcitation 3 2 -accel "Sine 0.0 10.0 0.7 -shift 0.2 -factor 0.3"
pattern UniformExcitation 4 2 -accel "Sine 0.0 10.0 1.1 -factor 0.2"
constraints Transformation
numberer RCM
system BandGeneral
test NormDispIncr 1e-10 8
algorithm Newton
integrator Newmark 0.5 0.25
analysis Transient
analyze 8 0.05
wipe
