# One distorted element. The support reaction is its end force at node 1.
model basic -ndm 3 -ndf 6
node 1 0.0 0.0 0.0
node 2 1.4 0.1 0.0
node 3 1.2 1.0 0.0
fix 1 1 1 1 1 1 1

nDMaterial ElasticIsotropic 1 30000.0 0.23 2.1
nDMaterial PlateFiber 2 1
section PlateFiber 1 2 0.24 Gauss

if {$shellType eq "ShellDKGT"} {
    # The DKT membrane permits rigid in-plane spin about the fixed corner.
    fix 2 0 1 0 0 0 0
    element ShellDKGT 1 1 2 3 1
} else {
    node 4 -0.07 0.8 0.0
    fix 4 1 1 1 1 1 1
    element $shellType 1 1 2 3 4 1 {*}$shellOptions
}

timeSeries Linear 1
pattern Plain 1 1 {
    load 2 0.17 0.32 0.13 -0.018 0.014 0.01
    load 3 0.42 0.14 -0.31 0.025 -0.036 0.016
}

recorder Node -precision 16 -file static_disp.out -node 2 3 -dof 1 2 3 4 5 6 disp
recorder Node -precision 16 -file static_force.out -node 1 -dof 1 2 3 4 5 6 reaction
constraints Plain
numberer Plain
system BandGeneral
test NormDispIncr 1.0e-11 40 0
algorithm Newton
integrator LoadControl 0.5
analysis Static
analyze 2

wipeAnalysis
setTime 0.0
remove loadPattern 1

recorder Node -precision 16 -file dynamic_disp.out -node 2 3 -dof 1 2 3 4 5 6 disp
constraints Plain
numberer Plain
system BandGeneral
test NormDispIncr 1.0e-11 40 0
algorithm Newton
integrator Newmark 0.5 0.25
analysis Transient
analyze 8 0.015
