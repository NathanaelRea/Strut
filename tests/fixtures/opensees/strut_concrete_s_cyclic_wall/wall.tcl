# Two ShellMITC4 panels in the X-Z plane. Units N, m, Pa.
wipe
model BasicBuilder -ndm 3 -ndf 6
node 1 0 0 0
node 2 2 0 0
node 3 0 0 1
node 4 2 0 1
node 5 0 0 2
node 6 2 0 2
fix 1 1 1 1 1 1 1
fix 2 1 1 1 1 1 1
fix 3 0 1 0 1 0 1
fix 4 0 1 0 1 0 1
fix 5 0 1 0 1 0 1
fix 6 0 1 0 1 0 1

nDMaterial ConcreteS 1 36e9 0.20 38.5e6 2.85e6 3.6e9
nDMaterial PlateFromPlaneStress 2 1 15e9
uniaxialMaterial Steel02 3 400e6 200e9 0.01 18.5 0.925 0.15
nDMaterial PlateRebar 4 3 0
nDMaterial PlateRebar 5 3 90
section LayeredShell 1 8 2 0.02 4 0.003 5 0.003 2 0.074 2 0.074 5 0.003 4 0.003 2 0.02
element ShellMITC4 1 1 2 4 3 1
element ShellMITC4 2 3 4 6 5 1

timeSeries Path 1 -dt 1 -filePath lateral_load.txt
pattern Plain 1 1 {
    load 5 200000 0 0 0 0 0
    load 6 200000 0 0 0 0 0
}

recorder Node -file tip_disp.out -time -node 5 6 -dof 1 3 disp
recorder Node -file base_reaction.out -time -node 1 2 -dof 1 3 reaction
constraints Plain
numberer RCM
system SparseGeneral
test NormDispIncr 1e-12 200
algorithm Newton
integrator LoadControl 1.0
analysis Static
analyze 48
