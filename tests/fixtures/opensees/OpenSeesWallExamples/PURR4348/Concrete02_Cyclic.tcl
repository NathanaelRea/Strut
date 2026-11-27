# Runs the material behavior on Concrete02 for externally defined material properties.

# Units: kip, in, sec

# Remove existing model                                                                                                       
wipe;

# Section Dimensions
set D 1.0;        # Section Depth (in)
set W 1.0;        # Section Width -Inside (in)
set W2 1.0;       # Section Width -Outside (in)
set t1 0.0;       # Web thickness (in)
set t2 0.0;       # Web thickness (in)

# Number of fibers across section
set NOFC    20;

set SecPoi 2

# Create ModelBuilder (with two-dimensions and 3 DOF/node)   
model BasicBuilder -ndm 2 -ndf 3

# Create nodes                                     
node 1   0.0  0.0
node 2   1.0  0.0     

# Set the boundary conditions   
fix 1 1 1 1

# Geometric Transformation
geomTransf Linear 1;

# Define Material
uniaxialMaterial Concrete02   2   $w_fpcc $w_epcc $w_ffinal $w_efinal $w_lambda $w_ft [expr {$w_ft/$w_et}]

# Define Section
section Fiber 1 {
         patch quad  2 1 $NOFC  [expr -$D/2+$t1] [expr $W/2-$t2]  [expr -$D/2+$t1] [expr -$W/2+$t2]  [expr $D/2-$t1] [expr -$W/2+$t2]  [expr $D/2-$t1] [expr $W/2-$t2];  # Define core
}

element dispBeamColumn 1 1 2 8 1 1

# Recorder
recorder Node -file $dataDirConc/CyclicStress.txt -node 1 -dof 1  reaction
recorder Node -file $dataDirConc/CyclicStrain.txt -node 2 -dof 1  disp

# Create time series and load pattern
set P 1.0
timeSeries Linear 1	 
pattern Plain 1 1 {

load 2 $P 0.0 0.0
}

# Analysis
constraints Lagrange
numberer Plain
system BandGeneral
test EnergyIncr 1.e-7 60
algorithm Newton

# Loop to capture cyclic behavior
set disploop [list 1 1.1 2 2.2 3 3.3 5 5.5 9 9.9]
for {set i 1} {$i<=[llength $disploop]} {incr i 1} {
         set dispincr [expr 1.0*pow((-1.0),[expr $i-1])*[lindex $disploop [expr $i-1]]/10000]
         integrator DisplacementControl 2 1 [expr -1*$dispincr];
         analysis Static
         analyze 100
}