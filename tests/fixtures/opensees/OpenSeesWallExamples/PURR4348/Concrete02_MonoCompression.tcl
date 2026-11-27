# Runs the material behavior for externally defined material properties.
# Looks at the monotonic stress-strain behavior in compression of Concrete02

# Units: kip, in, sec

# Remove existing model                                                                                                       
wipe;

# Section Dimensions
set D 1.0;        # Section Depth (in)
set W 1.0;        # Section Width -Inside (in)
set W2 1.0;       # Section Width -Outside (in)
set t1 0.0;       # Web thickness (in)
set t2 0.0;       # Web thickness (in)

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
         patch quad  2 10 10  [expr -$D/2+$t1] [expr $W/2-$t2]  [expr -$D/2+$t1] [expr -$W/2+$t2]  [expr $D/2-$t1] [expr -$W/2+$t2]  [expr $D/2-$t1] [expr $W/2-$t2];  # Define core
}

# Create element
element dispBeamColumn 1 1 2 8 1 1

# Recorder
recorder Node -file $dataDirConc/MonoCompStress.out -node 1 -dof 1  reaction
recorder Node -file $dataDirConc/MonoCompStrain.out -node 2 -dof 1  disp

# Create time series and load pattern
set P -1.0
timeSeries Linear 1	 
pattern Plain 1 1 {
load 2 $P 0.0 0.0
}

# Analysis
constraints Plain
numberer RCM
test EnergyIncr 1.0e-6 6 0
algorithm KrylovNewton
system BandGeneral
integrator DisplacementControl 2 1 -0.001
analysis Static
analyze 1500