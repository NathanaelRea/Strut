# Runs the material behavior for externally defined material properties.
# Looks at the monotonic stress-strain behavior in tension of ReinforcingSteel

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

# Create ModelBuilder (with two-dimensions and 3 DOF/node)   
model BasicBuilder -ndm 2 -ndf 3

# Create nodes                                     
node 1   0.0  0.0
node 2   1.0  0.0     

# Set the boundary conditions   
fix 1 1 1 1

# Geometric Transformation
geomTransf Linear 1;

#Define Material
uniaxialMaterial ReinforcingSteel      1   $w_Fy $w_Fu  $w_E $w_Esh $w_esh $w_eult  -GABuck  $w_lsr $w_beta $w_r $w_gamma

#Define Section
section Fiber 1 {
     patch quad  1 1 $NOFC  [expr -$D/2+$t1] [expr $W/2-$t2]  [expr -$D/2+$t1] [expr -$W/2+$t2]  [expr $D/2-$t1] [expr -$W/2+$t2]  [expr $D/2-$t1] [expr $W/2-$t2];  # Define core
}

element nonlinearBeamColumn     1      1      2        10            1      1

# Recorder
recorder Node -file $dataDirSteel/MonoTenLoad1.out -node 1 -dof 1  reaction
recorder Node -file $dataDirSteel/MonoTenDisp2.out -time -node 2 -dof 1  disp

# Create time series and load pattern
set P 1.0
timeSeries Linear 1	 
pattern Plain 1 1 {
load 2 $P 0.0 0.0
}

# Analysis
constraints Plain
numberer RCM
test NormDispIncr 1.0e-6 6 0
algorithm Newton
system BandGeneral
integrator DisplacementControl 2 1 0.00001
analysis Static
analyze 5000