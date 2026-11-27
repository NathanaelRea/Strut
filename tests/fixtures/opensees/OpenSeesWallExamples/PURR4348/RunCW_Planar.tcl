proc Run {GMfile {lscalefactor "1. 2. 3. 4. 5. 6. 7. 8. 9. 10. 11. 12."} {lanalysisstep "0.005 0.01 0.02"}} {
# This file runs codes associated the time history response of coupled composite plate shear wall systems. See associated ReadMe.pdf file prior to proceeding
# authors: M. Broberg, A. Varma

# The current parameters corresponding with PG-1B.

# output units are kips, in, sec.

wipe all;                           # Clear out previous analysis
set startTime [clock seconds];      # Record start time to find run time at end
source LibUnits.tcl;                # Load the file with units. This file converts inputted values to kip, in, sec

###################################################################################################################################

#   Material Parameters - Wall

###################################################################################################################################

### ReinforcingSteel ###

# General Material properties
set w_E  [expr {29000.0*$ksi}];         # Steel Modulus of elasticity (ksi) 
set w_b 0.01000;                        # Steel Hardening ratio  
set w_Esh [expr {$w_E*$w_b}];           # Steel Modulus of elasticity (ksi)          
set w_Fy [expr {50.*$ksi}];             # Yield stress of Steel (ksi)        
set w_Fu [expr {65.*$ksi}];             # Ultimate stress of Steel (ksi) 
set w_esh [expr {2.0*$w_Fy/$w_E}];      # Strain corresponding to initial strain hardening
set w_eult 0.1000;                      # Strain at Peak Stress

# Buckling parameters
set w_lsr 10.0;                          # Slenderness Ratio 
set w_beta 1.0;                         # Amplification factor for the buckled stress strain curve  
set w_r 0.65;                            # Buckling reduction factor, r can be a real number between [0.0 and 1.0] 
                                            # r=1.0 full reduction (no buckling) r=0.0 no reduction 0.0<r<1.0 linear interpolation between buckled and unbuckled curves 
set w_gamma 0.5;                        # Buckling constant 

# Fatigue Parameters
set w_Cf 0.6;                           # Fatigue ductility coefficient
set w_alpha 0.5;                        # Fatigue ductility exponent
set w_Cd 0.35;                           # Cyclic strength reduction constant

### Concrete02 ###

# Compression Parameters
set w_fpcc [expr {-6.*$ksi}];                                   # Compressive strength 
set w_epcc -0.0022286;	                                        # Strain at compressive strength
set w_Ec [expr {round(57.0*$ksi*sqrt(abs($w_fpcc/$psi)))}];     # Concrete initial modulus (used for elastic calcs)
set w_ffinal [expr {0.6*$w_fpcc}];                              # Concrete crushing strength
set w_efinal -0.008;                                            # Strain at Concrete crushing strength
set w_lambda 0.1;                                               # Ratio between unloading slope and initial slope

# Tension Parameters
set w_ft [expr {-0.1*$w_fpcc}];                                 # Tensile strength
set w_et 0.00008;	                                            # Strain at tensile strength

###################################################################################################################################

#   Material Parameters - Coupling Beam

###################################################################################################################################

### Elastic Parameters ###

# Steel Parameters
set b_E  [expr {29000.0*$ksi}];                         # Steel Modulus of elasticity (ksi) 

# Concrete Parameters
set b_fpcc [expr {-6.*$ksi}];                          # Compressive strength (f'c),(ksi)
set b_Ec [expr {round(57.*$ksi*sqrt(abs($b_fpcc/$psi)))}];     # Concrete initial modulus (used for elastic calcs)

### Concentrated Plasticity Element Parameters ### - Assumes symmetric postive and negative rotation behavior
set K0 25000000.;                   # elastic stiffness
set as_Pos 0.005;                   # strain hardening ratio for positive loading direction
set My_Plus 24000.;                 # effective yield strength for positive loading direction
set FprPos 1.0;                     # Ratio of the force at which reloading begins to force corresponding to the maximum historic deformation demand (positive loading direction)
set A_pinch 1.0;                    # Ratio of reloading stiffness
set Lamda_S 0.5;                    # Cyclic deterioration parameter for strength deterioration [E_t=Lamda_S*M_y, see Lignos and Krawinkler (2011); set Lamda_S = 0 to disable this mode of deterioration]
set Lamda_C 1.0;                    # Cyclic deterioration parameter for post-capping strength deterioration [E_t=Lamda_C*M_y, see Lignos and Krawinkler (2011); set Lamda_C = 0 to disable this mode of deterioration]
set Lamda_A 1.0;                    # Cyclic deterioration parameter for accelerated reloading stiffness deterioration [E_t=Lamda_A*M_y, see Lignos and Krawinkler (2011); set Lamda_A = 0 to disable this mode of deterioration]
set Lamda_K 1.0;                    # Cyclic deterioration parameter for unloading stiffness deterioration [E_t=Lamda_K*M_y, see Lignos and Krawinkler (2011); set Lamda_K = 0 to disable this mode of deterioration]
set c_S 1.0;                        # rate of strength deterioration. The default value is 1.0.
set c_C 0.0;                        # rate of post-capping strength deterioration. The default value is 1.0.
set c_A 1.0;                        # rate of accelerated reloading deterioration. The default value is 1.0.
set c_K 1.0;                        # rate of unloading stiffness deterioration. The default value is 1.0.
set theta_p_Plus 0.025;             # pre-capping rotation for positive loading direction (often noted as plastic rotation capacity)
set theta_pc_Plus 0.04;             # post-capping rotation for positive loading direction
set Res_Pos 0.25;                   # residual strength ratio for positive loading direction
set theta_u_Plus 0.05;              # ultimate rotation capacity for positive loading direction
set D_Plus 1.0;                     # rate of cyclic deterioration in the positive loading direction (this parameter is used to create assymetric hysteretic behavior for the case of a composite beam). For symmetric hysteretic response use 1.0.

set E [expr {29000.*$ksi}]

puts "Material Defined"

###################################################################################################################################

#   Define Wall Geometry

###################################################################################################################################

                                             # build section
                                             
#                                                   Z (axis of bending)
#                            t1                     ^                     
#                          '---'                    |                    '---'        _  
#                          |   |                    |                    |   |        |
#                     -    |   |-----------------------------------------|   |  -     |
#                     |    | S |               Steel web     1           | S |  | t2  |
#                     |    | t |-----------------------------------------| t |  -     |
#                     |    | e |                                         | e |        |
#                     |    | e |                Concrete                 | e |        |
#   W (Inside width)  |    | l |                  Core                   | l |         
#                     |    |   |                    +                    |   |     W2 (Outside width) -------> -Y
#                     |    | f |                                         | f |         
#                     |    | l |                                         | l |        |
#                     |    | a |-----------------------------------------| a |        |    
#                     |    | n |               Steel web     2           | n |        |
#                     _    | g |-----------------------------------------| g |        |     
#                          | e |                                         | e |        |
#                          | 1 |                                         | 2 |        |
#                          '---'                                         '---'        _
#                              |-----------------   D     ---------------|

set storyn 8;                           # Number of stories
set h1 [expr {17.0*$ft}];               # Height of first story
set htyp [expr {14.0*$ft}];             # Height of typical story
set StoryW [expr {2880.*$kip}];         # Story weight

### Wall Panel Dimensions ###

set w_D [expr {132.*$in}];              # Section Depth (in)
set w_W2 [expr {24.*$in}];              # Section Width -Outside (in)
set w_W [expr {$w_W2}];                 # Section Width -Inside (in)
set w_t1 [expr {0.5625*$in}];           # FLANGE thickness (in)
set w_t2 [expr {0.5625*$in}];           # Web thickness (in)

### Coupling Beam Dimensions ###

set b_D [expr {24.0*$in}];              # Section Depth (in)
set b_W2 [expr {24.0*$in}];             # Section Width -Outside (in)
set b_W [expr {$b_W2}];                 # Section Width -Inside (in)
set b_t1 [expr {0.5*$in}];              # FLANGE thickness (in)
set b_t2 [expr {0.375*$in}];            # Web thickness (in)
set b_Length [expr {8.0*$ft}];          # Total specimen length

### Discretization Parameters ###
# Note the nonlinear behavior is restricted to hinge regions defined below

set w_HL [expr {$w_D/2.0}];             # Hinge Length in the Wall
set w_nle 4.0;                          # Number of elements over the hinge length - walls(must be greater than 0; should be an integer)

puts "Wall Geometry Defined\n"

###################################################################################################################################

#   Ground Motion Information 

###################################################################################################################################

set fields [split $GMfile "-"]
lassign $fields RcdNum CptNum deltat temp

set fields [split $temp "."]
lassign $fields TotalNumberOfSteps temp

###################################################################################################################################

#   Directory Setup

###################################################################################################################################

# Set up data directories for the results. If the data directory described below currently exists, the program asks for user input and either deletes the old files or quits

foreach scalefactor $lscalefactor {

set Scalefact $scalefactor
set tempanalysisstep [lindex $lanalysisstep 0]
set dataDir CWResults
append dataDir "_RN" $RcdNum "_CN" $CptNum "_SF" $scalefactor "_TS" $tempanalysisstep
set dataDirSteel $dataDir/Steel
set dataDirConc $dataDir/Concrete
set dataDirMono $dataDir/Monotonic
set dataDirEq $dataDir/Earthquake
set dataDirMC $dataDir/SectionMomentCurvataure
if { [file exists $dataDir]==1 } {
    puts "The directory $dataDir currently exits. To continue and delete type 1; to exit without continuing type 0."
    set var [gets stdin]
    if { $var == 1} {
         file delete -force -- $dataDir
         } else {
            exit
            }
};
file mkdir $dataDir;
file mkdir $dataDirSteel;
file mkdir $dataDirConc;
file mkdir $dataDirMono;
file mkdir $dataDirEq;
file mkdir $dataDirMC;

###################################################################################################################################

#   Symmetric coupling beam behavior enforced

###################################################################################################################################

# Assumption of symmetric behavior in the hinge enforced
set as_Neg $as_Pos;                 # strain hardening ratio for negative loading direction
set My_Neg [expr {-$My_Plus}];      # effective yield strength for negative loading direction
set FprNeg $FprPos;                 # Ratio of the force at which reloading begins to force corresponding to the absolute maximum historic deformation demand (negative loading direction)
set theta_p_Neg $theta_p_Plus;      # pre-capping rotation for negative loading direction (often noted as plastic rotation capacity) (must be defined as a positive value)
set theta_pc_Neg $theta_pc_Plus;    # post-capping rotation for negative loading direction (must be defined as a positive value)
set Res_Neg $Res_Pos;               # residual strength ratio for negative loading direction (must be defined as a positive value)
set theta_u_Neg $theta_u_Plus;      # ultimate rotation capacity for negative loading direction (must be defined as a positive value)
set D_Neg $D_Plus;                  # rate of cyclic deterioration in the positive loading direction (this parameter is used to create assymetric hysteretic behavior for the case of a composite beam). For symmetric hysteretic response use 1.0.

###################################################################################################################################

#   Monotonic and cyclic stress-strain of wall steel

###################################################################################################################################

source Steel_MonoTension.tcl
wipe
source Steel_MonoCompression.tcl
wipe
puts "Steel Monotonic Behavior Recorded.\n"

source Steel_Cyclic.tcl
wipe
puts "Steel Cyclic Behavior Recorded.\n"

####################################################################################################################################

#   Monotonic and cyclic stress-strain of wall concrete

####################################################################################################################################

source Concrete02_MonoCompression.tcl
wipe
source Concrete02_MonoCompression.tcl
wipe
puts "Concrete Monotonic Behavior Recorded.\n"

source Concrete02_Cyclic.tcl
wipe
puts "Concrete Cyclic Behavior Recorded.\n"

####################################################################################################################################

#   Wall Moment Curvature

####################################################################################################################################

source CW_Materials.tcl
source MomentCurvature2D.tcl
MomentCurvature2D $w_SecTag 0.0 0.1 $dataDirMC 100
wipe
puts "Wall Section Moment Curvature Recorded.\n"

###################################################################################################################################

#   Structure Eigenvalue Analysis

###################################################################################################################################

set dataDirNow $dataDirMono
source CW_Materials.tcl
source CW_Geometry-Planar.tcl
source CW_Eigen_8.tcl
wipe
puts "Eigenvalue analysis performed.\n"

###################################################################################################################################

#   Time History Analysis

###################################################################################################################################

set dataDirNow $dataDirEq
source CW_Materials.tcl
source CW_Geometry_Planar.tcl
source CW_Gravity.tcl
source CW_Dynamic_NH.tcl
wipe
puts "Coupled Wall Earthquake Response Recorded.\n"

# Calculate and report total time
set endTime [clock seconds]
set duration [expr {round(($endTime-$startTime)/60.*100.)/(100.)}]
puts "Analysis took $duration minutes."
}
}

