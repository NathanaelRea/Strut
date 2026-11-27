# ----------------------------------------------------------------
# Example 2: Dynamic analysis of a wall-frame system using SFI-MVLEM
# Created by: Kristijan Kolozvari (kkolozvari@fullerton.edu)
# Date: 7/2015
# ----------------------------------------------------------------

# *******************************************************************************************	
# ************************* BUILD MODEL AND RUN GRAVITY ANALYSIS ****************************
# *******************************************************************************************

# Clear all memory
wipe all
	
# Source display procedures
source DisplayModel2D.tcl
source DisplayPlane.tcl
	
# ************************************* INPUT START *****************************************

# ----------------------------------- Set Up Directories ------------------------------------
# Define Model Name
set modelName "5story_WF_SFI";

# Set up output directory
set dataDir $modelName;			# name of output folder
file mkdir $dataDir;			# create output folder

# ----------------------------------- Building Geometry ------------------------------------

# Define model builder
model BasicBuilder -ndm 2 -ndf 3

# Units: kips, inches, seconds

# Define basic geometry ....................................................................
# Vertical geometry
set NumStories 5;
set NumFramesPerWall 1; 
set HStory [expr 12.0*12.0];			# Story height, inches
set HTotal [expr $NumStories*$HStory];	# Total building height

# Horizontal geometry
set L1 [expr 20.0*12.0];		# Span L1, inches
set L2 [expr 20.0*12.0];		# Span L2, inches

# Wall geometry
set Lw [expr 20.0*12.0];		# Wall length, inches
set LengthBoundEl 30;			# Boundary Length
set Twall 12.0;					# Wall thickness
set WallElPerStory 2;			# Number of wall elements per story (if use other than 2 have to correct node 1xx coordinates input)
	
# Define beam geometry & parameters
set Dbeam 32.0;										# Beam Depth
set Wbeam 18.0;										# Beam Width
set Abeam [expr $Dbeam*$Wbeam];						# Cross-sectional area, full section
set Ibeam [expr 0.35*$Wbeam*pow($Dbeam,3.0)/12.0];	# Moment of inertia, cracked section at 0.35EI
set Abeam [expr $NumFramesPerWall*$Abeam];			# Multiply Area of a single beam with number of frames
set Ibeam [expr $NumFramesPerWall*$Ibeam];			# Multiply Moment of Inertia of a single beam with number of frames
set MybeamTop 11232.0;								# Yield moment at plastic hinge location, 7-#9 top
set MybeamBot 6624.0;								# Yield moment at plastic hinge location, 5-#8 bottom

# Define column geometry & parameters
set Dcol 28.0;								# Beam Depth
set Wcol 28.0;								# Beam Width
set Acol [expr $Dcol*$Wcol];				# Cross-sectional area, full section
set kcol {0.55 0.42 0.34 0.30 0.30}; 		# Array of column stiffness modifiers {level_1 level_2 ...}
set Icol [expr $Wcol*pow($Dcol,3.0)/12.0];	# Moment of inertia, uncracked
set Acol [expr $NumFramesPerWall*$Acol]; 	# Multiply Area of a single column with number of frames
set Icol [expr $NumFramesPerWall*$Icol]; 	# Multiply Moment of Inertia of a single column with number of frames
set Mycol 11250.0;							# Average col yield moment at plastic hinge location 
#(taken as the average of col My at all levels), corresponding to what axial load Pmin.

set BeamPHL [expr $Dcol/2.0]; 				# Beam plastic hinge location w.r.t. the joint node (@ face of column)
set ColumnPHL [expr $Dbeam/2.0];			# Column plastic hinge location w.r.t. the joint node (@face of beam)
	
# Define rigid beam/column properties
set A_rigid 1.0e9;
set I_rigid 1.0e9;
set Econ_rigid 1.0e9;
	
# Calculate floor masses - nodal mass
set g 386.4;								# Acceleration due to gravity, in/(sec^2)
set pi 3.141593;							# pi
set WallWeight 360.0;						# Wall weight, 0.15 ksf x tributary wall mass =0.15 ksf*60'*20'
set FrameWeight 270.0;						# Frame weight, 0.15 ksf x tributary frame mass =0.15 ksf*60'*10'
set PDWeight 360;							# P-D column weight, 0.15 psf remaining floor mass  =0.15 ksf*60'*40'
set WallNodalMass 	[expr $WallWeight/$g];	# Floor nodal mass 
set FrameNodalMass 	[expr $FrameWeight/$g];	# Floor nodal mass 
set PDNodalMass 	[expr $PDWeight/$g];	# Floor nodal mass 
set Negligible 1.0e-9;						# A very small number to avoid problems with zero
	
# Define gravity loads on beams (line) and walls (nodal)
set wLineBeam_floor 0.133;	# DL + 0.25*LL = (0.15+0.25*0.04)*10.0ft = 1.6 k/ft = 0.133 k/in
set wNodalWall_floor 128;	# DL + 0.25*LL = (0.15+0.25*0.04)*20.0ft = 3.2 k/ft => nodalLoad = 3.2k/ft*40ft = 128 kips
	
# ********************************* MODEL GENERATION *************************************

# ------------------------------------- Nodes --------------------------------------------
	
# Command: node nodeID x-coord y-coord -mass mass_dof1 mass_dof2 mass_dof3
# Ground Floor Nodes - no mass
node 100	0.0	0.0; 							# Axis 1 - COORDINATE SYSTEM ORIGIN
node 200	[expr $Lw/2.0 + $L1]	0.0; 		# Axis 2
node 300	[expr $Lw/2.0 + $L1 + $L2]	0.0; 	# Axis 3
	
for {set i 1} {$i <= $NumStories} {incr i} { 
	node [expr 100 + (2*$i - 1)]	0.0										[expr (2*$i - 1)*$HStory/$WallElPerStory];	# 1xx - Wall @ 1 - no mass at inter-story nodes
	node [expr 100 + 2*$i]			0.0										[expr $i*$HStory] -mass $WallNodalMass $Negligible $Negligible;	# 1xx - Wall   @ 1 - mass
	node [expr 200 + $i]			[expr $Lw/2.0 + $L1]					[expr $i*$HStory] -mass $FrameNodalMass $Negligible $Negligible;	# 2xx - Column @ 2 - mass
	node [expr 300 + $i]			[expr $Lw/2.0 + $L1 + $L2]				[expr $i*$HStory] -mass $FrameNodalMass $Negligible $Negligible;	# 3xx - Column @ 3 - mass
	node [expr 700 + $i]			[expr $Lw/2.0 + $L1 + $BeamPHL]			[expr $i*$HStory];  					# 7xx - Right beam plastic hinge location @ 2
	node [expr 7000 + $i]			[expr $Lw/2.0 + $L1 + $BeamPHL]			[expr $i*$HStory];  					# 7xxx - Right beam plastic hinge location @ 2
	node [expr 800 + $i]			[expr $Lw/2.0 + $L1  + $L2 - $BeamPHL]	[expr $i*$HStory];  					# 8xx - Left beam plastic hinge location @ 3
	node [expr 8000 + $i]			[expr $Lw/2.0 + $L1  + $L2 - $BeamPHL]	[expr $i*$HStory];  					# 8xxx - Left beam plastic hinge location @ 3
	node [expr 2000 +(2*$i - 1)] 	[expr $Lw/2.0 + $L1]					[expr ($i - 1)*$HStory + $ColumnPHL]; 	# 2xxx - Column plastic hinge location @ 2
	node [expr 2000 +2*$i] 			[expr $Lw/2.0 + $L1]					[expr $i*$HStory - $ColumnPHL];  		# 2xxx - Column plastic hinge location @ 2
	node [expr 20000 +(2*$i - 1)] 	[expr $Lw/2.0 + $L1]					[expr ($i - 1)*$HStory + $ColumnPHL]; 	# 2xxxx - Column plastic hinge location @ 2
	node [expr 20000 +2*$i] 		[expr $Lw/2.0 + $L1]					[expr $i*$HStory - $ColumnPHL];  		# 2xxxx - Column plastic hinge location @ 2
	node [expr 3000 +(2*$i - 1)] 	[expr $Lw/2.0 + $L1 + $L2]				[expr ($i - 1)*$HStory + $ColumnPHL]; 	# 3xxx - Column plastic hinge location @ 3
	node [expr 30000 +(2*$i - 1)] 	[expr $Lw/2.0 + $L1 + $L2]				[expr ($i - 1)*$HStory + $ColumnPHL];  	# 3xxxx - Column plastic hinge location @ 3
	node [expr 3000 +2*$i] 			[expr $Lw/2.0 + $L1 + $L2]				[expr $i*$HStory - $ColumnPHL];  		# 3xxx - Column plastic hinge location @ 3
	node [expr 30000 +2*$i] 		[expr $Lw/2.0 + $L1 + $L2]				[expr $i*$HStory - $ColumnPHL];  		# 3xxxx - Column plastic hinge location @ 3
	
}
	
# Define boundary conditions at ground nodes
fix 100 1 1 1;		# Fix node 1 in X, Y, Z-dir 
fix 200 1 1 1;		# Fix node 2 in X, Y, Z-dir
fix 300 1 1 1;		# Fix node 2 in X, Y, Z-dir

# Apply rigid diaphragm, i.e. all nodes in a floor to have the same lateral displacement
for {set i 1} {$i <= $NumStories} {incr i} { 
	equalDOF [expr 100 + 2*$i] [expr 200 + $i] 1; 
	equalDOF [expr 100 + 2*$i] [expr 300 + $i] 1; 
}
	
# Set controlling parameters for displacement controlled analysis
set IDctrlNode [expr 100 + 2*$NumStories];	# Controlling node, Right-side, roof node
set IDctrlDOF 1;							# Controlling DOF, Constrain X-dir movements
	
puts "Nodes defined."

# ----------------------------------- Material / Element Tags ------------------------------------	
# Material tags
set MatReinf 1; 	# Steel
set MatUncConc 2;	# Unconfined concrete
set MatConConc 3;	# Confined conditions
set MatFSAM_Unc 4; 	# Unconfined concrete(FSAM) - wall
set MatFSAM_Con 5;	# Confined concrete(FSAM) - wall
	
# --------------------------------------- Wall Section -------------------------------------------
# Define steel material ..........................................................................
# Command: uniaxialMaterial Steel02 $tag $fy $Es $b $R0 $cR1 $cR2 $a1 $a2 $a3 $a4

# Define reinforcement in X(horizontal), Y(vertical) in both boundary and web
set fy 60.0;		# fy
set b 0.015;		# strain hardening

# Reinforcing steel parameters
set Es 29000.0;		# Young's modulus
set R0 20.0;		# Initial value of curvature parameter
set cR1 0.925;		# Curvature degradation parameter
set cR2 0.15;		# Curvature degradation parameter
set a1 0.0;			# Default isotropic hardening parameter			
set a2 1.0;			# Default isotropic hardening parameter
set a3 0.0;			# Default isotropic hardening parameter
set a4 1.0;			# Default isotropic hardening parameter

# Build steel material
uniaxialMaterial Steel02 $MatReinf $fy  $Es $b  $R0 $cR1 $cR2 $a1 $a2 $a3 $a4

# Define concrete materials ........................................................................
# Command: uniaxialMaterial ConcreteCM $matTag $fpcc $epcc $Ec $rc $xcrn $ft $et $rt $xcrp  <-GapClose $gap>

# Unconfined concrete
set fc_uc 5.0;		# peak compressive stress 
set ec0_uc -0.0020; # strain at peak compressive stress
set Ec_uc [expr 57.0*pow($fc_uc*1000.0,0.5)];	# Young's modulus
set r_uc 15.0; 		# shape parameter - compression
set xcrn_uc 1.015;	# cracking strain - compression
set ft [expr 3.7334/1000*pow(abs($fc_uc)*1000,0.5)]; # peak tensile stress
set et 0.00008;		# strain at peak tensile stress
set rt 1.2;			# shape parameter - tension
set xcrp 10000.0;	# cracking strain - tension

# Confined concrete 
set fc_con 7.705; 	# peak compressive stress 
set ec0_con -0.0050;# strain at peak compressive stress
set Ec_con [expr 57.0*pow($fc_con*1000.0,0.5)];	# Young's modulus
set r_con 13; 		# shape parameter - compression
set xcrn_con 1.03;  # cracking strain - compression

# Build concrete materials
uniaxialMaterial ConcreteCM $MatUncConc -$fc_uc  $ec0_uc  $Ec_uc  $r_uc  $xcrn_uc  $ft $et $rt $xcrp; # Unconfined concrete
uniaxialMaterial ConcreteCM $MatConConc -$fc_con $ec0_con $Ec_con $r_con $xcrn_con $ft $et $rt $xcrp; # Confined concrete

# Define FSAM (Fixed-Strut Angle Model) ..............................................................
# Command: nDMaterial FSAM $matTag $rho $sX $sY $conc $rouX $rouY $nu $alfadow

set rho   0.0;			# Density, use 0.0 (mass assigned at nodes)
set rouXw 0.0025;		# Reinforcing in X (horizontal) direction, web
set rouYw 0.0025;		# Reinforcing in Y (vertical) direction, web
set rouXb $rouXw;		# Reinforcing in X (horizontal) direction, boundary
set rouYb 0.0693;		# Reinforcing in Y (vertical) direction, boundary (16-#11 @ each boundary)

set nu 1.0;				# Friction coefficient (0.0 < $nu < 1.0)
set alfadow 0.01;		# Stiffness coefficient of reinforcing dowel action (0.0 < $alfadow < 0.1)

# Build FSAM RC panel materials
nDMaterial FSAM $MatFSAM_Unc $rho $MatReinf $MatReinf $MatUncConc $rouXw $rouYw $nu $alfadow; 		# Unconfined concrete, web
nDMaterial FSAM $MatFSAM_Con $rho $MatReinf $MatReinf $MatConConc $rouXb $rouYb $nu $alfadow;		# Confined concrete, boundary

# Define SFI_MVLEM wall elements ......................................................................
set n_fibers 6; 	# No. of macro fibers in wall (1 per each boundary, rest for web)
set widthWebEl [expr ($Lw - 2*$LengthBoundEl)/($n_fibers-2)]; # Width of web element
set c_rot 0.4; 		# Location of center of rotation with respect to iNode (0.4 recommended)

# Command: element SFI_MVLEM eleTag iNode jNode m c -thick fiberThick -width fiberWidth -mat matTags	
for {set i 1} {$i <= [expr 2*$NumStories]} {incr i} { 		
	element SFI_MVLEM [expr 1000 + $i] [expr 100 + $i - 1] [expr 100 + $i] $n_fibers $c_rot -thick $Twall $Twall $Twall $Twall $Twall $Twall -width $LengthBoundEl $widthWebEl $widthWebEl $widthWebEl $widthWebEl $LengthBoundEl -mat $MatFSAM_Con $MatFSAM_Unc $MatFSAM_Unc $MatFSAM_Unc  $MatFSAM_Unc $MatFSAM_Con;
}

puts "Wall elements defined."

# --------------------------------------- Beam Section -----------------------------------------

# Define section tags & geometric transformation for beam-column element 
set BeamTransTag 1;
	
# Command: geomTransf Linear $transfTag <-jntOffset $dXi $dYi $dXj $dYj>
geomTransf Linear $BeamTransTag;
	
# Define beam elements
# Command: element elasticBeamColumn $eleID $iNode $jNode $A $E $I $transfID
for {set i 1} {$i <= $NumStories} {incr i} { 		
	element elasticBeamColumn  [expr 3300 + $i] [expr 700 + $i] 	[expr 800 + $i] $Abeam $Ec_uc $Ibeam $BeamTransTag; # 3xx - Beams 2-3
	element elasticBeamColumn  [expr 33300 + $i] [expr 200 + $i] 	[expr 7000 + $i] $Abeam $Ec_uc $Ibeam $BeamTransTag;# 333xx - Joint Beams @ 2R 
	element elasticBeamColumn  [expr 88800 + $i] [expr 8000 + $i] 	[expr 300 + $i] $Abeam $Ec_uc $Ibeam $BeamTransTag; # 888xx - Joint Beams @ 3L 
}
puts "Beam elements defined."

# --------------------------------------- Column Section -----------------------------------------
# Define section tags & geometric transformation for beam-column element
set ColTransTag 2;
	
# Command: geomTransf Linear $transfTag <-jntOffset $dXi $dYi $dXj $dYj>
geomTransf Linear $ColTransTag;
	
# Define column elements
# Command: element elasticBeamColumn $eleID $iNode $jNode $A $E $I $transfID
for {set i 1} {$i <= $NumStories} {incr i} { 		

	set coeff [lindex $kcol [expr $i - 1]]
	set Icol_st [expr $coeff*$Icol];
	
	element elasticBeamColumn  [expr 2000 + $i] [expr 2000 + (2*$i - 1)] [expr 2000 + 2*$i] $Acol $Ec_uc $Icol_st $ColTransTag; # 2xxx - Columns @ 2
	element elasticBeamColumn  [expr 3000 + $i] [expr 3000 + (2*$i - 1)] [expr 3000 + 2*$i] $Acol $Ec_uc $Icol_st $ColTransTag; # 3xxx - Columns @ 3
	
	element elasticBeamColumn  [expr 20000 + (2*$i - 1)] [expr 20000 + 2*$i] [expr 200 + $i] $Acol $Ec_uc $Icol_st $ColTransTag; # 200xx - Joint Column @ 2D - property? 
	element elasticBeamColumn  [expr 30000 + (2*$i - 1)] [expr 30000 + 2*$i] [expr 300 + $i] $Acol $Ec_uc $Icol_st $ColTransTag; # 300xx - Joint Column @ 3D - property? 
}
	# exceptions - bottom column joint elements 
	set coeff [lindex $kcol 0]
	set Icol_st [expr $coeff*$Icol];
	
	element elasticBeamColumn  [expr 20000] [expr 200]     [expr 20001]  $Acol $Ec_uc $Icol_st $ColTransTag; # 20000 - Joint Column @ 2U - property? 
	element elasticBeamColumn  [expr 30000] [expr 300]     [expr 30001]  $Acol $Ec_uc $Icol_st $ColTransTag; # 30000 - Joint Column @ 3U - property? 

for {set i 1} {$i <= [expr $NumStories - 1]} {incr i} { 		
	
	set coeff [lindex $kcol [expr $i]]
	set Icol_st [expr $coeff*$Icol];
	
	element elasticBeamColumn  [expr 20000 + 2*$i] [expr 200 + $i]     [expr 20000 + (2*$i + 1)]  $Acol $Ec_uc $Icol_st $ColTransTag; # 200xx - Joint Column @ 2U - property? 
	element elasticBeamColumn  [expr 30000 + 2*$i] [expr 300 + $i]     [expr 30000 + (2*$i + 1)]  $Acol $Ec_uc $Icol_st $ColTransTag; # 300xx - Joint Column @ 3U - property? 
}

puts "Column elements defined."

# --------------------------------------- BEAM Rotational Springs ------------------------------------	
# Define rotational spring properties and create spring elements using "rotSpring2DModIKModel" procedure

# rotSpring2DModIKModel creates a uniaxial material spring with a bilinear response based on Modified Ibarra Krawinkler Deterioration Model
set K0 [expr $Ec_uc*$Ibeam];	# Elastic stiffness, k*in/rad 
set as_Plus 0.01;				# Strain hardening ratio for positive loading direction
set as_Neg 0.01;				# Strain hardening ratio for negative loading direction
set Lamda_S	1.0;				# Cyclic deterioration parameter for strength deterioration [see definitions in Lignos and Krawinkler (2011)]
set Lamda_C	1.0;				# Cyclic deterioration parameter for post-capping strength deterioration [see definitions in Lignos and Krawinkler (2011)]
set Lamda_A	1.0;				# Cyclic deterioration parameter for acceleration reloading stiffness deterioration [see definitions in Lignos and Krawinkler (2011)]
set Lamda_K	1.0;				# Cyclic deterioration parameter for unloading stiffness deterioration [see definitions in Lignos and Krawinkler (2011)]
set c_S	1.0;					# Rate of strength deterioration. The default value is 1.0.
set c_C	1.0;					# Rate of post-capping strength deterioration. The default value is 1.0.
set c_A	1.0;					# Rate of accelerated reloading deterioration. The default value is 1.0.
set c_K	1.0;					# Rate of unloading stiffness deterioration. The default value is 1.0.
set theta_p_Plus 0.05;			# Pre-capping rotation for positive loading direction (often noted as plastic rotation capacity) !!! ASCE-41
set theta_p_Neg	0.05;			# Pre-capping rotation for negative loading direction (often noted as plastic rotation capacity) (must be defined as a positive value) !!!
set theta_pc_Plus 0.03;			# Post-capping rotation for positive loading direction
set theta_pc_Neg 0.03;			# Post-capping rotation for negative loading direction (must be defined as a positive value)
set Res_Pos 0.2;				# Residual strength ratio for positive loading direction
set Res_Neg	0.2;				# Residual strength ratio for negative loading direction (must be defined as a positive value)
set theta_u_Plus 0.2;			# Ultimate rotation capacity for positive loading direction
set theta_u_Neg	0.2;			# Ultimate rotation capacity for negative loading direction (must be defined as a positive value)
set D_Plus 1.0;					# Rate of cyclic deterioration in the positive loading direction (this parameter is used to create assymetric hysteretic behavior for the case of a composite beam). For symmetric hysteretic response use 1.0.
set D_Neg 1.0;					# Rate of cyclic deterioration in the negative loading direction (this parameter is used to create assymetric hysteretic behavior for the case of a composite beam). For symmetric hysteretic response use 1.0.

# Command: uniaxialMaterial ModIMKPeakOriented $matTag $K0 $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
uniaxialMaterial ModIMKPeakOriented 3002 $K0 $as_Plus $as_Neg $MybeamBot [expr -$MybeamTop] $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
	
# Create zero-length elements for plastic hinge
# Command: element zeroLength $eleTag $iNode $jNode -mat $matTag1 $matTag2 ... -dir $dir1 $dir2 ...<-doRayleigh $rFlag> <-orient $x1 $x2 $x3 $yp1 $yp2 $yp3>
# Direction 6 is rotation about local z-axis PH
for {set i 1} {$i <= $NumStories} {incr i} { 	

	# Zero Length elements - hinges
	element zeroLength [expr 700 + $i] [expr 7000 + $i] [expr 700 + $i] -mat 3002 -dir 6;	# @ 2R
	element zeroLength [expr 800 + $i] [expr 800 + $i] [expr 8000 + $i] -mat 3002 -dir 6;	# @ 3L

	# Constraint
	equalDOF [expr 700 + $i] [expr 7000 + $i] 1 2; # @ 2R
	equalDOF [expr 800 + $i] [expr 8000 + $i] 1 2; # @ 3L

}

puts "Beam plastic hinges defined."
	
# --------------------------------------- COLUMN Rotational Springs ------------------------------------	
# Define rotational spring properties and create spring elements using "rotSpring2DModIKModel" procedure

# rotSpring2DModIKModel creates a uniaxial material spring with a bilinear response based on Modified Ibarra Krawinkler Deterioration Model
set K0_c [expr $Mycol/0.01];		# Elastic stiffness, k*in/rad !!!
set as_Plus_c 0.01;					# Strain hardening ratio for positive loading direction
set as_Neg_c 0.01;					# Strain hardening ratio for negative loading direction
set Lamda_S_c	1.0;				# Cyclic deterioration parameter for strength deterioration [see definitions in Lignos and Krawinkler (2011)]
set Lamda_C_c	1.0;				# Cyclic deterioration parameter for post-capping strength deterioration [see definitions in Lignos and Krawinkler (2011)]
set Lamda_A_c	1.0;				# Cyclic deterioration parameter for acceleration reloading stiffness deterioration [see definitions in Lignos and Krawinkler (2011)]
set Lamda_K_c	1.0;				# Cyclic deterioration parameter for unloading stiffness deterioration [see definitions in Lignos and Krawinkler (2011)]
set c_S_c	1.0;					# Rate of strength deterioration. The default value is 1.0.
set c_C_c	1.0;					# Rate of post-capping strength deterioration. The default value is 1.0.
set c_A_c	1.0;					# Rate of accelerated reloading deterioration. The default value is 1.0.
set c_K_c	1.0;					# Rate of unloading stiffness deterioration. The default value is 1.0.
set theta_p_Plus_c 0.05;			# Pre-capping rotation for positive loading direction (often noted as plastic rotation capacity)
set theta_p_Neg_c	0.05;			# Pre-capping rotation for negative loading direction (often noted as plastic rotation capacity) (must be defined as a positive value)
set theta_pc_Plus_c 0.03;			# Post-capping rotation for positive loading direction
set theta_pc_Neg_c 0.03;			# Post-capping rotation for negative loading direction (must be defined as a positive value)
set Res_Pos_c 0.3;					# Residual strength ratio for positive loading direction
set Res_Neg_c	0.3;				# Residual strength ratio for negative loading direction (must be defined as a positive value)
set theta_u_Plus_c 0.2;				# Ultimate rotation capacity for positive loading direction
set theta_u_Neg_c	0.2;			# Ultimate rotation capacity for negative loading direction (must be defined as a positive value)
set D_Plus_c 1.0;					# Rate of cyclic deterioration in the positive loading direction (this parameter is used to create assymetric hysteretic behavior for the case of a composite beam). For symmetric hysteretic response use 1.0.
set D_Neg_c 1.0;					# Rate of cyclic deterioration in the negative loading direction (this parameter is used to create assymetric hysteretic behavior for the case of a composite beam). For symmetric hysteretic response use 1.0.
	
# Command: uniaxialMaterial ModIMKPeakOriented $matTag $K0 $as_Plus $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
uniaxialMaterial ModIMKPeakOriented 30002 $K0_c $as_Plus_c $as_Neg_c $Mycol [expr -$Mycol] $Lamda_S_c $Lamda_C_c $Lamda_A_c $Lamda_K_c $c_S_c $c_C_c $c_A_c $c_K_c $theta_p_Plus_c $theta_p_Neg_c $theta_pc_Plus_c $theta_pc_Neg_c $Res_Pos_c $Res_Neg_c $theta_u_Plus_c $theta_u_Neg_c $D_Plus_c $D_Neg_c
	
# Create zero-length elements for plastic hinge
# Command: element zeroLength $eleTag $iNode $jNode -mat $matTag1 $matTag2 ... -dir $dir1 $dir2 ...<-doRayleigh $rFlag> <-orient $x1 $x2 $x3 $yp1 $yp2 $yp3>
# Direction 6 is rotation about local z-axis PH
# "2Down" - @ axis 2 below the beam-column joint
for {set i 1} {$i <= $NumStories} {incr i} { 	

	# Zero Length elements - hinges
	element zeroLength [expr 200000 + (2*$i)] 		[expr 2000 + (2*$i)] 		[expr 20000 + (2*$i)] 		-mat 30002 -dir 6; 	# @ 2Down
	element zeroLength [expr 200000 + (2*$i - 1)]	[expr 20000 + (2*$i - 1)] 	[expr 2000 + (2*$i - 1)]  	-mat 30002 -dir 6; 	# @ 2Up
	
	element zeroLength [expr 300000 + (2*$i)] 		[expr 3000 + (2*$i)] 		[expr 30000 + (2*$i)] 		-mat 30002 -dir 6; 	# @ 3Down
	element zeroLength [expr 300000 + (2*$i - 1)]	[expr 30000 + (2*$i - 1)] 	[expr 3000 + (2*$i - 1)]  	-mat 30002 -dir 6; 	# @ 3Up

	# Constraint
	equalDOF [expr 2000 + (2*$i)] 		[expr 20000 + (2*$i)] 		 1 2; 	# @ 2Down
	equalDOF [expr 20000 + (2*$i - 1)] 	[expr 2000 + (2*$i - 1)]  	 1 2; 	# @ 2Up
	
	equalDOF [expr 3000 + (2*$i)] 		[expr 30000 + (2*$i)] 		 1 2; 	# @ 3Down
	equalDOF [expr 30000 + (2*$i - 1)] 	[expr 3000 + (2*$i - 1)]  	 1 2; 	# @ 3Up

}

puts "Column plastic hinges defined."
	
# --------------------------------------- Define Recorders ------------------------------------

# Define Recorders
# Recorder Node <-file $fileName> <-precision $nSD> <-time> <-dT $deltaT> <-closeOnWrite> <-node $node1 $node2...> <-nodeRange $startNode $endNode> <-dof $dof1 $dof2> $respType
# Response Type: disp, vel, accel, incrDisp, "eigen i", reaction, rayleighForces

# Node recorders
# Displacements
recorder Node -file $dataDir/NodeDisp1.out -time -nodeRange 100 110 -dof 1 2 disp; # Nodes @ 1
recorder Node -file $dataDir/NodeDisp2.out -time -nodeRange 200 205 -dof 1 2 disp; # Nodes @ 2
recorder Node -file $dataDir/NodeDisp3.out -time -nodeRange 300 305 -dof 1 2 disp; # Nodes @ 3	

# Reactions
recorder Node -file $dataDir/NodeReactions.out -time -node 100 200 300 -dof 1 2 3 reaction; 

# Record drift histories
# Command: recorder Drift -file $filename -time -iNode $NodeI_ID -jNode $NodeJ_ID -dof $dof -perpDirn $Record.drift.perpendicular.to.this.direction
recorder Drift -file $dataDir/DriftRoof.out -time -iNode 100 -jNode 110 -dof 1 -perpDirn 2; # Roof
recorder Drift -file $dataDir/DriftStory.out -time -iNode 100 102 104 106 108 -jNode 102 104 106 108 110 -dof 1 -perpDirn 2; # Story
		
# Record response history of ALL BEAM plastic hinge springs (one file for moment, one for rotation)	
# Axis 2Left
recorder Element -file $dataDir/PH_BeamMoment2R.out -time -ele 701 702 703 704 705 force;
recorder Element -file $dataDir/PH_BeamDeformation2R.out -time -ele 701 702 703 704 705 deformation;
# Axis 3Rigth
recorder Element -file $dataDir/PH_BeamMoment3L.out -time -ele 801 802 803 804 805 force;
recorder Element -file $dataDir/PH_BeamDeformation3L.out -time -ele 801 802 803 804 805 deformation;

# Record response history of ALL COLUMN plastic hinge springs (one file for moment, one for rotation)	
# Axis 2		
recorder Element -file $dataDir/PH_ColMoment2.out -time -ele 200001 200002 200003 200004 200005 200006 200007 200008 200009 200010  force;
recorder Element -file $dataDir/PH_ColDeformation2.out -time -ele 200001 200002 200003 200004 200005 200006 200007 200008 200009 200010 deformation;
# Axis 3
recorder Element -file $dataDir/PH_ColMoment3.out -time -ele 300001 300002 300003 300004 300005 300006 300007 300008 300009 300010 force;
recorder Element -file $dataDir/PH_ColDeformation3.out -time -ele 300001 300002 300003 300004 300005 300006 300007 300008 300009 300010 deformation;
	
# Record responses for wall elements
recorder Element -file $dataDir/WallGlobalForces.out -time -ele 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 globalForce; 
recorder Element -file $dataDir/WallCurvature.out -time -ele 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 Curvature;
recorder Element -file $dataDir/WallShearDef.out -time -ele 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 ShearDef; 
	
# Record responses for wall fibers (one panel per recorder)
# Command: RCPanel $fibTag $Response
recorder Element -file $dataDir/WallFiberStrain_f1.out -time -ele 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 RCPanel 1 panel_strain
recorder Element -file $dataDir/WallFiberStrain_f6.out -time -ele 1001 1002 1003 1004 1005 1006 1007 1008 1009 1010 RCPanel 6 panel_strain
recorder Element -file $dataDir/WallPanelStrain.out -time -ele 1001 RCPanel 1 panel_strain
recorder Element -file $dataDir/WallPanelStress.out -time -ele 1001 RCPanel 1 panel_stress
recorder Element -file $dataDir/WallStressConcrete.out -time -ele 1001 RCPanel 1 panel_stress_concrete
recorder Element -file $dataDir/WallStressSteel.out -time -ele 1001 RCPanel 1 panel_stress_steel
recorder Element -file $dataDir/WallPanelSteel1.out -time -ele 1001 RCPanel 1 strain_stress_steelX
recorder Element -file $dataDir/WallPanelSteel2.out -time -ele 1001 RCPanel 1 strain_stress_steelY
recorder Element -file $dataDir/WallPanelConcrete1.out -time -ele 1001 RCPanel 1 strain_stress_concrete1
recorder Element -file $dataDir/WallPanelConcrete2.out -time -ele 1001 RCPanel 1 strain_stress_concrete2

# --------------------------------- Gravity Loads & Gravity Analysis ------------------------------------	
	
# Apply gravity loads
# Construct a  time series where load factor applied is linearly proportional to the time domain
# Command: pattern PatternType $PatternID TimeSeriesType
pattern Plain 1 "Linear" {
	# Command for ndm=2 : eleLoad -ele $eleTag1 <$eleTag2...> -type -beamUniform $Wy <$Wx> 
		# Line loads on all beams
	for {set i 1} {$i <= $NumStories} {incr i} { 		
		
		eleLoad -ele [expr 3300 + $i] -type -beamUniform -$wLineBeam_floor;
		eleLoad -ele [expr 33300 + $i] -type -beamUniform -$wLineBeam_floor;
		eleLoad -ele [expr 88800 + $i] -type -beamUniform -$wLineBeam_floor;
			
	}

	# Nodal load on walls - command: load nodeID xForce yForce
	for {set i 1} {$i <= $NumStories} {incr i} { 		
			
		load [expr 100 + 2*$i]  0.0 -$wNodalWall_floor 0.0;
			
	}
	
}

# Gravity-analysis: load-controlled static analysis
set Tol 1.0e-6;							# convergence tolerance for test
set NstepGravity 10;					# apply gravity in 10 steps
set DGravity [expr 1.0/$NstepGravity];	# load increment
constraints Plain;						# how it handles boundary conditions
numberer RCM;							# renumber dof's to minimize band-width (optimization)
system BandGeneral;						# how to store and solve the system of equations in the analysis (large model: try UmfPack)
test NormDispIncr $Tol 6;				# determine if convergence has been achieved at the end of an iteration step
algorithm Newton;						# use Newton's solution algorithm: updates tangent stiffness at every iteration
integrator LoadControl $DGravity;		# determine the next time step for an analysis
analysis Static;						# define type of analysis: static or transient
analyze $NstepGravity;					# apply gravity

puts "Model built & gravity analysis completed."		