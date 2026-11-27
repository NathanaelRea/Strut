# Define the materials and sections used for the coupled wall 

# Remove existing model                                                                                                       
wipe;

# Create ModelBuilder (with two-dimensions and 3 DOF/node)   
model BasicBuilder -ndm 2 -ndf 3

set geomTransfTag 1
set PDeltaTransfTag 1;
set w_SecPoi 3

set MatTag_W_St_Flange 1
set MatTag_W_St_Web 2
set MatTag_W_Con 3

set MatTag_B_St_Flange 11
set MatTag_B_St_Web 12
set MatTag_B_Con 13

# Number of fibers
set w_NOFD    50;
set w_NOFC    50;
set w_NOFt    20;

set w_SecTag 1
set b_SecTag 2

set w_e_SecTag 3
set b_e_SecTag 4

source rotSpring2DModIMKPinching.tcl
source RotPdeltaCol.tcl ;

geomTransf PDelta $PDeltaTransfTag; # USING ALL PDELTA

# Define P-delta columns and rigid links

	set TrussMatID 600;		                     	# define a material ID
	set Arigid 100000.0;		                    	# define area of truss section (make much larger than A of frame elements)
	set Irigid 1000000.0;	                          	# moment of inertia for p-delta columns  (make much larger than I of frame elements)
	uniaxialMaterial Elastic $TrussMatID [expr {1000.*$w_E}];		# define truss material


 #Wall Materials

uniaxialMaterial ReinforcingSteel      $MatTag_W_St_Flange   $w_Fy $w_Fu  $w_E $w_Esh $w_esh $w_eult  -GABuck  $w_lsr $w_beta $w_r $w_gamma
uniaxialMaterial ReinforcingSteel     $MatTag_W_St_Web    $w_Fy $w_Fu  $w_E $w_Esh $w_esh $w_eult  -GABuck  $w_lsr $w_beta $w_r $w_gamma
uniaxialMaterial Concrete02   $MatTag_W_Con   $w_fpcc $w_epcc $w_ffinal $w_efinal $w_lambda $w_ft [expr {$w_ft/$w_et}] 

 #Wall Nonlinear Section

section Fiber $w_SecTag {
  patch quad  $MatTag_W_St_Flange 1 $w_NOFt [expr $w_D/2.-$w_t1]  [expr $w_W2/2.]  [expr $w_D/2.-$w_t1] [expr -$w_W2/2.] [expr $w_D/2.] [expr -$w_W2/2.] [expr $w_D/2.] [expr $w_W2/2.];                         # Define Steel flange 1
  patch quad  $MatTag_W_St_Flange 1 $w_NOFt [expr -$w_D/2.] [expr $w_W2/2.]   [expr -$w_D/2.] [expr -$w_W2/2.]  [expr -$w_D/2.+$w_t1] [expr -$w_W2/2.] [expr -$w_D/2.+$w_t1] [expr $w_W2/2.];                    # Define Steel flange 2
  patch quad  $MatTag_W_St_Web 1 $w_NOFD  [expr -$w_D/2.+$w_t1] [expr -$w_W2/2.+$w_t2]   [expr -$w_D/2.+$w_t1] [expr -$w_W2/2.]  [expr $w_D/2.-$w_t1] [expr -$w_W2/2.]  [expr $w_D/2.-$w_t1] [expr -$w_W2/2.+$w_t2];      # Define Steel web 2
  patch quad  $MatTag_W_St_Web 1 $w_NOFD  [expr -$w_D/2.+$w_t1] [expr $w_W2/2.]   [expr -$w_D/2.+$w_t1] [expr $w_W2/2.-$w_t2]  [expr $w_D/2.-$w_t1] [expr $w_W2/2.-$w_t2]  [expr $w_D/2.-$w_t1] [expr $w_W2/2.];           # Define Steel web 1
  patch quad  $MatTag_W_Con 1 $w_NOFC  [expr -$w_D/2.+$w_t1] [expr $w_W2/2.-$w_t2]  [expr -$w_D/2.+$w_t1] [expr -$w_W2/2.+$w_t2]  [expr $w_D/2.-$w_t1] [expr -$w_W2/2.+$w_t2]  [expr $w_D/2.-$w_t1] [expr $w_W2/2.-$w_t2];  # Define core
}

# Wall Elastic Section

set w_n [expr {round(100.*$w_E/$w_Ec)/100.}]
set w_tweb_eff [expr {round(100.*(2.*$w_t2+($w_W-2.*$w_t2)/$w_n))/100.}]
set w_Ag [expr {round(100.*(2.*$w_W2*$w_t1+$w_D*$w_tweb_eff))/100.}]
set w_It [expr {round(100.*(1./12.*$w_tweb_eff*pow($w_D,3)+2.*(1./12.*$w_W2*pow($w_t1,3)+2.*$w_W2*$w_t1*pow(($w_D+$w_t1)/2.,2))))/100.}]
set w_Asteel [expr {2.*$w_t2*$w_D+2.*$w_t1*$w_W2}]
set w_Aconc [expr {$w_D*($w_W-2.*$w_t2)}]

set w_C3 [expr {min(0.9,[expr {(0.45+3.*$w_Asteel/($w_Asteel+$w_Aconc))}])}]
set w_Isteel [expr {1./12.*2.*$w_t2*pow($w_D,3)+2.*(1./12.*$w_W2*pow($w_t1,3)+2.*$w_W2*$w_t1*pow(($w_D+$w_t1)/2.,2))}]
set w_Iconc [expr {1./12.*($w_W-2.*$w_t2)*pow($w_D,3)}]
set w_Iz [expr {($w_E*$w_Isteel+$w_C3*$w_Ec*$w_Iconc)/$w_E}]

# Beam Elastic Section

set b_n [expr {round(100.*$b_E/$b_Ec)/100.}]
set b_tweb_eff [expr {round(100.*(2.*$b_t2+($b_W-2.*$b_t2)/$b_n))/100.}]
set b_Ag [expr {round(100.*(2.*$b_W2*$b_t1+$b_D*$b_tweb_eff))/100.}]
set b_It [expr {round(100.*(1./12.*$b_tweb_eff*pow($b_D,3)+2.*(1./12.*$b_W2*pow($b_t1,3)+2.*$b_W2*$b_t1*pow(($b_D+$b_t1)/2.,2))))/100.}]
set b_Asteel [expr {2.*$b_t2*$b_D+2.*$b_t1*$b_W2}]
set b_Aconc [expr {$b_D*($b_W-2.*$b_t2)}]

set b_C3 [expr {min(0.9,[expr {(0.45+3.*$b_Asteel/($b_Asteel+$b_Aconc))}])}]
set b_Isteel [expr {1./12.*2.*$b_t2*pow($b_D,3)+2.*(1./12.*$b_W2*pow($b_t1,3)+2.*$b_W2*$b_t1*pow(($b_D+$b_t1)/2.,2))}]
set b_Iconc [expr {1./12.*($b_W-2.*$b_t2)*pow($b_D,3)}]
set b_Iz [expr {($b_E*$b_Isteel+$b_C3*$b_Ec*$b_Iconc)/$b_E}]
