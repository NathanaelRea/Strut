# Runs the material behavior for externally defined material properties.
# Looks at the monotonic stress-strain behavior in compression of ReinforcingSteel

# Units: kip, in, sec
# Remove existing model
wipe

# SET UP ----------------------------------------------------------------------------
# units: kip, inch, sec
model BasicBuilder -ndm 2 -ndf 3;		# Define the model builder, ndm=#dimension, ndf=#dofs

# define GEOMETRY -------------------------------------------------------------
set LCol [expr {1.*$in}]; 		         # Column length

# define section geometry
set HCol [expr {1.*$in}]; 		         # Column Depth
set BCol [expr {1.*$in}];		         # Column Width
set A [expr {$HCol*$BCol}]

# nodal coordinates:
node 1 0 0;			                     # node#, X, Y
node 2 0 $LCol 		

# Single point constraints -- Boundary Conditions
fix 1 1 1 1; 			                  # node DX DY RZ

# Define ELEMENTS & SECTIONS  -------------------------------------------------------------
set ColMatTag 2;			              # assign a tag number to the column flexural behavior
set ColSecTag 1;				          # assign a tag number to the column section tag

uniaxialMaterial ReinforcingSteel      $ColMatTag   $w_Fy $w_Fu  $w_E $w_Esh $w_esh $w_eult  -GABuck  $w_lsr $w_beta $w_r $w_gamma -CMFatigue $w_Cf $w_alpha $w_Cd;
section Fiber $ColSecTag {
	patch rect $ColMatTag 10 10 -0.5 -0.5 0.5 0.5}

# define geometric transformation: performs a linear geometric transformation of beam stiffness and resisting force from the basic system to the global-coordinate system
set ColTransfTag 1; 			# associate a tag to column transformation
geomTransf Linear $ColTransfTag  ; 	

# element connectivity:
set numIntgrPts 5;								# number of integration points for force-based element
element nonlinearBeamColumn 1 1 2 $numIntgrPts $ColSecTag $ColTransfTag;

# Define RECORDERS -------------------------------------------------------------
recorder Node -file $dataDirSteel/CyclicLoad1.out -node 1 -dof 2  reaction
recorder Node -file $dataDirSteel/CyclicDisp2.out -time -node 2 -dof 2  disp

# Set up cyclic loading
set IDctrlNode 2;				# node where displacement is read for displacement control
set IDctrlDOF 2;				# degree of freedom of displacement read for displacement control

# characteristics of cyclic analysis	
set iDmax "1 1 2 2 3 3 5 5 10 10";		# vector of displacement-cycle peaks, in terms of multiplier on ey
set Dincr 0.0001;			         # displacement increment for pushover.
set Fact [expr {$w_Fy/$w_E}];			# scale drift ratio by ey for displacement cycles
set CycleType Full;				   # you can do Full / Push / Half cycles with the proc
set Ncycles 1;				         # specify the number of cycles at each peak

# create load pattern for lateral pushover load
set Load 1;			# define the lateral load as a proportion of the weight so that the pseudo time equals the lateral-load coefficient when using linear load pattern
set PushNode 2;			         # define nodes where lateral load is applied in static lateral analysis
pattern Plain 200 Linear {;		# define load pattern
	load $PushNode  0.0 $Load 0.0
}

# ----------- Load analysis parameters
source LibAnalysisStaticParameters.tcl;	# constraintsHandler,DOFnumberer,system-ofequations,convergenceTest,solutionAlgorithm,integrator

#  ---------------------------------    perform Static Cyclic Displacements Analysis
source LibGeneratePeaks.tcl;           # Load file that will generate displacement curve
set fmt1 "%s Cyclic analysis: CtrlNode %.2i, dof %.2i, Disp=%.4f %s";	# format for screen/file output of DONE/PROBLEM analysis
set Tol 1e-6
foreach Dmax $iDmax {
	set iDstep [GeneratePeaks $Dmax $Dincr $CycleType $Fact];	# generate displacement curve
	for {set i 1} {$i <= $Ncycles} {incr i 1} {
		set zeroD 0
		set D0 0.0
		foreach Dstep $iDstep {
			set D1 $Dstep
			set Dincr [expr $D1 - $D0]
			integrator DisplacementControl  $IDctrlNode $IDctrlDOF $Dincr
			analysis Static
			# ----------------------------------------------first analyze command------------------------
			set ok [analyze 1]
			# ----------------------------------------------if convergence failure-------------------------
			if {$ok != 0} {
				# if analysis fails, we try some other stuff
				# performance is slower inside this loop	global maxNumIterStatic;	    # max no. of iterations performed before "failure to converge" is ret'd
				if {$ok != 0} {
					puts "Trying Newton with Initial Tangent .."
					test NormDispIncr   $Tol 2000 0
					algorithm Newton -initial
					set ok [analyze 1]
					test $testTypeStatic $TolStatic      $maxNumIterStatic    0
					algorithm $algorithmTypeStatic
				}
				if {$ok != 0} {
					puts "Trying Broyden .."
					algorithm Broyden 8
					set ok [analyze 1 ]
					algorithm $algorithmTypeStatic
				}
				if {$ok != 0} {
					puts "Trying NewtonWithLineSearch .."
					algorithm NewtonLineSearch 0.8 
					set ok [analyze 1]
					algorithm $algorithmTypeStatic
				}
				if {$ok != 0} {
					set putout [format $fmt1 "PROBLEM" $IDctrlNode $IDctrlDOF [nodeDisp $IDctrlNode $IDctrlDOF] $LunitTXT]
					puts $putout
					return -1
				}; # end if
			}; # end if
			# -----------------------------------------------------------------------------------------------------
			set D0 $D1;			# move to next step
		}; # end Dstep
	};		# end i
};	# end of iDmaxCycl
# -----------------------------------------------------------------------------------------------------
if {$ok != 0 } {
	puts [format $fmt1 "PROBLEM" $IDctrlNode $IDctrlDOF [nodeDisp $IDctrlNode $IDctrlDOF] $LunitTXT]
} else {
	puts [format $fmt1 "DONE"  $IDctrlNode $IDctrlDOF [nodeDisp $IDctrlNode $IDctrlDOF] $LunitTXT]
}