# ----------------------------------------------------
# Modal Analysis
# ----------------------------------------------------
	
# Generate the model and run gravity analysis	
source 5storyWF_SFI.tcl

# --------------------------------- Rayleigh Damping ------------------------------------
# Apply Rayleigh damping from $xDamp: (http://opensees.berkeley.edu/OpenSees/manuals/usermanual/1099.htm)
# D=$alphaM*M + $betaKcurr*Kcurrent + $betaKcomm*KlastCommit + $betaKinit*$Kinitial
	set xDamp 0.02;										# Damping ratio
	set MpropSwitch 1.0;								# Type 1.0 for each M and K matrix you want damping matrix to be proportional to.
	set KcurrSwitch 1.0;								# Use this: tangent stiffness changes per time series in dynamic nonlinear analysis
	set KcommSwitch 0.0;								
	set KinitSwitch 0.0;
	set nEigenI 1;										# Mode i: 1
	set nEigenJ 3;										# Mode j: 3
	#OLD: set lambdaN [eigen -generalized -fullGenLapack $nEigenJ];
	set lambdaN [eigen $nEigenJ]
	set lambdaI [lindex $lambdaN [expr $nEigenI-1]]; 	# Eigenvalue mode i
	set lambdaJ [lindex $lambdaN [expr $nEigenJ-1]]; 	# Eigenvalue mode j
	set omegaI [expr pow($lambdaI,0.5)];
	set omegaJ [expr pow($lambdaJ,0.5)];
	set alphaM [expr $MpropSwitch*$xDamp*(2*$omegaI*$omegaJ)/($omegaI+$omegaJ)];	# M-prop. damping; D = alphaM*M
	set betaKcurr [expr $KcurrSwitch*2.0*$xDamp/($omegaI+$omegaJ)];         		# Current-K + betaKcurr*KCurrent
	set betaKcomm [expr $KcommSwitch*2.0*$xDamp/($omegaI+$omegaJ)];   				# Last-committed K + betaKcomm*KlastCommitt
	set betaKinit [expr $KinitSwitch*2.0*$xDamp/($omegaI+$omegaJ)];         		# initial- K + betaKinit*Kini

# Eigen analysis - for period	
	set T {};
	foreach lam $lambdaN {
		lappend Tperiod [expr (2.0*$pi)/sqrt($lam)];
	}
	puts "T1 = [lindex $Tperiod 0] s"
	puts "T2 = [lindex $Tperiod 1] s"
	
# Apply reyleigh damping 
rayleigh $alphaM $betaKcurr $betaKinit $betaKcomm; 	

# Display Deformed Shape
set ViewScale 10;					    				# Amplify display of deformed shape
DisplayModel2D nill $ViewScale;							# Display optional