# ----------------------------------------------------------------
# Dynamic Time-History Analysis
# ----------------------------------------------------------------

# Perform modal analysis
source Modal.tcl

# First, set gravity loads acting constant and time in domain to 0.0
	loadConst -time 0.0
	
# Define ground motion parameters
	set GMfile "MCEScaledEQ1X.acc";					# Ground motion filename, input is in 'g' units
	set DtSeries	0.04;							# time-step Dt for definition of time series
	set NSteps 		2201.0;							# Number of steps in ground motion

# Define ground motion parameters
	set patternID 2;								# Load pattern ID
	set GMdirection 1;								# Ground motion direction (1 = x)
	set Scalefactor [expr 1*$g];					# Ground motion scaling factor

	set DtAnalysis	0.04;							# time-step Dt for lateral analysis
	set TmaxAnalysis [expr $DtSeries*$NSteps];		# maximum duration of ground-motion analysis -- should be 50*$sec
	
# Define the acceleration series for the ground motion
	# Command: "Series -dt $timestep_of_record -filePath $filename_with_acc_history -factor $scale_record_by_this_amount"
	set accelSeries "Series -dt $DtSeries -filePath $GMfile -factor $Scalefactor";
	
# Create load pattern: apply acceleration to all fixed nodes with UniformExcitation
	# Command: pattern UniformExcitation $patternID $GMdir -accel $timeSeriesID 
	pattern UniformExcitation $patternID $GMdirection -accel $accelSeries;	
	
	puts "GM parameters and acceleration series defined."
	
	set start [clock seconds]
	
	# Convergence Test
	set Tol 1.e-3;                      # tolerance
	set maxNumIter 1000;              	# maximum number of iterations  
	set printFlag 0;                	# print convergence information flag 
	set TestType NormDispIncr;			# test type
	
	# Algorithm
	set algorithmType ModifiedNewton	
 	
	# Newmark-integrator parameters 
	set NewmarkGamma 0.5;				# gamma  
	set NewmarkBeta 0.25;				# beta
	
	constraints Transformation; 
	numberer RCM
	system BandGeneral
	test $TestType $Tol $maxNumIter; # $printFlag
	algorithm $algorithmType;     
	integrator Newmark $NewmarkGamma $NewmarkBeta 	
	analysis Transient
	
	set Nsteps [expr int($TmaxAnalysis/$DtAnalysis)];	
	set ok [analyze $Nsteps $DtAnalysis];	# Perform analysis: ok=0 (successful)

if {$ok != 0} { ;		
	# If analysis was not successful change 
	# analysis parameters to achieve convergence.
	set ok 0;
	set controlTime [getTime];
	while {$controlTime < $TmaxAnalysis && $ok == 0} {
		set ok [analyze 1 $DtAnalysis]
		set controlTime [getTime]
		set ok [analyze 1 $DtAnalysis]
		if {$ok != 0} {
			puts "Trying Newton with Initial Tangent .."
			test NormDispIncr   $Tol 1000  0
			algorithm Newton -initial
			set ok [analyze 1 $DtAnalysis]
			test $TestType $Tol $maxNumIter  0
			algorithm $algorithmType
		}
		if {$ok != 0} {
			puts "Trying Broyden .."
			algorithm Broyden 8
			set ok [analyze 1 $DtAnalysis]
			algorithm $algorithmType
		}
		if {$ok != 0} {
			puts "Trying NewtonWithLineSearch .."
			algorithm NewtonLineSearch .8
			set ok [analyze 1 $DtAnalysis]
			algorithm $algorithmType
		}
		if {$ok != 0} {
			puts "Trying BFGS.."
			algorithm BFGS
			set ok [analyze 1 $DtAnalysis]
			algorithm $algorithmType
		}
	}
};      # end if ok !0

puts "Ground Motion Done."

	set currentTime [getTime];	# get current analysis time	
	puts "The current time is: $currentTime";	
	
# Print the state at control node
	print node $IDctrlNode