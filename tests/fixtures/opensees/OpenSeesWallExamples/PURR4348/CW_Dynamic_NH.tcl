#####################################################################################
#   Time History/Dynamic Analysis               			   			   #
#####################################################################################	
 
# Rayleigh Damping
# calculate damping parameters
set zeta 0.05;		                                 	# percentage of critical damping
set a0 [expr $zeta*2.0*$w1*$w2/($w1 + $w2)];			# mass damping coefficient based on first and second modes
set a1 [expr $zeta*2.0/($w1 + $w2)];				# stiffness damping coefficient based on first and second modes
#set a1_mod [expr $a1*(1.0+$n)/$n];				# modified stiffness damping coefficient used for n modified elements. See Zareian & Medina 2010.
 
# assign damping to frame beams and columns		
# command: region $regionID -eleRange $elementIDfirst $elementIDlast -rayleigh $alpha_mass $alpha_currentStiff $alpha_initialStiff $alpha_committedStiff
rayleigh $a0 0.0 $a1 0.0

#region 4 -eleRange 130 830 -rayleigh 0.0 0.0 $a1_mod 0.0;	# assign stiffness proportional damping to frame beams & columns w/ n modifications
#region 5 -eleRange 11 82 -rayleigh 0.0 0.0 $a1_mod 0.0;	# assign stiffness proportional damping to frame beams & columns w/ n modifications

#region 4 -eleRange 130 830 -rayleigh 0.0 0.0 $a1 0.0;	       	# assign stiffness proportional damping to frame beams w/ n modifications
#region 5 -eleRange 11 82 -rayleigh 0.0 0.0 $a1 0.0;		# assign stiffness proportional damping to frame columns w/ n modifications
#region 6 -nodeRange 101 802 -rayleigh $a0 0.0 0.0 0.0;		# assign mass proportional damping to structure (assign to nodes with mass)

######################################################
 
# define ground motion parameters
set patternID 1;					# load pattern ID
set GMdirection 1;					# ground motion direction (1 = x)
set GMtime [expr {$deltat*$TotalNumberOfSteps+10.0}];	# total time of ground motion + 10 sec of free vibration
set timeSeriesID 1

timeSeries Path $timeSeriesID -dt $deltat -filePath $GMfile -factor $Scalefact;
pattern UniformExcitation $patternID $GMdirection -accel $timeSeriesID;

####################################################

# define dynamic analysis parameters
# set dt_analysis $analysisstep;			# timestep of analysis
# set NumSteps [expr round(($GMtime)/$dt_analysis)];	# number of steps in analysis

###########################################################

wipeAnalysis;
constraints Plain;
numberer RCM;
system UmfPack;	
test EnergyIncr 1.0e-4 2000;
algorithm KrylovNewton;
integrator Newmark 0.5 0.25;	     			
analysis Transient;
#analyze $NumSteps $dt_analysis
set dt_analysis 0.005

set messagesfile [open "$dataDir/Messages.txt" w+];
set ok 1
foreach analysisstep $lanalysisstep {
# perform the dynamic analysis and display whether analysis was successful

# ok = 0 if analysis was completed
if {$ok == 0} {
   } else {
      puts $messagesfile "Trying: $dt_analysis";
      set dt_analysis $analysisstep
      set NumSteps [expr round(($GMtime)/$dt_analysis)];
      set ok [analyze $NumSteps $dt_analysis];
      if {$ok == 0} {
         set tarDir CWResults
         append tarDir "_RN" $RcdNum "_CN" $CptNum "_SF" $scalefactor "_TS" $dt_analysis;
         if {$tempanalysisstep != $dt_analysis} {
         file rename -force -- $dataDir $tarDir
         }
         puts "Dynamic analysis complete. Analysis step ADJUSTED: $dt_analysis";
         puts $messagesfile "Dynamic analysis complete. Analysis step ADJUSTED: $dt_analysis";
         # renaming directory
         } else {
               }
      }
}

if {$ok != 0} {
puts "Dynamic analysis didn't converged";
puts $messagesfile "Dynamic analysis didn't converged";
#file delete -force $dataDir;
}

close $messagesfile
 




