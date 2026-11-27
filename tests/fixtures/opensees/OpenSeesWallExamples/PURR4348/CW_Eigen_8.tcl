############################################################################
#                       Eigenvalue Analysis                    			   
############################################################################
set pi [expr 2.0*asin(1.0)];						# Definition of pi
set nEigenI 1;								# mode i = 1
set nEigenJ 2;								# mode j = 2

set lambdaN [eigen -fullGenLapack $nEigenJ];				# eigenvalue analysis for nEigenJ modes

set lambdaI [lindex $lambdaN [expr 0]];					# eigenvalue mode i = 1
set lambdaJ [lindex $lambdaN [expr $nEigenJ-1]];			# eigenvalue mode j = 2

set w1 [expr {pow($lambdaI,0.5)}];					# w1 (1st mode circular frequency)
set w2 [expr {pow($lambdaJ,0.5)}];					# w2 (2nd mode circular frequency)

set T1 [expr {2.0*$pi/$w1}];						# 1st mode period of the structure
set T2 [expr {2.0*$pi/$w2}];						# 2nd mode period of the structure

puts "T1 = $T1 s";							# display the first mode period in the command window
puts "T2 = $T2 s";							# display the second mode period in the command window

set f1 [nodeEigenvector 1001 1 1]
set f2 [nodeEigenvector 2001 1 1]
set f3 [nodeEigenvector 3001 1 1]
set f4 [nodeEigenvector 4001 1 1]
set f5 [nodeEigenvector 5001 1 1]
set f6 [nodeEigenvector 6001 1 1]
set f7 [nodeEigenvector 7001 1 1]
set f8 [nodeEigenvector 8001 1 1]

set outfile [open "$dataDir/Period.txt" w+]
puts $outfile "T1 $T1\nT2 $T2"
puts $outfile "f1 = $f1";
puts $outfile "f2 = $f2";
puts $outfile "f3 = $f3";
puts $outfile "f4 = $f4";
puts $outfile "f5 = $f5";
puts $outfile "f6 = $f6";
puts $outfile "f7 = $f7";
puts $outfile "f8 = $f8";
# puts $outfile "ybar=$ybar";
# puts $outfile "TotalD=$TotalD";
close $outfile