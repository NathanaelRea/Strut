#                                                                                                   
# set some variables                                                                                
#                                                                                                   

set targetDisp 1.2; # target displacement                                                           
set tol 0.01;       # tolerance                                                                     

set Amin 1.0
set Amax 10.0

#                                                                                                   
# define a procedure: getA() which calls matlab script getA.m in which our new A is computed           
#                                                                                                   

proc getA {Amin Amax} {

    # write current min and max values
    set fileID [open data1 w]
    puts $fileID "$Amin $Amax"
    close $fileID

    # invoke matlab
    if {[catch {exec matlab -nosplash -nodesktop -r "getA; quit"}]} {
	puts "Ignore this $msg"
    }

    # read matlab result
    set fileIN [open data2 r]
    set fileData [read $fileIN]
    close $fileIN

    #cleanup
    file delete data1
    file delete data2
    
    # return result
    set lines [split $fileData "\n"]
    set A [lindex $lines 0]  
}

set A [getA $Amin $Amax]

#                                                                                                   
# create the model & perform one analysis step                                                      
#                                                                                                   

model BasicBuilder -ndm 2 -ndf 2

node 1   0.0  0.0
node 2 144.0  0.0
node 3 168.0  0.0
node 4  72.0 96.0

fix 1 1 1
fix 2 1 1
fix 3 1 1
uniaxialMaterial Elastic 1 3000


element truss 1 1 4 $A 1
element truss 2 2 4 $A 1
element truss 3 3 4 $A 1

timeSeries Constant 1
pattern Plain 1 1 {
    load 4 100 -50
}

#create analysis

system BandSPD
numberer RCM
constraints Plain
integrator LoadControl 1.0
algorithm Linear
analysis Static

analyze 1

#                                                                                                   
# iterate until convergence is reached, or failure                                                  
#                                                                                                   

set xDisp [nodeDisp 4 1]
set done 0

while {$done == 0} {

    if {$xDisp > $targetDisp} {
	set Amin $A
    } else {
	set Amax $A
    }

    set diffA [expr $Amax-$Amin]
    if {$diffA  < $tol} {
	set done 2
      break
    }

    set A [getA $Amin $Amax]
    setParameter -value $A -ele A
    
    analyze 1
    set xDisp [nodeDisp 4 1]

    set diff [expr abs($xDisp - $targetDisp)]
    if {$diff < $tol} {
	set done 1
    }

    puts "$A $xDisp"
}

#                                                                                                   
# print success or failure                                                                          
#                  

if {$done == 1} {
    puts "SUCCESS: A = $A gives Displacement of $xDisp versus target of $targetDisp"
} else {
    puts "FAILURE: Hit limit A = $A gives Displacement of $xDisp versus target of $targetDisp"
}
