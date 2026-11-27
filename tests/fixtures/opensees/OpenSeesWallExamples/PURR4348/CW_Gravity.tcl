############################################################################
#              Gravity Loads & Gravity Analysis
############################################################################
# apply gravity loads
	
pattern Plain 101 Constant {
	set P_PD [expr {$StoryW/2.}];	                # point loads on leaning column nodes
	for {set i 1} {$i <= $storyn} {incr i 1} {
	load [expr {$i*1000+3}] 0.0 $P_PD 0.0;		
			}	
}

# Gravity-analysis: load-controlled static analysis
	constraints Plain;
	numberer RCM;
	system SparseGeneral -piv;
	test EnergyIncr 1.0e-4 2000;
	algorithm KrylovNewton;	
	integrator LoadControl 0.1;
	analysis Static;
	analyze 10;

# maintain constant gravity loads and reset time to zero
  loadConst -time 0.0
	
puts "Gravity Load Applied"
