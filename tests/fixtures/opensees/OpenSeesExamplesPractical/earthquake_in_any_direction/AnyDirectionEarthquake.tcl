
############################################################################
#                                                                          #
#      3D Elastic Frame Model Subjected to Any Direction of Earthquake     #
#                                                                          #
############################################################################
#                                                                          #
# Written By: Seyed Mohammad Javad Foroughi Moghadam (SMJ.Foroughi)        #
# As a part of OpenSEES Book (in Persian): www.opensees.ir                 #
# Date: April 2017                                                         #
# Any Comment: smj.foroughi@gmail.com                                      #
#                                                                          #
# Units N,m,Sec                                                            #
#                                                                          #
############################################################################


wipe ;

model BasicBuilder -ndm 3 -ndf 6 ;

# Define Parameters
set L 7.0 ;
set S 4.0 ;
set H1 3.0 ;
set H2 3.0 ;

set Alpha 90.0 ;

set Pi 3.14 ;
set g 9.91 ;

# Loads (Kg/m2)
set DL 500.0
set LL 200.0

set ang [expr int($Alpha)] ;
set FileName "OutPut_$ang" ;
file mkdir $FileName ;


# Define Nodes
set AlphaRad [expr $Alpha*$Pi/180.0] ;
set Beta [expr atan($S/$L)] ;
set Teta [expr $AlphaRad+$Beta] ;
set D [expr [expr pow((pow($S,2.0)+pow($L,2.0)),0.5)]] ;

# On Ground
#    Tag   X                            Y                             Z            
node  1    0.0                          0.0                           0.0 ;
node  2   [expr $L*cos($AlphaRad)]      [expr $L*sin($AlphaRad)]      0.0 ;
node  3   [expr $D*cos($Teta)]          [expr $D*sin($Teta)]          0.0 ;
node  4   [expr -$S*sin($AlphaRad)]     [expr $S*cos($AlphaRad)]      0.0 ;

# First Floor
#    Tag  X                             Y                             Z            
node 101 0.0                            0.0                           $H1 ;
node 102 [expr $L*cos($AlphaRad)]       [expr $L*sin($AlphaRad)]      $H1 ;
node 103 [expr $D*cos($Teta)]           [expr $D*sin($Teta)]          $H1 ;
node 104 [expr -$S*sin($AlphaRad)]      [expr $S*cos($AlphaRad)]      $H1 ;

# Second Floor
#    Tag  X                             Y                             Z            
node 201 0.0                            0.0                           [expr $H1+$H2] ;
node 202 [expr $L*cos($AlphaRad)]       [expr $L*sin($AlphaRad)]      [expr $H1+$H2] ;
node 203 [expr $D*cos($Teta)]           [expr $D*sin($Teta)]          [expr $H1+$H2] ;
node 204 [expr -$S*sin($AlphaRad)]      [expr $S*cos($AlphaRad)]      [expr $H1+$H2] ;


fix 1 1 1 1 1 1 1 ;
fix 2 1 1 1 1 1 1 ;
fix 3 1 1 1 1 1 1 ;
fix 4 1 1 1 1 1 1 ;

# Define Elements
geomTransf PDelta 1 [expr -cos($AlphaRad)]  [expr -sin($AlphaRad)] 0 ;

geomTransf Linear 2 0 0 1 ;
#geomTransf PDelta 1 -1 0 0 ;


# IPE270
set Iy_B 5790.0E-8 ;
set Iz_B 420.0E-8 ;
set J_B [expr $Iz_B+$Iy_B] ;
set A_B 45.9E-4 ;

# IPB300
set Iy_C 25170.0E-8 ;
set Iz_C 8560.0E-8 ;
set J_C [expr $Iz_C+$Iy_C] ;
set A_C 194.0E-4 ;

set RSteel 7850.0 ;
set E 2.001E11 ;
set G [expr $E/(2.0*(1.0+0.3))] ;


# Columns
set mCol [expr $A_C*$RSteel] ;
# First Floor
#                        Tag iNode jNode Area  E  G    J    Iy    Iz  Trans  Mass
element elasticBeamColumn 11    1   101  $A_C $E $G  $J_C  $Iy_C $Iz_C 1  -mass $mCol ;
element elasticBeamColumn 12    2   102  $A_C $E $G  $J_C  $Iy_C $Iz_C 1  -mass $mCol ;
element elasticBeamColumn 13    3   103  $A_C $E $G  $J_C  $Iy_C $Iz_C 1  -mass $mCol ;
element elasticBeamColumn 14    4   104  $A_C $E $G  $J_C  $Iy_C $Iz_C 1  -mass $mCol ;

# Second Feloor
element elasticBeamColumn 21  101   201  $A_C  $E  $G  $J_C  $Iy_C $Iz_C 1  -mass $mCol ;
element elasticBeamColumn 22  102   202  $A_C  $E  $G  $J_C  $Iy_C $Iz_C 1  -mass $mCol ;
element elasticBeamColumn 23  103   203  $A_C  $E  $G  $J_C  $Iy_C $Iz_C 1  -mass $mCol ;
element elasticBeamColumn 24  104   204  $A_C  $E  $G  $J_C  $Iy_C $Iz_C 1  -mass $mCol ;

# Beams
set mass [expr ($DL+(0.2*$LL))*($L/2.0)] ;
set mSt1 [expr $mass+($A_B*$RSteel)] ;
set mSt2 [expr $A_B*$RSteel] ;

# First Floor
#                         Tag iNode jNode Area  E   G   J    Iy    Iz  Trans  Mass
element elasticBeamColumn 101  101  102  $A_B  $E  $G  $J_B  $Iy_B $Iz_B 2  -mass $mSt2 ;
element elasticBeamColumn 102  102  103  $A_B  $E  $G  $J_B  $Iy_B $Iz_B 2  -mass $mSt1 ;
element elasticBeamColumn 103  103  104  $A_B  $E  $G  $J_B  $Iy_B $Iz_B 2  -mass $mSt2 ;
element elasticBeamColumn 104  104  101  $A_B  $E  $G  $J_B  $Iy_B $Iz_B 2  -mass $mSt1 ;

# Second Floor
element elasticBeamColumn 201  201  202  $A_B  $E  $G  $J_B  $Iy_B $Iz_B 2  -mass $mSt2 ;
element elasticBeamColumn 202  202  203  $A_B  $E  $G  $J_B  $Iy_B $Iz_B 2  -mass $mSt1 ;
element elasticBeamColumn 203  203  204  $A_B  $E  $G  $J_B  $Iy_B $Iz_B 2  -mass $mSt2 ;
element elasticBeamColumn 204  204  201  $A_B  $E  $G  $J_B  $Iy_B $Iz_B 2  -mass $mSt1 ;

puts "End of Define Model Geometry"


# Define Eigen Parameters
set a [eigen 3] ;
set W12 [lindex $a 0] ;
set W22 [lindex $a 1] ;
set W1 [expr pow($W12,0.5)] ;
set W2 [expr pow($W22,0.5)] ;
set T1 [expr 2.0*$Pi/$W1] ;
set T2 [expr 2.0*$Pi/$W2] ;

puts "*****************"
puts "W1=$W1 Rad/Sec"
puts "T1=$T1 Sec"
puts "*****************"
puts "W2=$W2 Rad/Sec"
puts "T2=$T2 Sec"



# Define Loads
set WzFloor [expr ($DL+$LL)*($L/2.0)*$g] ;

pattern Plain 1 Linear { ;
eleLoad -ele 102 104 202 204 -type -beamUniform 0.0 -$WzFloor 
} ;


recorder display 3D 500 10 500 500 -wipe
vup 0 0 1
vpn 1 1 0
vrp 4 4 2
prp 50 50 20
viewWindow -20 20 -10 10
display 1 1 10


# Static Analysis
wipeAnalysis ;	                                                            
constraints Transformation ;                                                                     
numberer RCM ;                                                                                      
system SparseGeneral ;                                                        
test EnergyIncr 1e-7 25 0 ;
algorithm ModifiedNewton ;	                                                         	
integrator LoadControl 0.01 ;                                                 
analysis Static ;  
analyze 100 ;                                                                 
loadConst -time 0.0 ; 

puts "End of Static Analysis"


# Define Recorders
recorder Node  -file $FileName/Base1.txt -node 1 -dof 1 reaction ;
recorder Node  -file $FileName/Base2.txt -node 2 -dof 1 reaction ;
recorder Node  -file $FileName/Base3.txt -node 3 -dof 1 reaction ;
recorder Node  -file $FileName/Base4.txt -node 4 -dof 1 reaction ;

recorder Node  -file $FileName/Disp201.txt -node 201 -dof 1 disp ;
recorder Node  -file $FileName/Disp202.txt -node 202 -dof 1 disp ;
recorder Node  -file $FileName/Disp203.txt -node 203 -dof 1 disp ;
recorder Node  -file $FileName/Disp204.txt -node 204 -dof 1 disp ;


# Define Acceleration
set dtInput 0.02 ;
set accelSeries "Series -dt $dtInput -filePath accel.txt -factor $g ";

#                         Tag  DOF          Accel
pattern UniformExcitation  2    1 -accel $accelSeries ;

# Time History Analysis
set NInput 2686 ;
set ndt 2 ;
set dtAnalysis [expr $dtInput/$ndt] ;
set NAnalysis [expr $NInput*$ndt] ;

wipeAnalysis ;	                                                           
constraints Transformation ;                                                      				
numberer RCM ;                                                               
system SparseGeneral ;                                                        
test EnergyIncr 1e-7 25 0 ;                                                                                                                                                   
algorithm ModifiedNewton ;	                                                				
integrator Newmark 0.5 0.25 ;                                                 
analysis Transient ;                                                          
analyze $NAnalysis $dtAnalysis ;                                              

puts "End of Time History Analysis"



