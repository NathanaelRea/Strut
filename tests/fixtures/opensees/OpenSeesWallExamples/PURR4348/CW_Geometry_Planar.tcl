# Geometry of Wall Section Defined

# Create nodes for first story

# First story wall node and element numbering
#  101    o   -                                                                             
#         |   ^
#         |
#  1199   o   Elastic elements (two elastic elements above NL elements)
#         |
#         |   v
#  .      o   -
#  .      o   ^
#  .      o   
#  1103   o
#         |
#  1102   o  Nonlinear elements (as many as specified)
#         |
#  1101   o
#         |   v
#  1      o   -

#Pdelta column

#  103    o   
#  1399   o  zero length rotational spring 
#         |
#         | 
#         | 
#         |  
#         |   
#         |
#         |
#         | 
#         |
#         |
#         |   
#   3     o   

set nodeload [expr {$StoryW/$g/4.0}];

set TotalD [expr {$w_D+$b_Length}]
set HingeL [expr {$b_D}]
set RigidL [expr {$w_D/2.0}]

node 1 0.0 0.0
node 2 $TotalD 0.0
node 3 [expr {2*$TotalD}] 0.0;  #node of the P-delta column     

fix 1 1 1 1
fix 2 1 1 1
fix 3 1 1 0; #P-delta column is pinned

# Create nodes for first story walls (nonlinear and elastic elements0)
for {set i 1} {$i <= $w_nle} {incr i 1} {
 set nl_height [expr {(1.0*$i)*$w_HL/$w_nle}]
if {$i >= 10} {
 node 11$i 0.0 $nl_height
 node 12$i $TotalD $nl_height
} else {
 node 110$i 0.0 $nl_height
 node 120$i $TotalD $nl_height
}
}

# Split remaining first story height into two elements
set newheight [expr {$nl_height+(($h1-$nl_height)/2.0)}]
node 1199 0.0 $newheight
node 1299 $TotalD $newheight

#Pdelta nodes for first story
node 1399 [expr {2*$TotalD}] $h1

##### Schematic for node and element numbering #####

# This layout is implemented for each floor in the for loop below. X is the story number.

# KEY:
#     - H: Hinge, concentrated plasticity zero length element
#     - RL: Rigid link, all DOF linked
#     - E: Beam Elastic Element
#     - WE: Wall Elastic Element

#  nodeStart nodeR1  nodeH1      nodeM1       nodeM2     node H2  nodeR2 nodeEnd  nodeH3  Truss Element   nodePDelta
#   X001       X011  X021        X031          X032         X022 X012 X002 X070 X061        X060            X003
#     o------------oo--------------o-------------o--------------oo------------oo------------------------------o
#  WE |     RL     H       E               E             E       H    RL      |  WE                colnodeT3  o zero length rotational spring X50
#  X5 |    X10    X20     X30             X32           X31     X21   X11     |  X6                  X399     |
#     |                                                                       |                               |
#     o  colnodeT1                                                 colnodeT2  o                               |	 
#  WE |     X199                                                      X299    |  WE                           | 
#  X3 |                                                                       |  X4                           |  Pdelta Column
#     |                                                                       |                               |     X7
#     o  colnode1                                                   colnode2  o                               | 
#  WE |     X101                                                      X201    |  WE                           |
#  X1 |                                                                       |  X2                colnode3   | 
#     |                                                                       |                      X301     o zero length rotational spring X40
#     o                                                                       o                               o
#   nodebelow_l                                                          nodebelow_r                  nodebelow_PDelta 
#    X01-100                                                               X02-100                         X03-100
 
for {set i 1} {$i <= $storyn} {incr i 1} {
 
 # redefine variables for next story
 set nodeStart [expr {$i*1000+1}]
 set nodeEnd [expr {$nodeStart+1}]
 set nodePDelta [expr {$nodeEnd+1}]
 set nodeR1 [expr {$i*1000+10+1}]
 set nodeR2 [expr {$nodeR1+1}]
 set nodeH1 [expr {$i*1000+20+1}]
 set nodeH2 [expr {$nodeH1+1}]
 set nodeM1 [expr {$i*1000+30+1}]
 set nodeM2 [expr {$nodeM1+1}]
 set nodeH3 [expr {$i*1000+61}]
 
 set Heightnow [expr {$h1+($i-1)*$htyp}]
 
 if {$i!=1} {
  set colnode1  [expr {$i*10000+101}]
  set colnodeT1 [expr {$i*10000+199}]
  set colnode2  [expr {$i*10000+201}]
  set colnodeT2 [expr {$i*10000+299}]
  set colnode3  [expr {$i*10000+301}]
  set colnodeT3 [expr {$i*10000+399}]
  node $colnode3  [expr {2*$TotalD}] [expr {$Heightnow-$htyp}]
  node $colnodeT3 [expr {2*$TotalD}] [expr {$Heightnow}]
  set nodebelow_l [expr {$nodeStart-1000}]
  set nodebelow_r [expr {$nodeEnd-1000}]
  set nodebelow_PDelta [expr {$nodePDelta-1000}]
 } else {
  set nodebelow_l 1199
  set nodebelow_r 1299
 }
 
 # create new nodes
 node $nodeStart  0.0                                                  $Heightnow    -mass $nodeload 0.00001 0.00001    
 node $nodeEnd    $TotalD                                              $Heightnow    -mass $nodeload 0.00001 0.00001    
 node $nodePDelta [expr {2*$TotalD}]                                   $Heightnow    
 node $nodeR1     $RigidL                                              $Heightnow
 node $nodeR2     [expr {$TotalD-$RigidL}]                             $Heightnow
 node $nodeH1     $RigidL                                              $Heightnow
 node $nodeH2     [expr {$TotalD-$RigidL}]                             $Heightnow

# New element numbering
 set col11 [expr {$i*10+1}]
 set col21 [expr {$i*10+2}]
 set col12 [expr {$i*10+3}]
 set col22 [expr {$i*10+4}]
 set col13 [expr {$i*10+5}]
 set col23 [expr {$i*10+6}]
 set rl1 [expr {$i*1000+10}]
 set rl2 [expr {$rl1+1}]
 set hg1 [expr {$i*1000+20}]
 set hg2 [expr {$hg1+1}]
 set mde1 [expr {$i*1000+30}]
 set mde2 [expr {$mde1+1}]
 set mde3 [expr {$mde1+2}]

#####P-Delta Columns########

# Truss links 
set linkID [expr {$i*100+60}];
element truss $linkID $nodeEnd $nodePDelta $Arigid $TrussMatID;	
	
# P-delta columns
if {$i!=1} {
 set colID [expr {$i*10+7}]
 element elasticBeamColumn  $colID $colnodeT3 $colnode3 $Arigid [expr {1000000.*$w_E}] $Irigid $PDeltaTransfTag;	
} else {
 element elasticBeamColumn  17 3 1399 $Arigid [expr {1000000.*$w_E}] $Irigid $PDeltaTransfTag;	
}

# Define P-delta column spring: zero-stiffness elastic spring	
if {$i!=1} {
 RotPdeltaCol [expr {$i*100+40}] $nodebelow_PDelta $colnode3 ;
 RotPdeltaCol [expr {$i*100+50}] $colnodeT3 $nodePDelta ;
} else { 	
 RotPdeltaCol 150 1399 1003 ;
}
 
##################################
 
 ### Wall elements ###
 
 # for typical stories add three elastic elements along the height of each wall (allow for something other than perfectly linear curvature)
 # for the first story add the number of hinge elements specified then two elastic elements above this region
 
 if {$i!=1} {
  element elasticBeamColumn $col11 $nodebelow_l $nodeStart $w_Ag $w_E $w_Iz $geomTransfTag
  element elasticBeamColumn $col21 $nodebelow_r $nodeEnd $w_Ag $w_E $w_Iz $geomTransfTag
 } else {
    for {set j 1} {$j <= $w_nle} {incr j 1} {
   if {$j==1} {
    element dispBeamColumn 1101 1 1101 $w_SecPoi $w_SecTag $geomTransfTag;
    element dispBeamColumn 1201 2 1201 $w_SecPoi $w_SecTag $geomTransfTag
   } else {
    element dispBeamColumn [expr {1100+$j}] [expr {1100+$j-1}] [expr {1100+$j}] $w_SecPoi $w_SecTag $geomTransfTag;
    element dispBeamColumn [expr {1200+$j}] [expr {1200+$j-1}] [expr {1200+$j}] $w_SecPoi $w_SecTag $geomTransfTag 
  }
  }
  
  element elasticBeamColumn $col11 [expr {round(1100+$w_nle)}] $nodebelow_l $w_Ag $w_E $w_Iz $geomTransfTag
  element elasticBeamColumn $col21 [expr {round(1200+$w_nle)}] $nodebelow_r $w_Ag $w_E $w_Iz $geomTransfTag
  element elasticBeamColumn $col12 $nodebelow_l $nodeStart $w_Ag $w_E $w_Iz $geomTransfTag
  element elasticBeamColumn $col22 $nodebelow_r $nodeEnd $w_Ag $w_E $w_Iz $geomTransfTag
 }

 ### Beam elements ###
 
 # Using elastic elements with high stiffness
 element elasticBeamColumn $rl1 $nodeStart $nodeR1 [expr {1000.*$w_Ag}] [expr {10000.*$w_E}] [expr {1000.*$w_Iz}] $geomTransfTag
 element elasticBeamColumn $rl2 $nodeEnd $nodeR2 [expr {1000.*$w_Ag}] [expr {10000.*$w_E}] [expr {1000.*$w_Iz}] $geomTransfTag
 
 # Concentrated Plasticity Element
 #rotSpring2DModIMKPinching $hg1 $nodeR1 $nodeH1 $K0 $as_Pos $as_Neg $My_Plus $My_Neg $FprPos $FprNeg $A_pinch $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
 #rotSpring2DModIMKPinching $hg2 $nodeR2 $nodeH2 $K0 $as_Pos $as_Neg $My_Plus $My_Neg $FprPos $FprNeg $A_pinch $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg

 source RotSpring2DModIKModel.tcl
 #
 rotSpring2DModIKModel $hg1 $nodeR1 $nodeH1 $K0 $as_Pos $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_K $Lamda_A $Lamda_C $c_S $c_K $c_A $c_C $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
 rotSpring2DModIKModel $hg2 $nodeR2 $nodeH2 $K0 $as_Pos $as_Neg $My_Plus $My_Neg $Lamda_S $Lamda_K $Lamda_A $Lamda_C $c_S $c_K $c_A $c_C $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
 
 # Middle element
 element elasticBeamColumn $mde3 $nodeH1 $nodeH2 $b_Ag $b_E $b_Iz $geomTransfTag

} 

recorder Node -file $dataDirNow/Dispcompare.txt -time -node 6001 6011 6021 6031 6032 6022 6012 6002 6070 6061 6060 6003 -dof 1 2 disp

# Define beam hinges region and moment-rotation recorder
if {$storyn==8} {
  region 1 -ele 1020 1021 2020 2021 3020 3021 4020 4021 5020 5021 6020 6021 7020 7021 8020 8021
  region 2 -ele 150 240 250 340 350 440 450 540 550 640 650 740 750 840 850
  region 3 -node 6001 6011 6021 6031 6032 6022 6012 6002
  region 4 -node 6001 6011 6021 6031 6032 6022 6012 6002
  region 5 -node 1 1001 2001 3001 4001 5001 6001 7001 8001
  region 6 -node 2 1002 2002 3002 4002 5002 6002 7002 8002
  region 7 -node 3 1003 2003 3003 4003 5003 6003 7003 8003
  region 8 -node 8001 8002 8003
  region 9 -node 1011 1012 2011 2012 3011 3012 4011 4012 5011 5012 6011 6012 7011 7012 8011 8012
  recorder Drift -file $dataDirNow/DriftWall1.txt -time -iNode 1 1001 2001 3001 4001 5001 6001 7001 -jNode 1001 2001 3001 4001 5001 6001 7001 8001 -dof 1 -perpDirn 2
  recorder Drift -file $dataDirNow/DriftWall2.txt -time -iNode 2 1002 2002 3002 4002 5002 6002 7002 -jNode 1002 2002 3002 4002 5002 6002 7002 8002 -dof 1 -perpDirn 2
  recorder Drift -file $dataDirNow/CB_Chord_Rot.txt -time -iNode 1011 2011 3011 4011 5011 6011 7011 8011 -jNode 1012 2012 3012 4012 5012 6012 7012 8012 -dof 2 -perpDirn 1
  
 } else {
  region 1 -ele 1020 1021 2020 2021 3020 3021 4020 4021 5020 5021 6020 6021 7020 7021 8020 8021 9020 9021 10020 10021 11020 11021 12020 12021
  region 2 -ele 150 240 250 340 350 440 450 540 550 640 650 740 750 840 850 940 950 1040 1050 1140 1150 1240 1250
  region 3 -node 6001 6011 6021 6031 6032 6022 6012 6002
  region 4 -node 6001 6011 6021 6031 6032 6022 6012 6002
  region 5 -node 1 1001 2001 3001 4001 5001 6001 7001 8001 9001 10001 11001 12001
  region 6 -node 2 1002 2002 3002 4002 5002 6002 7002 8002 9002 10002 11002 12002
  region 7 -node 3 1003 2003 3003 4003 5003 6003 7003 8003 9003 10003 11003 12003
  region 8 -node 12001 12002 12003
  region 9 -node 1011 1012 2011 2012 3011 3012 4011 4012 5011 5012 6011 6012 7011 7012 8011 8012 9011 9012 10011 10012 11011 11012 12011 12012
  recorder Drift -file $dataDirNow/DriftWall1.txt -time -iNode 1 1001 2001 3001 4001 5001 6001 7001 8001 9001 10001 11001 -jNode 1001 2001 3001 4001 5001 6001 7001 8001 9001 10001 11001 12001 -dof 1 -perpDirn 2
  recorder Drift -file $dataDirNow/DriftWall2.txt -time -iNode 2 1002 2002 3002 4002 5002 6002 7002 8002 9002 10002 11002 -jNode 1002 2002 3002 4002 5002 6002 7002 8002 9002 10002 11002 12002 -dof 1 -perpDirn 2
  recorder Drift -file $dataDirNow/CB_Chord_Rot.txt -time -iNode 1011 2011 3011 4011 5011 6011 7011 8011 9011 10011 11011 12011 -jNode 1012 2012 3012 4012 5012 6012 7012 8012 9012 10012 11012 12012 -dof 2 -perpDirn 1
  }

region 10 -node 1101 1102 1103 1104
region 11 -node 1201 1202 1203 1204

recorder Element -file $dataDirNow/Beam-Mom-Hist.txt -time -region 1 force;
recorder Element -file $dataDirNow/Beam-Rot-Hist.txt -time -region 1 deformation;

# Define PDelta hinge region and moment-rotation recorder

recorder Element -file $dataDirNow/PDelCol-Mom-Hist.txt -time -region 2 force;
recorder Element -file $dataDirNow/PDelCol-Rot-Hist.txt -time -region 2 deformation;

recorder Node -file $dataDirNow/Load1.txt -time -node 1 -dof 1 2 3 reaction
recorder Node -file $dataDirNow/Load2.txt -time -node 2 -dof 1 2 3 reaction
recorder Node -file $dataDirNow/Load3.txt -time -node 3 -dof 1 2 3 reaction
recorder Node -file $dataDirNow/PDeltaColBaseRotation.txt -time -node 3 -dof 3 disp

recorder Node -file $dataDirNow/Story6NodeDisp1.txt -time -region 3 -dof 1 disp
recorder Node -file $dataDirNow/Story6NodeDisp2.txt -time -region 4 -dof 2 disp

recorder Node -file $dataDirNow/DispWall1.txt -time -region 5 -dof 1 2 3 disp
recorder Node -file $dataDirNow/DispWall2.txt -time -region 6 -dof 1 2 3 disp
recorder Node -file $dataDirNow/ForceWall1.txt -time -region 5 -dof 1 2 3 reaction
recorder Node -file $dataDirNow/ForceWall2.txt -time -region 6 -dof 1 2 3 reaction
recorder Node -file $dataDirNow/DispLCol.txt -time -region 7 -dof 1 disp
recorder Node -file $dataDirNow/RoofDisp.txt -time -region 8 -dof 1 disp

recorder Node -file $dataDirNow/Wall1NL_Disp.txt -time -region 10 -dof 1 2 3 disp
recorder Node -file $dataDirNow/Wall1Nl_Force.txt -time -region 10 -dof 1 2 3 reaction

recorder Node -file $dataDirNow/Wall2NL_Disp.txt -time -region 11 -dof 1 2 3 disp
recorder Node -file $dataDirNow/Wall2Nl_Force.txt -time -region 11 -dof 1 2 3 reaction

recorder Node -file $dataDirNow/CB_End_Disp.txt -time -region 9 -dof 1 2 3 disp

set y [expr $w_D/2.-$w_t1];
set z [expr 0.0];

recorder Element -file $dataDirNow/SSWall_11.txt -time -ele 1101 section 1 fiber $y $z $MatTag_W_St_Flange stressStrain
recorder Element -file $dataDirNow/SSWall_21.txt -time -ele 1201 section 1 fiber $y $z $MatTag_W_St_Flange stressStrain
recorder Element -file $dataDirNow/SSWall_12.txt -time -ele 1101 section 1 fiber -$y $z $MatTag_W_St_Flange stressStrain
recorder Element -file $dataDirNow/SSWall_22.txt -time -ele 1201 section 1 fiber -$y $z $MatTag_W_St_Flange stressStrain

recorder Element -file $dataDirNow/Defo_ele1101.txt -time -ele 1101 section 2 deformation
recorder Element -file $dataDirNow/Defo_ele1201.txt -time -ele 1201 section 2 deformation
recorder Element -file $dataDirNow/Defo_ele1102.txt -time -ele 1102 section 2 deformation
recorder Element -file $dataDirNow/Defo_ele1202.txt -time -ele 1202 section 2 deformation
recorder Element -file $dataDirNow/Defo_ele1103.txt -time -ele 1103 section 2 deformation
recorder Element -file $dataDirNow/Defo_ele1203.txt -time -ele 1203 section 2 deformation
recorder Element -file $dataDirNow/Defo_ele1104.txt -time -ele 1104 section 2 deformation
recorder Element -file $dataDirNow/Defo_ele1204.txt -time -ele 1204 section 2 deformation

recorder Element -file $dataDirNow/Force_ele1101.txt -time -ele 1101 section 2 forces
recorder Element -file $dataDirNow/Force_ele1201.txt -time -ele 1201 section 2 forces
recorder Element -file $dataDirNow/Force_ele1102.txt -time -ele 1102 section 2 forces
recorder Element -file $dataDirNow/Force_ele1202.txt -time -ele 1202 section 2 forces
recorder Element -file $dataDirNow/Force_ele1103.txt -time -ele 1103 section 2 forces
recorder Element -file $dataDirNow/Force_ele1203.txt -time -ele 1203 section 2 forces
recorder Element -file $dataDirNow/Force_ele1104.txt -time -ele 1104 section 2 forces
recorder Element -file $dataDirNow/Force_ele1204.txt -time -ele 1204 section 2 forces