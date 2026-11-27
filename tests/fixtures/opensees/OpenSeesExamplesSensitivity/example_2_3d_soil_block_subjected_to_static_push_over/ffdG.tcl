# refer to "Quan Gu, Joel P. Conte, Ahmed Elgamal and Zhaohui Yang. Response Sensitivity Analysis of a Multi-Yield-Surface J2 Plasticity Model by
# Direct Differentiation Method.June, 2009. Vol 198(30-32), Computer Methods in Applied Mechanics and Engineering, Pages 2272-2285. 

set jj 1
foreach ii { 0.1  0.001 0.00001} {
  set perturb [expr $ii+1]
  set fileName "disp3a$jj.out"
 
  
  
  set jj [expr $jj+1]

  
  
wipe



model BasicBuilder -ndm 3 -ndf 3


#                                          massDen      G                       B                  cohension  peakshear
nDMaterial MultiYieldSurfaceClay        2 3   1.5     [expr 60000.0* $perturb]    240000.                30.0      0.1

#updateMaterialStage -material 2 -stage 101

#         create meshes
node        1      0.00000     0.0000    0.00000
node        2      0.00000     0.0000    0.5000
node        3      0.00000     0.0000    1.000000
node        4      0.00000     0.5000    0.00000
node        5      0.00000     0.5000    0.5000
node        6      0.00000     0.5000    1.000000
node        7      0.00000     1.00000    0.00000
node        8      0.00000     1.00000    0.5000
node        9      0.00000     1.00000   1.000000
node        10     0.5000     0.0000    0.00000
node        11     0.5000     0.0000    0.5000
node        12     0.5000     0.0000    1.000000
node        13     0.5000     0.5000    0.00000
node        14     0.5000     0.5000    0.5000
node        15     0.5000     0.5000    1.000000
node        16     0.5000     1.00000    0.00000
node        17     0.5000     1.00000    0.5000
node        18     0.5000     1.00000   1.000000
node        19    1.000000     0.0000    0.00000
node        20    1.000000     0.0000    0.5000
node        21    1.000000     0.0000    1.000000
node        22    1.000000     0.5000    0.00000
node        23    1.000000     0.5000    0.5000
node        24    1.000000     0.5000    1.000000
node        25    1.000000     1.00000    0.00000
node        26    1.000000     1.00000    0.50000
node        27    1.000000     1.00000   1.000000

#                                    ele    Node                                  mat
element bbarBrickWithSensitivity      1      1    10   13   4    2   11   14   5   2               
element bbarBrickWithSensitivity      2      2    11   14   5    3   12   15   6   2               
element bbarBrickWithSensitivity      3     10    19   22  13   11   20   23  14   2               
element bbarBrickWithSensitivity      4     11    20   23  14   12   21   24  15   2               
element bbarBrickWithSensitivity      5      4    13   16   7    5   14   17   8   2               
element bbarBrickWithSensitivity      6     13    22   25  16   14   23   26  17   2               
element bbarBrickWithSensitivity      7     14    23   26  17   15   24   27  18   2               
element bbarBrickWithSensitivity      8      5    14   17   8    6   15   18   9   2               

fix      1      1      1      1   0   0   0
fix      4      1      1      1   0   0   0
fix      7      1      1      1   0   0   0
fix     10      1      1      1   0   0   0
fix     13      1      1      1   0   0   0
fix     16      1      1      1   0   0   0
fix     19      1      1      1   0   0   0
fix     22      1      1      1   0   0   0
fix     25      1      1      1   0   0   0


# -------------- recorder ------------------------------------------------

recorder Node -file $fileName  -time -node 3 -dof 1 2 3 -precision 16  disp


pattern Plain 1 "Sine 0.0 1000.0 10.0 -factor 0.8 " { 
load  3    2.5    0.0   0   
load  6    2.5    0.0   0   
load  9    2.5    0.0   0   
}
 
pattern Plain 2 "Sine 0.0 1000.0 10.0  -shift 1.5708 -factor 0.8" { 
load  3     0.0    2.5   0    
load  12    0.0    2.5   0    
load  21    0.0    2.5   0    
} 



system BandGeneral
test NormDispIncr 1.E-12 50 2
constraints Transformation
integrator LoadControl 1 1 1 1
algorithm Newton 
numberer RCM

 


analysis Static



set startT [clock seconds]
analyze 50
set endT [clock seconds]
puts "Execution time: [expr $endT-$startT] seconds."




}




