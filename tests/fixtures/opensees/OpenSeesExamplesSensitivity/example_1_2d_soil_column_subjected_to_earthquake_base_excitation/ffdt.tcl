# refer to "Quan Gu, Joel P. Conte, Ahmed Elgamal and Zhaohui Yang. Response Sensitivity Analysis of a Multi-Yield-Surface J2 Plasticity Model by
# Direct Differentiation Method.June, 2009. Vol 198(30-32), Computer Methods in Applied Mechanics and Engineering, Pages 2272-2285. "

set jj 1
foreach ii {0.5 0.1 0.05 0.01 } {
  set perturb [expr $ii+1]
  set fileName "disp29t$jj.out"
 
  
  
  set jj [expr $jj+1]

  
  
wipe

set h G
set g  -9.8
model basic -ndm 2 -ndf 2








#        29---------------58
#         |              |
#         |              |
#         |              |
#         |      1(#1)   |
#         |              |
#         |              |
#        28---------------57
#         |              |
#         |              |          element    material#   
#         |              |          1-2           1
#         |      2(#1)   |          3-6           2
#         |              |          7-8           3
#          ---------------          9-14          4
#         |              |          15-22         5    
#         |              |          23-28         6  
#         |              |
#         |              |
#         |              |
#          ---------------
#         |              |
#         |              |
#         |              |
#         |              |
#         |              |
#          ---------------
#         |              |
#        .|              |
#        .|              |
#         |              |
#         |              |
#        3 ---------------32
#         |              |
#         |              |
#         |              |
#         |     27(#6)   |
#         |              |
#         |              |
#        2 ---------------31
#         |              |
#         |              |
#         |              |
#         |     28(#6)   |
#        1|              |30
#  -------------------------------





#-----------------------------------------------------
# Nodes Definition
#-----------------------------------------------------

node    1          272.9     -43.426  -mass             0             0
node    2          272.9     -38.728  -mass             0             0
node    3          272.9     -34.031  -mass             0             0
node    4          272.9     -29.334  -mass             0             0
node    5          272.9     -24.636  -mass             0             0
node    6          272.9     -19.939  -mass             0             0
node    7          272.9     -15.242  -mass             0             0
node    8          272.9     -11.813  -mass             0             0
node    9          272.9     -8.3835  -mass             0             0
node    10         272.9     -4.9543  -mass             0             0
node    11         272.9     -1.5252  -mass             0             0
node    12         272.9       1.904  -mass             0             0
node    13         272.9      5.3331  -mass             0             0
node    14         272.9      8.7623  -mass             0             0
node    15         272.9      12.191  -mass             0             0
node    16         272.9      14.096  -mass             0             0
node    17         272.9      16.001  -mass             0             0
node    18         272.9      17.905  -mass             0             0
node    19         272.9       19.81  -mass             0             0
node    20         272.9      20.877  -mass             0             0
node    21         272.9      21.945  -mass             0             0
node    22         272.9      22.402  -mass             0             0
node    23         272.9      22.859  -mass             0             0
node    24         272.9      24.459  -mass             0             0
node    25         272.9      26.059  -mass             0             0
node    26         272.9      27.659  -mass             0             0
node    27         272.9      29.259  -mass             0             0
node    28         272.9      29.602  -mass             0             0
node    29         272.9      29.945  -mass             0             0
node    30        267.27     -43.426  -mass             0             0
node    31        267.27     -38.728  -mass             0             0
node    32        267.27     -34.031  -mass             0             0
node    33        267.27     -29.334  -mass             0             0
node    34        267.27     -24.636  -mass             0             0
node    35        267.27     -19.939  -mass             0             0
node    36        267.27     -15.242  -mass             0             0
node    37        267.27     -11.813  -mass             0             0
node    38        267.27     -8.3835  -mass             0             0
node    39        267.27     -4.9543  -mass             0             0
node    40        267.27     -1.5252  -mass             0             0
node    41        267.27       1.904  -mass             0             0
node    42        267.27      5.3331  -mass             0             0
node    43        267.27      8.7623  -mass             0             0
node    44        267.27      12.191  -mass             0             0
node    45        267.27      14.096  -mass             0             0
node    46        267.27      16.001  -mass             0             0
node    47        267.27      17.905  -mass             0             0
node    48        267.27       19.81  -mass             0             0
node    49        267.27      20.877  -mass             0             0
node    50        267.27      21.945  -mass             0             0
node    51        267.27      22.402  -mass             0             0
node    52        267.27      22.859  -mass             0             0
node    53        267.27      24.432  -mass             0             0
node    54        267.27      26.012  -mass             0             0
node    55        267.27      27.593  -mass             0             0
node    56        267.27      29.166  -mass             0             0
node    57        267.27      29.497  -mass             0             0
node    58        267.27      29.828  -mass             0             0


   


#                                          massDen    G       B       cohension  PeakShearStrain
nDMaterial MultiYieldSurfaceClay      1 2  0.0      54450	1.6e5     33.           .1          
nDMaterial MultiYieldSurfaceClay      2 2  0.0      33800	1.0e5     26.           .1          
nDMaterial MultiYieldSurfaceClay      3 2  0.0      96800	2.9e5     44.           .1          
nDMaterial MultiYieldSurfaceClay      4 2  0.0      61250	 1.8e5    [expr 35. * $perturb]          .1          
nDMaterial MultiYieldSurfaceClay      5 2  0.0      180000	5.4e5     60.           .1          
nDMaterial MultiYieldSurfaceClay      6 2  0.0      369800	1.1e6     86.           .1          





#-----------------------------------------------------
# Soil Elememets Definition
#-----------------------------------------------------
#                                                   thick    material   Mat  Press  mDensity   gravity

element quadWithSensitivity   1    57  28  29  58 6.0957 "PlaneStrain" 1    0       2.0       0 $g
element quadWithSensitivity   2    56  27  28  57 6.0957 "PlaneStrain" 1    0       2.0       0 $g
element quadWithSensitivity   3    55  26  27  56 6.0957 "PlaneStrain" 2    0       2.0       0 $g
element quadWithSensitivity   4    54  25  26  55 6.0957 "PlaneStrain" 2    0       2.0       0 $g
element quadWithSensitivity   5    53  24  25  54 6.0957 "PlaneStrain" 2    0       2.0       0 $g
element quadWithSensitivity   6    52  23  24  53 6.0957 "PlaneStrain" 2    0       2.0       0 $g
element quadWithSensitivity   7    51  22  23  52 6.0957 "PlaneStrain" 3    0       2.0       0 $g
element quadWithSensitivity   8    50  21  22  51 6.0957 "PlaneStrain" 3    0       2.0       0 $g
element quadWithSensitivity   14   15  16  45  44 6.0957 "PlaneStrain" 4    0       2.0       0 $g
element quadWithSensitivity   13   16  17  46  45 6.0957 "PlaneStrain" 4    0       2.0       0 $g
element quadWithSensitivity   12   17  18  47  46 6.0957 "PlaneStrain" 4    0       2.0       0 $g
element quadWithSensitivity   11   18  19  48  47 6.0957 "PlaneStrain" 4    0       2.0       0 $g
element quadWithSensitivity   10   19  20  49  48 6.0957 "PlaneStrain" 4    0       2.0       0 $g
element quadWithSensitivity   9    20  21  50  49 6.0957 "PlaneStrain" 4    0       2.0       0 $g
element quadWithSensitivity   15   44  43  14  15 6.0957 "PlaneStrain" 5    0       2.0       0 $g
element quadWithSensitivity   16   43  42  13  14 6.0957 "PlaneStrain" 5    0       2.0       0 $g
element quadWithSensitivity   17   42  41  12  13 6.0957 "PlaneStrain" 5    0       2.0       0 $g
element quadWithSensitivity   18   41  40  11  12 6.0957 "PlaneStrain" 5    0       2.0       0 $g
element quadWithSensitivity   19   40  39  10  11 6.0957 "PlaneStrain" 5    0       2.0       0 $g
element quadWithSensitivity   20   39  38   9  10 6.0957 "PlaneStrain" 5    0       2.0       0 $g
element quadWithSensitivity   21   38  37   8   9 6.0957 "PlaneStrain" 5    0       2.0       0 $g
element quadWithSensitivity   22   37  36   7   8 6.0957 "PlaneStrain" 5    0       2.0       0 $g
element quadWithSensitivity   23   36  35   6   7 6.0957 "PlaneStrain" 6    0       2.0       0 $g
element quadWithSensitivity   24   35  34   5   6 6.0957 "PlaneStrain" 6    0       2.0       0 $g
element quadWithSensitivity   25   34  33   4   5 6.0957 "PlaneStrain" 6    0       2.0       0 $g
element quadWithSensitivity   26   33  32   3   4 6.0957 "PlaneStrain" 6    0       2.0       0 $g
element quadWithSensitivity   27   32  31   2   3 6.0957 "PlaneStrain" 6    0       2.0       0 $g
element quadWithSensitivity   28   31  30   1   2 6.0957 "PlaneStrain" 6    0       2.0       0 $g



#updateMaterialStage -material 1 -stage 101
#updateMaterialStage -material 2 -stage 101
#updateMaterialStage -material 3 -stage 101
#updateMaterialStage -material 4 -stage 101
#updateMaterialStage -material 5 -stage 101
#updateMaterialStage -material 6 -stage 101

 

fix    1   1    1
fix   30   1    1




equalDOF 29 58 1 2
equalDOF 28 57 1 2
equalDOF 27 56 1 2
equalDOF 26 55 1 2
equalDOF 25 54 1 2
equalDOF 24 53 1 2
equalDOF 23 52 1 2
equalDOF 22 51 1 2
equalDOF 21 50 1 2
equalDOF 20 49 1 2
equalDOF 19 48 1 2
equalDOF 18 47 1 2
equalDOF 17 46 1 2
equalDOF 16 45 1 2
equalDOF 15 44 1 2
equalDOF 14 43 1 2
equalDOF 13 42 1 2
equalDOF 12 41 1 2
equalDOF 11 40 1 2
equalDOF 10 39 1 2
equalDOF  9 38 1 2
equalDOF  8 37 1 2
equalDOF  7 36 1 2
equalDOF  6 35 1 2
equalDOF  5 34 1 2
equalDOF  4 33 1 2
equalDOF  3 32 1 2
equalDOF  2 31 1 2




# ------------------------ recorder ----------------------------------

remove recorders 
recorder Node -file $fileName  -time -node 29  -dof 1 -precision 16  disp
 
#--------------------------------------------------------
# Soil Gravity Analysis Model Definition
#--------------------------------------------------------
system BandGeneral
test NormDispIncr 1.E-12 50 2
constraints Transformation
integrator LoadControl 1 1 1 1
algorithm Newton
numberer RCM


analysis Static

analyze 1

puts "soil gravity nonlinear analysis completed ..."
 
 
 
wipeAnalysis



constraints Transformation
test NormDispIncr 1.E-12 50  2
algorithm Newton
numberer RCM
system BandGeneral

integrator Newmark  0.55 0.2756   
analysis Transient

pattern UniformExcitation    1     1    -accel "Series -factor 1 -filePath acce.txt -dt 0.01"


set startT [clock seconds]
analyze 1000 0.01
set endT [clock seconds]
puts "Execution time: [expr $endT-$startT] seconds."

}