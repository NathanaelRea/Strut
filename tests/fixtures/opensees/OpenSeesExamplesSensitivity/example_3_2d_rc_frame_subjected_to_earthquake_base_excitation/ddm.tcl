

#   unit. meter, K-N, K-Pa,K-Kg ver 2.0
# -- refer to: Q.Gu. FINITE ELEMENT RESPONSE SENSITIVITY AND RELIABILITY ANALYSIS OF SOIL-FOUNDATION-STRUCTURE-INTERACTION (SFSI) SYSTEMS. 
#                    PhD Dissertation, Structural Engineeeing Dept. UCSD, 2008.  
#
#
#
#        p              9'           10'     
#      .-------> 3============6 =============9                  =| 7.2
#       .        ||           ||            ||                   |
#        .       || 2'        || 4'         || 6'                |
#         .      ||     7'    ||     8'     ||                   |
#          .-->  2============5 =============8                  =| 3.6
#           .    ||           ||            ||                   |
#            .   || 1'        || 3'         || 5'                |
#             .  ||           ||            ||                   |
#              . ||           ||            ||                   |
# 92_____________1|___________4|____________7|_____________110  -|  0.0
 
#    -7.2                                         19.2  23.2          
#   |__|__|__|__|_|_|____|___|_|_|____|___|_|_|__|__|__|__|
#       -5.2  -1.2|1.2      5.8 8.2     12.8 15.2                       
#  -9.2    -3.2   0     3.5   7.0    10.5  14.0 17.2  21.2         
#  
#

# unit KN, KPa, KKg.
# CREATE THE MODEL BUILDERS FOR UPGROUND FRAMES
 

model BasicBuilder -ndm 2 -ndf 3
reliability
# ------------------- DEFINE beam NODES  ---------------------------------

set framemass1 15.0
set framemass2 30.0
set framemass3  4.0

#    tag  X     Y     
node  1   0     0    -mass $framemass1 $framemass1 0.0
node  2   0    3.6   -mass $framemass1 $framemass1 0.0
node  3   0    7.2   -mass $framemass1 $framemass1 0.0
node  4  7.0   0.0   -mass $framemass2 $framemass2 0.0
node  5  7.0   3.6   -mass $framemass2 $framemass2 0.0
node  6  7.0   7.2   -mass $framemass2 $framemass2 0.0
node  7 14.0   0.0   -mass $framemass1 $framemass1 0.0
node  8 14.0   3.6   -mass $framemass1 $framemass1 0.0
node  9 14.0   7.2   -mass $framemass1 $framemass1 0.0



fix 1 1 1 1
fix 4 1 1 1
fix 7 1 1 1
 
# ------------------- weight load -------------------------------
set framemass1 15.0
set framemass2 30.0

set upperload1 [expr -$framemass1*10.0]
set upperload2 [expr -$framemass2*10.0]





pattern Plain 1 "Constant"  {
load 1 0.0 $upperload1 0
load 2 0.0 $upperload1 0
load 3 0.0 $upperload1 0
load 4 0.0 $upperload2 0
load 5 0.0 $upperload2 0
load 6 0.0 $upperload2 0
load 7 0.0 $upperload1 0
load 8 0.0 $upperload1 0
load 9 0.0 $upperload1 0

}

 



# --------------------------- DEFINE MATERIALS ---------------------------
# --------------------------- UPPERGROUND FRAMES -------------------------

# Cover concrete            tag      -f'c      -epsco      -f'cu     -epscu
#uniaxialMaterial Concrete01  1     -27588.5    -0.002         0.0    -0.008
#                                                  fc              fu             Ec           eps0     epsu     eta
uniaxialMaterial    SmoothPSConcrete     1       27579.04        1000.0         2.4910e7      0.002    0.012    0.2


# Core concrete             tag      -f'c      -epsco      -f'cu     -epscu
#uniaxialMaterial Concrete01  2     -34485.6    -0.004      -20691.4  -0.014
#                                                  fc              fu           Ec         eps0     epsu     eta
uniaxialMaterial    SmoothPSConcrete     2      34473.8        25723.0       2.7851e7      0.005    0.02     0.2

#  b=Hkin/(E+Hkin)=0.008
# Steel model              tag      E        fy          Hiso      Hkin
#uniaxialMaterial Hardening   3     2.0e8   248200.       0.0     1.6129e6 

uniaxialMaterial SteelMP  3  248200.  2.1e8    0.02

# -----------------------------------------------------------------------------


# --------------------------- DEFINE SECTIONS ----------------------------
# --------------------------- UPPERGROUND FRAMES -------------------------
# Interior column section

section fiberSec 1 {
   #           mat nfIJ nfJK   yI     zI      yJ      zJ      yK      zK       yL     zL
   patch quadr  2    1   12 -0.2500  0.2000 -0.2500 -0.2000  0.2500 -0.2000  0.2500  0.2000
   patch quadr  1    1   14 -0.3000 -0.2000 -0.3000 -0.2500  0.3000 -0.2500  0.3000 -0.2000
   patch quadr  1    1   14 -0.3000  0.2500 -0.3000  0.2000  0.3000  0.2000  0.3000  0.2500
   patch quadr  1    1    2 -0.3000  0.2000 -0.3000 -0.2000 -0.2500 -0.2000 -0.2500  0.2000
   patch quadr  1    1    2  0.2500  0.2000  0.2500 -0.2000  0.3000 -0.2000  0.3000  0.2000

   #              mat nBars   area     yI         zI        yF       zF
   layer straight  3    3   0.000645 -0.2000   0.2000    -0.2000   -0.2000
   layer straight  3    3   0.000645  0.2000   0.2000     0.2000   -0.2000
}
#   steel:  Ag/As = 6*0.000645/(0.6*0.5) =1.29%




# Exterior column section
section fiberSec 2 {
   #          mat nfIJ nfJK   yI        zI      yJ      zJ      yK      zK      yL     zL
   patch quadr 2   1   10    -0.2000  0.2000 -0.2000 -0.2000  0.2000 -0.2000  0.2000  0.2000
   patch quadr 1   1   12    -0.2500 -0.2000 -0.2500 -0.2500  0.2500 -0.2500  0.2500 -0.2000
   patch quadr 1   1   12    -0.2500  0.2500 -0.2500  0.2000  0.2500  0.2000  0.2500  0.2500
   patch quadr 1   1    2    -0.2500  0.2000 -0.2500 -0.2000 -0.2000 -0.2000 -0.2000  0.2000
   patch quadr 1   1    2     0.2000  0.2000  0.2000 -0.2000  0.2500 -0.2000  0.2500  0.2000

   #              mat nBars area       yI     zI      yF      zF
   layer straight 3    3   0.00051  -0.2000 0.2000 -0.2000 -0.2000
   layer straight 3    3   0.00051   0.2000 0.2000  0.2000 -0.2000
}
#   steel:  Ag/As = 6*0.000510/(0.5*0.5) =1.22%

# Girder section
section fiberSec 3 {
   #          mat nfIJ nfJK   yI          zI         yJ       zJ         yK        zK        yL        zL
   patch quadr 1  1     12   -0.2500     0.2000     -0.2500  -0.2000    0.2500   -0.2000    0.2500    0.2000

   #              mat nBars area       yI      zI         yF        zF
   layer straight 3    2    0.000645 -0.2000   0.2000    -0.2000   -0.2000
   layer straight 3    2    0.000645  0.2000   0.2000     0.2000   -0.2000
}
#   steel:  Ag/As = 4*0.000645/(0.4*0.4) =1.6%

# -----------------------------------------------------------------------------------------------------

# NUMBER OF INTEGRATION POINTS
set nP 5

# GEOMETRIC TRANSFORMATION
geomTransf Linear 1

# ----------------- DEFINE DISPLACEMENT BEAM-COLUMN ELEMENT ----------------------
# --------------------------- UPPERGROUND FRAMES -------------------------

# Columns                             tag ndI ndJ  nPts secID transf
element dispBeamColumnWithSensitivity  1   1   2    $nP   2      1
element dispBeamColumnWithSensitivity  2   2   3    $nP   2      1
element dispBeamColumnWithSensitivity  3   4   5    $nP   1      1
element dispBeamColumnWithSensitivity  4   5   6    $nP   1      1
element dispBeamColumnWithSensitivity  5   7   8    $nP   2      1
element dispBeamColumnWithSensitivity  6   8   9    $nP   2      1

# Beams
element dispBeamColumnWithSensitivity  7   2   5    $nP   3      1
element dispBeamColumnWithSensitivity  8   5   8    $nP   3      1
element dispBeamColumnWithSensitivity  9   3   6    $nP   3      1
element dispBeamColumnWithSensitivity 10   6   9    $nP   3      1






# ============= SENSITIVITY ANALYSIS MODEL =================


# -----------------1: Core  fc  -------------------------------

set h fc
set gradNumber 1


parameter      $gradNumber   -element 1    -section 2 -material 2 $h
addToParameter $gradNumber   -element 2    -section 2 -material 2 $h
addToParameter $gradNumber   -element 3    -section 1 -material 2 $h
addToParameter $gradNumber   -element 4    -section 1 -material 2 $h
addToParameter $gradNumber   -element 5    -section 2 -material 2 $h
addToParameter $gradNumber   -element 6    -section 2 -material 2 $h

## -----------------2: Core  fcu  -------------------------------

set h fcu
set gradNumber 2

parameter      $gradNumber   -element 1    -section 2 -material 2 $h
addToParameter $gradNumber   -element 2    -section 2 -material 2 $h
addToParameter $gradNumber   -element 3    -section 1 -material 2 $h
addToParameter $gradNumber   -element 4    -section 1 -material 2 $h
addToParameter $gradNumber   -element 5    -section 2 -material 2 $h
addToParameter $gradNumber   -element 6    -section 2 -material 2 $h



#-------------------- 3: Core epsco -----------------------

set h epsco
set gradNumber 3

parameter      $gradNumber   -element 1    -section 2 -material 2 $h
addToParameter $gradNumber   -element 2    -section 2 -material 2 $h
addToParameter $gradNumber   -element 3    -section 1 -material 2 $h
addToParameter $gradNumber   -element 4    -section 1 -material 2 $h
addToParameter $gradNumber   -element 5    -section 2 -material 2 $h
addToParameter $gradNumber   -element 6    -section 2 -material 2 $h

# -----------------4: Core epscu ---------------------

set h epscu
set gradNumber 4


parameter      $gradNumber   -element 1    -section 2 -material 2 $h
addToParameter $gradNumber   -element 2    -section 2 -material 2 $h
addToParameter $gradNumber   -element 3    -section 1 -material 2 $h
addToParameter $gradNumber   -element 4    -section 1 -material 2 $h
addToParameter $gradNumber   -element 5    -section 2 -material 2 $h
addToParameter $gradNumber   -element 6    -section 2 -material 2 $h


# -----------------5: Core  Ec  -------------------------------


set h Ec
set gradNumber 5


parameter      $gradNumber   -element 1    -section 2 -material 2 $h
addToParameter $gradNumber   -element 2    -section 2 -material 2 $h
addToParameter $gradNumber   -element 3    -section 1 -material 2 $h
addToParameter $gradNumber   -element 4    -section 1 -material 2 $h
addToParameter $gradNumber   -element 5    -section 2 -material 2 $h
addToParameter $gradNumber   -element 6    -section 2 -material 2 $h


# -----------------6: Steel E  -------------------------------

set h E
set gradNumber 6


parameter      $gradNumber   -element 1    -section 2 -material 3 $h
addToParameter $gradNumber   -element 2    -section 2 -material 3 $h
addToParameter $gradNumber   -element 3    -section 1 -material 3 $h
addToParameter $gradNumber   -element 4    -section 1 -material 3 $h
addToParameter $gradNumber   -element 5    -section 2 -material 3 $h
addToParameter $gradNumber   -element 6    -section 2 -material 3 $h
addToParameter $gradNumber   -element 7    -section 3 -material 3 $h
addToParameter $gradNumber   -element 8    -section 3 -material 3 $h
addToParameter $gradNumber   -element 9    -section 3 -material 3 $h
addToParameter $gradNumber   -element 10   -section 3 -material 3 $h




                                                                    
#-------------------- 7: Upper structural sigmaY -----------------------


set h sigmaY
set gradNumber 7


parameter      $gradNumber   -element 1    -section 2 -material 3 $h
addToParameter $gradNumber   -element 2    -section 2 -material 3 $h
addToParameter $gradNumber   -element 3    -section 1 -material 3 $h
addToParameter $gradNumber   -element 4    -section 1 -material 3 $h
addToParameter $gradNumber   -element 5    -section 2 -material 3 $h
addToParameter $gradNumber   -element 6    -section 2 -material 3 $h
addToParameter $gradNumber   -element 7    -section 3 -material 3 $h
addToParameter $gradNumber   -element 8    -section 3 -material 3 $h
addToParameter $gradNumber   -element 9    -section 3 -material 3 $h
addToParameter $gradNumber   -element 10   -section 3 -material 3 $h





#-------------------- 8: Upper structural b -----------------------

set h b
set gradNumber 8


parameter      $gradNumber   -element 1    -section 2 -material 3 $h
addToParameter $gradNumber   -element 2    -section 2 -material 3 $h
addToParameter $gradNumber   -element 3    -section 1 -material 3 $h
addToParameter $gradNumber   -element 4    -section 1 -material 3 $h
addToParameter $gradNumber   -element 5    -section 2 -material 3 $h
addToParameter $gradNumber   -element 6    -section 2 -material 3 $h
addToParameter $gradNumber   -element 7    -section 3 -material 3 $h
addToParameter $gradNumber   -element 8    -section 3 -material 3 $h
addToParameter $gradNumber   -element 9    -section 3 -material 3 $h
addToParameter $gradNumber   -element 10   -section 3 -material 3 $h





recorder Node -file node.out -time -node 6 -dof  1 2 -precision 16  disp

recorder Node -file node_sens1.out -time -node 6 -dof  1 "sensitivity 1"
recorder Node -file node_sens2.out -time -node 6 -dof  1 "sensitivity 2"
recorder Node -file node_sens3.out -time -node 6 -dof  1 "sensitivity 3"
recorder Node -file node_sens4.out -time -node 6 -dof  1 "sensitivity 4"
recorder Node -file node_sens5.out -time -node 6 -dof  1 "sensitivity 5"
recorder Node -file node_sens6.out -time -node 6 -dof  1 "sensitivity 6"
recorder Node -file node_sens7.out -time -node 6 -dof  1 "sensitivity 7"
recorder Node -file node_sens8.out -time -node 6 -dof  1 "sensitivity 8"


constraints Transformation
numberer RCM
#test NormUnbalance 1.0e-6 25 0
test NormDispIncr 1.0e-9  50 
integrator LoadControl 1 1 1  1
algorithm Newton
system BandGeneral

analysis Static 
#sensitivityIntegrator -static
sensitivityAlgorithm -computeAtEachStep


 


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
#sensitivityIntegrator -definedAbove
sensitivityAlgorithm -computeAtEachStep


pattern UniformExcitation    2     1    -accel "Series -factor 1 -filePath acce.txt -factor 10 -dt 0.01"


set startT [clock seconds]
analyze 1000 0.01
set endT [clock seconds]
puts "Execution time: [expr $endT-$startT] seconds."
