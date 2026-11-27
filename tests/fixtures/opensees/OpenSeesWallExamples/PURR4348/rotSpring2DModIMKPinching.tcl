# Modified Ibarra-Medina-Krawinkler Deterioration Model with Pinched Hysteretic Response

#   $matTag	        integer tag identifying material
#   $K0	            elastic stiffness
#   $as_Plus	    strain hardening ratio for positive loading direction
#   $as_Neg	        strain hardening ratio for negative loading direction
#   $My_Plus	    effective yield strength for positive loading direction
#   $My_Neg	        effective yield strength for negative loading direction (Must be defined as a negative value)
#   $FprPos	        Ratio of the force at which reloading begins to force corresponding to the maximum historic deformation demand (positive loading direction)
#   $FprNeg	        Ratio of the force at which reloading begins to force corresponding to the absolute maximum historic deformation demand (negative loading direction)
#   $A_Pinch	    Ratio of reloading stiffness
#   $Lamda_S	    Cyclic deterioration parameter for strength deterioration [E_t=Lamda_S*M_y, see Lignos and Krawinkler (2011); set Lamda_S = 0 to disable this mode of deterioration]
#   $Lamda_C	    Cyclic deterioration parameter for post-capping strength deterioration [E_t=Lamda_C*M_y, see Lignos and Krawinkler (2011); set Lamda_C = 0 to disable this mode of deterioration]
#   $Lamda_A	    Cyclic deterioration parameter for accelerated reloading stiffness deterioration [E_t=Lamda_A*M_y, see Lignos and Krawinkler (2011); set Lamda_A = 0 to disable this mode of deterioration]
#   $Lamda_K	    Cyclic deterioration parameter for unloading stiffness deterioration [E_t=Lamda_K*M_y, see Lignos and Krawinkler (2011); set Lamda_K = 0 to disable this mode of deterioration]
#   $c_S	        rate of strength deterioration. The default value is 1.0.
#   $c_C	        rate of post-capping strength deterioration. The default value is 1.0.
#   $c_A	        rate of accelerated reloading deterioration. The default value is 1.0.
#   $c_K	        rate of unloading stiffness deterioration. The default value is 1.0.
#   $theta_p_Plus	pre-capping rotation for positive loading direction (often noted as plastic rotation capacity)
#   $theta_p_Neg	pre-capping rotation for negative loading direction (often noted as plastic rotation capacity) (must be defined as a positive value)
#   $theta_pc_Plus	post-capping rotation for positive loading direction
#   $theta_pc_Neg	post-capping rotation for negative loading direction (must be defined as a positive value)
#   $Res_Pos	    residual strength ratio for positive loading direction
#   $Res_Neg	    residual strength ratio for negative loading direction (must be defined as a positive value)
#   $theta_u_Plus	ultimate rotation capacity for positive loading direction
#   $theta_u_Neg	ultimate rotation capacity for negative loading direction (must be defined as a positive value)
#   $D_Plus 	    rate of cyclic deterioration in the positive loading direction (this parameter is used to create assymetric hysteretic behavior for the case of a composite beam). For symmetric hysteretic response use 1.0.
#   $D_Neg	        rate of cyclic deterioration in the negative loading direction (this parameter is used to create assymetric hysteretic behavior for the case of a composite beam). For symmetric hysteretic response use 1.0.
#   $EA			effective modulus for axial deformations


# http://opensees.berkeley.edu/wiki/index.php/Modified_Ibarra-Medina-Krawinkler_Deterioration_Model_with_Pinched_Hysteretic_Response_(ModIMKPinching_Material)

proc rotSpring2DModIMKPinching {eleID nodeR nodeC K0 as_Pos as_Neg My_Plus My_Neg FprPos FprNeg A_pinch Lamda_S Lamda_C Lamda_A Lamda_K c_S c_C c_A c_K theta_p_Plus theta_p_Neg theta_pc_Plus theta_pc_Neg Res_Pos Res_Neg theta_u_Plus theta_u_Neg D_Plus D_Neg} {
    
    uniaxialMaterial ModIMKPinching $eleID $K0 $as_Pos $as_Neg $My_Plus $My_Neg $FprPos $FprNeg $A_pinch $Lamda_S $Lamda_C $Lamda_A $Lamda_K $c_S $c_C $c_A $c_K $theta_p_Plus $theta_p_Neg $theta_pc_Plus $theta_pc_Neg $Res_Pos $Res_Neg $theta_u_Plus $theta_u_Neg $D_Plus $D_Neg
    
	#uniaxialMaterial Elastic [expr {$eleID*2}] $EA

    element zeroLength $eleID $nodeR $nodeC -mat $eleID -dir 6
 
	#section Aggregator $eleID [expr {$eleID*2}] P $eleID Mz
    #equalDOF    $nodeR     $nodeC     2   
    equalDOF    $nodeR     $nodeC     1     2
}