Code included in this repository was developed to analyze the composite plate shear wall systems subjected to various ground motions and scale factors. This code takes the wall geometries as input in the RunCW_Planar file, generates an OpenSees fiber model for these walls, and then subjects the wall to a user provided acceleration time history at user provided scaling factors. 

To run this code, users need to have OpenSees downloaded. This software is a free download available from: https://opensees.berkeley.edu/OpenSees/user/download.php This code was built and run in 2018-2020 but does run in the 2023 version of OpenSees and likely will run in any version that is backwards compatable. 

This code was developed by Morgan Broberg and Amit Varma. Some of the referenced codes include files created by Silvia Mazzoni & Frank McKenna.

This modeling approach is discussed in greater detail in several publications including: 

Bruneau, M., Varma, A. H., Kizilarslan, E., Broberg, M., Shafaei, S., and Seo, J. (2019). "R- Factors for Coupled Composite Plate Shear Walls /Concrete Filled (CC-PSW/CF)." Charles Pankow Foundation Grant # 05-17, McLean, Virginia. 
Broberg, M., Agrawal, S., Varma, A., and Klemencic, R. (2023). “Seismic design parameters (R, Cd, and Ω0) for uncoupled composite plate shear walls−concrete filled (C-PSW/CF).” Earthquake Engineering and Structural Dynamics, 1- 22. https://doi.org/10.1002/eqe.3917
Broberg, M., Shafaei, S., Kizilarslan, E., Seo, J., Varma, A. H., Bruneau, M., and Klemencic, R. (2022). "Capacity design of coupled composite plate shear walls – concrete-filled system." Journal of Structural Engineering, 148(4). http://doi.org/10.1061/(asce)st.1943-541x.0003296 
Shafaei, S., Varma, A. H., Broberg, M., Klemencic, R. (2021). "Modeling the cyclic behavior of composite plate shear walls/concrete filled (C-PSW/CF)." Journal of Constructional Steel Research, 184. http://doi.org/10.1016/j.jcsr.2021.106810 
Kizilarslan, E., Broberg, M., Shafaei, S., Varma, A. H., and Bruneau, M. (2021). "Non-linear analysis models for composite plate shear walls-concrete filled (C-PSW/CF)." Journal of Constructional Steel Research, 184. http://doi.org/10.1016/j.jcsr.2021.106803 
Kizilarslan, E., Broberg, M., Shafaei, S., Varma, A. H., and Bruneau, M. (2021). "Seismic design coefficients and factors for coupled composite plate shear walls/ concrete filled (CC-PSW/CF)." Engineering Structures, 244. http://doi.org/10.1016/j.engstruct.2021.112766 

To use this code, you will need a ground motion file consisting of ground acceleration values at a constant time step. The file naming convention is as follows: [Record ID Number]-[Record Component Number]-[Record Time Step]-[Number of Entries in the Record]. A sample record (1-1-0.01-3000.txt) has been uploaded with these files. This file is adapted from the PEER Ground Motion Database (https://ngawest2.berkeley.edu/)

To run this program:
(1) Ensure that all files are downloaded and in the same folder
(2) Open OpenSees and change directories to the folder these files are in.
(2) Command 1: source RunCW_Planar.tcl
(3) Command 2: Run [ground motion text file name] [scale factors to consider]

This command will result in outputs for each groundmotion/scale factor combination including the concrete and steel tension, compression, and cyclic behavior, composite section moment curvature, composite wall static pushover, and composite wall time history response.

If users would like to change the wall geometry, floor geometry, or material models, this can be done in the "RunCW_PLanar.tcl" file. Default values used for "PG-1B" from the Bruneau et al (2019) study are currently input in this file.
