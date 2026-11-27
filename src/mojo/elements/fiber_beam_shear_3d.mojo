from utils import StaticTuple


# In OpenSees ForceBeamColumn3d the Vy row contributes equally to both Mz
# end DOFs. Vz has the same interpolation in the My plane.
@fieldwise_init
struct FiberBeamShear3D(ImplicitlyCopyable, Movable):
    var flexibility_y: Float64
    var flexibility_z: Float64
    var load_deformation_y: Float64
    var load_deformation_z: Float64

    fn y_deformation(self, moment_i: Float64, moment_j: Float64) -> Float64:
        return self.flexibility_y * (moment_i + moment_j) + self.load_deformation_y

    fn z_deformation(self, moment_i: Float64, moment_j: Float64) -> Float64:
        return self.flexibility_z * (moment_i + moment_j) + self.load_deformation_z

    fn add_flexibility(self, mut matrix: StaticTuple[Float64, 25]):
        for i in range(1, 3):
            for j in range(1, 3):
                matrix[5 * i + j] += self.flexibility_y
                matrix[5 * (i + 2) + j + 2] += self.flexibility_z
