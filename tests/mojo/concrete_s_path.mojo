from sys.arg import argv
from materials.concrete_s import (
    ConcreteSDef,
    ConcreteSState,
    PlaneStressTangent,
    PlaneStressVector,
    concrete_s_set_trial_strain,
)


def main():
    var args = argv()
    var material = ConcreteSDef(36e9, 0.2, -38.5e6, 2.85e6, -3.6e9)
    var state = ConcreteSState()
    var stress = PlaneStressVector(fill=0.0)
    var tangent = PlaneStressTangent(fill=0.0)
    var commit_interval = atol(args[1])
    var point = 0
    for i in range(2, len(args), 3):
        concrete_s_set_trial_strain(
            material, state, atof(args[i]), atof(args[i + 1]), atof(args[i + 2]), stress, tangent
        )
        point += 1
        if point % commit_interval == 0:
            state.commit()
        print(
            "ROW",
            stress[0],
            stress[1],
            stress[2],
            tangent[0],
            tangent[1],
            tangent[2],
            tangent[3],
            tangent[4],
            tangent[5],
            tangent[6],
            tangent[7],
            tangent[8],
        )
