from sys.arg import argv

from materials.multi_yield import SoilVector
from materials.pressure_depend_multi_yield import (
    PressureDependMultiYieldDef,
    PressureDependMultiYieldState,
)


def main():
    var args = argv()
    var material = PressureDependMultiYieldDef(
        50000.0,
        atof(args[3]),
        35.0,
        0.1,
        100.0,
        atof(args[1]),
        27.0,
        0.06,
        0.5,
        2.5,
        atof(args[2]),
        0.02,
        1.0,
        20,
        [],
        0.6,
        0.9,
        0.02,
        0.7,
        101.0,
        0.3,
        0.0,
        1.0,
    )
    var point = PressureDependMultiYieldState(material)
    for i in range(4, len(args), 4):
        var stage = atol(args[i])
        var strain = SoilVector(fill=0.0)
        strain[0] = atof(args[i + 1])
        strain[1] = atof(args[i + 2])
        strain[3] = atof(args[i + 3])
        if stage == 1:
            var discarded_strain = strain
            discarded_strain[3] -= 0.003
            point.set_trial_strain(material, stage, discarded_strain)
            point.revert()
        point.set_trial_strain(material, stage, strain)
        point.commit(stage)
        var tangent = point.tangent(material, stage, 2)
        var row = String("ROW")
        var stress = point.stress_c.vector()
        for component in range(4):
            row += " " + String(stress[component])
        row += " " + String(point.committed_stress_ratio(material, stage))
        var plane_strain_components = [0, 1, 3]
        for a in plane_strain_components:
            for b in plane_strain_components:
                row += " " + String(tangent[6 * a + b])
        print(row)
