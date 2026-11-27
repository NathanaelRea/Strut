from sys.arg import argv

from materials.pressure_independ_multi_yield import (
    PressureIndependMultiYieldDef,
    PressureIndependMultiYieldState,
    SoilVector,
)


def main():
    var args = argv()
    var material = PressureIndependMultiYieldDef(
        1.8, 50000.0, 100000.0, 25.0, 0.1, atof(args[1]), 100.0, 0.5, 20, []
    )
    var point = PressureIndependMultiYieldState(material)
    for i in range(2, len(args), 4):
        var stage = atol(args[i])
        var strain = SoilVector(fill=0.0)
        strain[0] = atof(args[i + 1])
        strain[1] = atof(args[i + 2])
        strain[3] = atof(args[i + 3])
        point.set_trial_strain(material, stage, strain)
        var tangent = point.tangent(material, stage)
        point.commit(stage)
        var row = String("ROW")
        var stress = point.stress_c.vector()
        for component in range(4):
            row += " " + String(stress[component])
        row += " " + String(point.committed_stress_ratio(stage))
        var plane_strain_components = [0, 1, 3]
        for a in plane_strain_components:
            for b in plane_strain_components:
                row += " " + String(tangent[6 * a + b])
        print(row)
