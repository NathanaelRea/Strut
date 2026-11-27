from memory import ArcPointer
from os import abort

from materials.uniaxial.core import (
    ParallelComponent,
    UniMaterialDef,
    UniMaterialLeafDef,
    uni_mat_initial_tangent,
)
from materials.uniaxial.min_max import StrainLimits
from tag_types import UniMaterialTypeTag


fn make_parallel_def(children: List[UniMaterialDef], factors: List[Float64]) -> UniMaterialDef:
    if len(children) == 0 or len(children) != len(factors):
        abort("Parallel requires one factor per component")
    var components = List[ParallelComponent](capacity=len(children))
    var tangent = 0.0
    for i in range(len(children)):
        ref child = children[i]
        var factor = factors[i]
        tangent += factor * uni_mat_initial_tangent(child)
        if child.mat_type == UniMaterialTypeTag.Parallel:
            # Flatten nested sums while giving every occurrence its own state.
            ref nested = child.components.value()[]
            for j in range(len(nested)):
                var definition = nested[j].definition
                if child.strain_limits:
                    add_strain_limits(definition.strain_limits, child.strain_limits.value())
                components.append(ParallelComponent(definition, factor * nested[j].factor))
        else:
            var leaf = UniMaterialLeafDef(
                child.mat_type,
                child.p0,
                child.p1,
                child.p2,
                child.p3,
                child.p4,
                child.p5,
                child.p6,
                child.p7,
                child.p8,
                child.p9,
                child.p10,
                child.p11,
            )
            if (
                child.mat_type == UniMaterialTypeTag.ViscousDamper
                or child.mat_type == UniMaterialTypeTag.Viscous
            ):
                abort("Parallel does not support rate-dependent viscous components")
            leaf.bilin = child.bilin
            leaf.imk_peak = child.imk_peak
            leaf.hysteretic = child.hysteretic
            leaf.fatigue = child.fatigue
            leaf.strain_limits = child.strain_limits
            components.append(ParallelComponent(leaf, factor))
    var result = UniMaterialDef(UniMaterialTypeTag.Parallel, tangent, 0.0, 0.0, 0.0)
    result.components = ArcPointer(components^)
    return result^


fn add_strain_limits(mut limits: Optional[StrainLimits], new_limits: StrainLimits):
    if limits:
        limits = limits.value().intersect(new_limits)
    else:
        limits = new_limits
