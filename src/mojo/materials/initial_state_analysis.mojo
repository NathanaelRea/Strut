from materials.multi_yield import SoilVector, _add_scaled


struct InitialStateAnalysisPoint(ImplicitlyCopyable, Movable):
    var offset: SoilVector

    fn __init__(out self):
        self.offset = SoilVector(fill=0.0)

    fn material_strain(self, element_strain: SoilVector) -> SoilVector:
        return _add_scaled(element_strain, self.offset, 1.0)

    fn capture(mut self, material_strain: SoilVector):
        # InitialStateAnalysisWrapper adds the material's total strain at
        # revertToStart while its global flag remains enabled.
        self.offset = _add_scaled(self.offset, material_strain, 1.0)
