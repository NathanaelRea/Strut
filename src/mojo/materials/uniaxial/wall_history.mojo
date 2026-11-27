from materials.uniaxial.concrete_cm import ConcreteCMHistory
from materials.uniaxial.steel_mpf import SteelMPFHistory


struct WallMaterialHistory(ImplicitlyCopyable, Movable):
    # Allocate history only for materials that need it. Indices 0 and 1
    # represent committed and trial history, respectively.
    var concrete: List[ConcreteCMHistory]
    var steel: List[SteelMPFHistory]

    fn __init__(out self):
        self.concrete = []
        self.steel = []

    fn __copyinit__(out self, existing: Self):
        self.concrete = existing.concrete.copy()
        self.steel = existing.steel.copy()

    fn commit(mut self):
        if len(self.concrete) != 0:
            self.concrete[0] = self.concrete[1]
        if len(self.steel) != 0:
            self.steel[0] = self.steel[1]

    fn revert(mut self):
        if len(self.concrete) != 0:
            self.concrete[1] = self.concrete[0]
        if len(self.steel) != 0:
            self.steel[1] = self.steel[0]
