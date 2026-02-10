struct ElementTypeTag:
    alias Unknown = 0
    alias ElasticBeamColumn2d = 1
    alias ForceBeamColumn2d = 2
    alias ElasticBeamColumn3d = 3
    alias Truss = 4
    alias Link = 5
    alias FourNodeQuad = 6
    alias Shell = 7
    alias ZeroLengthSection = 8


struct GeomTransfTag:
    alias Unknown = 0
    alias Linear = 1
    alias PDelta = 2
    alias Corotational = 3


struct RecorderTypeTag:
    alias Unknown = 0
    alias NodeDisplacement = 1
    alias ElementForce = 2
    alias NodeReaction = 3
    alias Drift = 4
    alias EnvelopeElementForce = 5
    alias ModalEigen = 6


struct TimeSeriesTypeTag:
    alias Unknown = 0
    alias Constant = 1
    alias Linear = 2
    alias Path = 3
    alias Trig = 4


struct UniMaterialTypeTag:
    alias Elastic = 0
    alias Steel01 = 1
    alias Concrete01 = 2
    alias Steel02 = 3
    alias Concrete02 = 4


struct NonlinearAlgorithmMode:
    alias Unknown = -1
    alias Newton = 0
    alias ModifiedNewton = 1
    alias ModifiedNewtonInitial = 2


struct LinkDirectionTag:
    alias UX = 1
    alias UY = 2
