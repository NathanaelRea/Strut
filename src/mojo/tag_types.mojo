struct ElementTypeTag:
    alias Unknown = 0
    alias ElasticBeamColumn2d = 1
    alias ForceBeamColumn2d = 2
    alias DispBeamColumn2d = 3
    alias ElasticBeamColumn3d = 5
    alias Truss = 6
    alias ZeroLength = 7
    alias TwoNodeLink = 13
    alias FourNodeQuad = 8
    alias Shell = 9
    alias ZeroLengthSection = 10
    alias ForceBeamColumn3d = 11
    alias DispBeamColumn3d = 12


struct ElementLoadTypeTag:
    alias Unknown = 0
    alias BeamUniform = 1
    alias BeamPoint = 2


struct GeomTransfTag:
    alias Unknown = 0
    alias Linear = 1
    alias PDelta = 2
    alias Corotational = 3


struct BeamIntegrationTag:
    alias Unknown = 0
    alias Lobatto = 1
    alias Legendre = 2
    alias Radau = 3


struct RecorderTypeTag:
    alias Unknown = 0
    alias NodeDisplacement = 1
    alias ElementForce = 2
    alias NodeReaction = 3
    alias Drift = 4
    alias EnvelopeElementForce = 5
    alias ModalEigen = 6
    alias SectionForce = 7
    alias SectionDeformation = 8
    alias ElementLocalForce = 9
    alias ElementBasicForce = 10
    alias ElementDeformation = 11


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
    alias UZ = 3
    alias RX = 4
    alias RY = 5
    alias RZ = 6
