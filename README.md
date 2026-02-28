# 🔥 Strut 🔥

GPU-first finite element analysis (FEA) research sandbox written in Mojo. The project focuses on structural mechanics workloads and aims to squeeze as much performance as possible from modern accelerators while remaining easy to validate against analytical solutions and other state of the art solvers.

## Background

This is a test project for now, primarily just to see how far I can push the vibes. The setup is a bit specific for my environment, and not very general right now.

- AMD GPU
- On Linux (using wine to run OpenSees)

## Quick Start

1. Fetch the dependencies `uv sync`
2. Download `OpenSees.exe` and necessary `tcl` files, put into [OpenSees](./benchmark/OpenSees/)
3. Run the [setup script](./scripts/setup.sh) to download docs and repos that can be used by the agent

## OpenSees Coverage Checklist

Element families below are based on the local OpenSees reference tree (`docs/agent-reference/OpenSees/SRC/element`) and current Strut implementation status. This is a family-level view (not the full command/alias list from `OpenSeesElementCommands.cpp`).

Legend: `✅` implemented in Strut, `❌` available in OpenSees but not implemented in Strut, blank cell = not available/not-applicable for that 2D/3D slot at family level. `Total` is full family coverage (`✅` only when all applicable 2D/3D variants are implemented; for non-2D/3D families it is the direct full-family status).

| Element family              | 2D  | 3D  | Total |
| --------------------------- | --- | --- | ----- |
| CEqElement                  | ❌  | ❌  | ❌    |
| HUelements                  |     | ❌  | ❌    |
| IGA                         |     |     | ❌    |
| LHMYS                       | ❌  |     | ❌    |
| PFEMElement                 | ❌  | ❌  | ❌    |
| PML                         | ❌  | ❌  | ❌    |
| RockingBC                   |     |     | ❌    |
| UP-ucsd                     | ❌  | ❌  | ❌    |
| UWelements                  | ❌  | ❌  | ❌    |
| XMUelements                 |     | ❌  | ❌    |
| absorbentBoundaries         | ❌  | ❌  | ❌    |
| adapter                     |     |     | ❌    |
| beam2d                      | ❌  |     | ❌    |
| beam3d                      |     | ❌  | ❌    |
| beamWithHinges              | ❌  | ❌  | ❌    |
| brick                       |     | ❌  | ❌    |
| catenaryCable               |     |     | ❌    |
| componentElement            | ❌  | ❌  | ❌    |
| dispBeamColumn              | ✅  | ❌  | ❌    |
| dispBeamColumnInt           | ❌  |     | ❌    |
| dmglib                      |     |     | ❌    |
| elasticBeamColumn           | ✅  | ✅  | ✅    |
| elastomericBearing          | ❌  | ❌  | ❌    |
| feap                        |     |     | ❌    |
| fmkPlanarTruss              | ❌  |     | ❌    |
| forceBeamColumn             | ✅  | ❌  | ❌    |
| fourNodeQuad                | ✅  |     | ✅    |
| frictionBearing             | ❌  | ❌  | ❌    |
| generic                     |     |     | ❌    |
| gradientInelasticBeamColumn | ❌  | ❌  | ❌    |
| joint                       | ❌  | ❌  | ❌    |
| masonry                     | ❌  | ❌  | ❌    |
| mefi                        |     |     | ❌    |
| mixedBeamColumn             | ❌  | ❌  | ❌    |
| mvlem                       | ❌  | ❌  | ❌    |
| nonlinearBeamColumn         | ✅  | ❌  | ❌    |
| pipe                        |     |     | ❌    |
| pyMacro                     | ❌  |     | ❌    |
| shell                       |     | ✅  | ✅    |
| surfaceLoad                 |     |     | ❌    |
| tetrahedron                 |     | ❌  | ❌    |
| triangle                    | ❌  |     | ❌    |
| truss                       | ✅  | ✅  | ✅    |
| twoNodeLink                 | ✅  | ❌  | ❌    |
| updatedLagrangianBeamColumn | ❌  |     | ❌    |
| zeroLength                  | ✅  | ❌  | ❌    |

Note: `nonlinearBeamColumn` currently maps to the `forceBeamColumn2d` path in Strut input parsing.

### Uniaxial Materials (`uniaxialMaterial`)

| Material             | Implemented in Strut |
| -------------------- | -------------------- |
| APDFMD               | ❌                   |
| APDMD                | ❌                   |
| APDVFD               | ❌                   |
| ASDConcrete1D        | ❌                   |
| ASDSteel1D           | ❌                   |
| ASD_SMA_3K           | ❌                   |
| AxialSp              | ❌                   |
| AxialSpHD            | ❌                   |
| BWBN                 | ❌                   |
| Backbone             | ❌                   |
| BarSlip              | ❌                   |
| Bilin                | ❌                   |
| Bilin02              | ❌                   |
| BilinMaterial        | ❌                   |
| Bond                 | ❌                   |
| Bond_SP01            | ❌                   |
| BoucWen              | ❌                   |
| BoucWenInfill        | ❌                   |
| BoucWenOriginal      | ❌                   |
| CFSSSWP              | ❌                   |
| CFSWSWP              | ❌                   |
| Cable                | ❌                   |
| Cast                 | ❌                   |
| CastFuse             | ❌                   |
| Concrete01           | ✅                   |
| Concrete02           | ✅                   |
| Concrete02IS         | ❌                   |
| Concrete04           | ❌                   |
| Concrete06           | ❌                   |
| Concrete07           | ❌                   |
| ConcreteCM           | ❌                   |
| ConcreteD            | ❌                   |
| ConcreteL01          | ❌                   |
| ConcreteZ01          | ❌                   |
| ConcreteZBH_fitted   | ❌                   |
| ConcreteZBH_original | ❌                   |
| ConcreteZBH_smoothed | ❌                   |
| ConcretewBeta        | ❌                   |
| CoulombDamper        | ❌                   |
| Creep                | ❌                   |
| CreepShrinkageACI209 | ❌                   |
| Damper               | ❌                   |
| DamperMaterial       | ❌                   |
| DoddRestr            | ❌                   |
| DoddRestrepo         | ❌                   |
| Dodd_Restrepo        | ❌                   |
| DowelType            | ❌                   |
| DuctileFracture      | ❌                   |
| ECC01                | ❌                   |
| ENT                  | ❌                   |
| Elastic              | ✅                   |
| Elastic2             | ❌                   |
| ElasticBilin         | ❌                   |
| ElasticBilinear      | ❌                   |
| ElasticPP            | ❌                   |
| ElasticPPGap         | ❌                   |
| ElasticPowerFunc     | ❌                   |
| ElasticThermal       | ❌                   |
| FRCC                 | ❌                   |
| Fatigue              | ❌                   |
| FlagShape            | ❌                   |
| GNG                  | ❌                   |
| Hardening            | ❌                   |
| HertzDamp            | ❌                   |
| Hertzdamp            | ❌                   |
| HookGap              | ❌                   |
| Hysteretic           | ❌                   |
| HystereticAsym       | ❌                   |
| HystereticSM         | ❌                   |
| HystereticSmooth     | ❌                   |
| IMKBilin             | ❌                   |
| IMKPeakOriented      | ❌                   |
| IMKPinching          | ❌                   |
| Impact               | ❌                   |
| ImpactMaterial       | ❌                   |
| InitStrain           | ❌                   |
| InitStress           | ❌                   |
| JankowskiImpact      | ❌                   |
| KikuchiAikenHDR      | ❌                   |
| KikuchiAikenLRB      | ❌                   |
| LimitState           | ❌                   |
| Masonry              | ❌                   |
| Masonryt             | ❌                   |
| Maxwell              | ❌                   |
| MaxwellMaterial      | ❌                   |
| MinMax               | ❌                   |
| MinMaxMaterial       | ❌                   |
| ModIMKPinching       | ❌                   |
| ModIMKPinching02     | ❌                   |
| MultiLinear          | ❌                   |
| Multiplier           | ❌                   |
| OOHysteretic         | ❌                   |
| OriginCentered       | ❌                   |
| PYUCLA               | ❌                   |
| Parallel             | ❌                   |
| Penalty              | ❌                   |
| Pinching4            | ❌                   |
| Pipe                 | ❌                   |
| PyLiq1               | ❌                   |
| PySimple1            | ❌                   |
| PySimple2            | ❌                   |
| QbSandCPT            | ❌                   |
| QzLiq1               | ❌                   |
| QzSimple1            | ❌                   |
| QzSimple2            | ❌                   |
| RambergOsgood        | ❌                   |
| Ratchet              | ❌                   |
| ReinforcingSteel     | ❌                   |
| ResilienceLow        | ❌                   |
| Restrepo             | ❌                   |
| SAWS                 | ❌                   |
| SAWSMaterial         | ❌                   |
| SLModel              | ❌                   |
| SMA                  | ❌                   |
| SPSW02               | ❌                   |
| Series               | ❌                   |
| ShearPanel           | ❌                   |
| SmoothPSConcrete     | ❌                   |
| Steel01              | ✅                   |
| Steel01Thermal       | ❌                   |
| Steel02              | ✅                   |
| Steel02Fatigue       | ❌                   |
| Steel02Thermal       | ❌                   |
| Steel03              | ❌                   |
| Steel2               | ❌                   |
| Steel4               | ❌                   |
| SteelBRB             | ❌                   |
| SteelDRC             | ❌                   |
| SteelFractureDI      | ❌                   |
| SteelMP              | ❌                   |
| SteelMPF             | ❌                   |
| SteelZ01             | ❌                   |
| SteelZ01Material     | ❌                   |
| TDConcrete           | ❌                   |
| TDConcreteEXP        | ❌                   |
| TDConcreteMC10       | ❌                   |
| TDConcreteMC10NL     | ❌                   |
| TDConcreteNL         | ❌                   |
| TendonL01            | ❌                   |
| TensionOnly          | ❌                   |
| Trilinwp             | ❌                   |
| Trilinwp2            | ❌                   |
| TzLiq1               | ❌                   |
| TzSandCPT            | ❌                   |
| TzSimple1            | ❌                   |
| TzSimple2            | ❌                   |
| UVCuniaxial          | ❌                   |
| ViscoelasticGap      | ❌                   |
| Viscous              | ❌                   |
| ViscousDamper        | ❌                   |
| pyUCLA               | ❌                   |

### ND Materials (`nDMaterial`)

| Material                         | Implemented in Strut |
| -------------------------------- | -------------------- |
| 3DJ2                             | ❌                   |
| ASDConcrete3D                    | ❌                   |
| ASDPlasticMaterial               | ❌                   |
| ASDPlasticMaterial3D             | ❌                   |
| AcousticMedium                   | ❌                   |
| BeamFiber                        | ❌                   |
| BeamFiber2d                      | ❌                   |
| BeamFiber2dPS                    | ❌                   |
| BeamFiberMaterial                | ❌                   |
| BoundingCamClay                  | ❌                   |
| CapPlasticity                    | ❌                   |
| ConcreteMcftNonlinear5           | ❌                   |
| ConcreteMcftNonlinear7           | ❌                   |
| ConcreteS                        | ❌                   |
| ContactMaterial2D                | ❌                   |
| ContactMaterial3D                | ❌                   |
| CycLiqCP                         | ❌                   |
| CycLiqCPSP                       | ❌                   |
| Damage2p                         | ❌                   |
| DruckerPrager                    | ❌                   |
| ElasticIsotropic                 | ✅                   |
| ElasticIsotropic3D               | ❌                   |
| ElasticIsotropic3DThermal        | ❌                   |
| ElasticIsotropicThermal          | ❌                   |
| ElasticOrthotropic               | ❌                   |
| ElasticOrthotropic3D             | ❌                   |
| ElasticOrthotropicPlaneStress    | ❌                   |
| ElasticPlaneStress               | ❌                   |
| FAFourSteelPCPlaneStress         | ❌                   |
| FAFourSteelRCPlaneStress         | ❌                   |
| FAPrestressedConcretePlaneStress | ❌                   |
| FAReinforceConcretePlaneStress   | ❌                   |
| FAReinforcedConcretePlaneStress  | ❌                   |
| FSAM                             | ❌                   |
| FluidSolidPorous                 | ❌                   |
| InitStrain                       | ❌                   |
| InitStress                       | ❌                   |
| InitStressND                     | ❌                   |
| InitStressNDMaterial             | ❌                   |
| InitialStateAnalysisWrapper      | ❌                   |
| J2                               | ❌                   |
| J2BeamFiber                      | ❌                   |
| J2Plasticity                     | ❌                   |
| J2PlasticityThermal              | ❌                   |
| J2PlateFibre                     | ❌                   |
| J2Thermal                        | ❌                   |
| LinearElasticGGmax               | ❌                   |
| MCP                              | ❌                   |
| ManzariDafalias                  | ❌                   |
| MaterialCMM                      | ❌                   |
| MinMax                           | ❌                   |
| MultiYieldSurfaceClay            | ❌                   |
| MultiaxialCyclicPlasticity       | ❌                   |
| Orthotropic                      | ❌                   |
| OrthotropicRAConcrete            | ❌                   |
| PM4Sand                          | ❌                   |
| PM4Silt                          | ❌                   |
| Parallel3D                       | ❌                   |
| PlaneStrain                      | ❌                   |
| PlaneStrainMaterial              | ❌                   |
| PlaneStress                      | ❌                   |
| PlaneStressMaterial              | ❌                   |
| PlaneStressSimplifiedJ2          | ❌                   |
| PlaneStressUserMaterial          | ❌                   |
| PlasticDamageConcrete3d          | ❌                   |
| PlasticDamageConcretePlaneStress | ❌                   |
| PlateFiber                       | ❌                   |
| PlateFiberMaterial               | ❌                   |
| PlateFiberMaterialThermal        | ❌                   |
| PlateFiberThermal                | ❌                   |
| PlateFromPlaneStress             | ❌                   |
| PlateFromPlaneStressMaterial     | ❌                   |
| PlateFromPlaneStressThermal      | ❌                   |
| PlateRebar                       | ❌                   |
| PlateRebarMaterial               | ❌                   |
| PlateRebarMaterialThermal        | ❌                   |
| PlateRebarThermal                | ❌                   |
| PressureDependMultiYield         | ❌                   |
| PressureDependMultiYield02       | ❌                   |
| PressureDependMultiYield03       | ❌                   |
| PressureDependentElastic3D       | ❌                   |
| PressureIndependMultiYield       | ❌                   |
| PrestressedConcretePlaneStress   | ❌                   |
| RAFourSteelPCPlaneStress         | ❌                   |
| RAFourSteelRCPlaneStress         | ❌                   |
| ReinforceConcretePlaneStress     | ❌                   |
| ReinforcedConcretePlaneStress    | ❌                   |
| SAniSandMS                       | ❌                   |
| Series3D                         | ❌                   |
| Simplified3DJ2                   | ❌                   |
| SmearedSteelDoubleLayer          | ❌                   |
| StressDensityModel               | ❌                   |
| TruncatedDP                      | ❌                   |
| UVCmultiaxial                    | ❌                   |
| UVCplanestress                   | ❌                   |
| VonPapaDamage                    | ❌                   |
