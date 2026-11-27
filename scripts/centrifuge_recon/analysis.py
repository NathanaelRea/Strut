"""Staged gravity followed by the three shaking sequences and intervening drainage."""

from __future__ import annotations

import re
from dataclasses import dataclass

GRAVITY_SETUP = """set gamma 1.6
integrator Newmark $gamma [expr pow($gamma+0.5, 2)/4] 0.00 0.0 0.00 0.0
test EnergyIncr 1.0e-8 400 1;
constraints Transformation
algorithm KrylovNewton
numberer RCM
system ProfileSPD
analysis Transient
"""

SHAKING_SETUP = """set pi 3.141593
set omega [expr 2*$pi/.5]
set T 0.5
set Timeseries_1 "Sine 0.0 10. $T -shift 0.0 -factor 0.6867"
set Timeseries_2 "Sine 120.0 130.0 $T -shift 0.0 -factor 1.0791"
set Timeseries_3 "Sine 240.0 250.0 $T -shift 0.0 -factor 2.9430"

pattern UniformExcitation 1 1 -accel $Timeseries_1;
pattern UniformExcitation 12 1 -accel $Timeseries_2;
pattern UniformExcitation 13 1 -accel $Timeseries_3;

set massProportionalDamping 0. ;
set InitStiffnessProportionalDamping 0.0002;
set gamma 0.65
test NormDispIncr 5.0e-4 50 2;
integrator Newmark $gamma [expr pow($gamma+0.5, 2)/4] \\
    $massProportionalDamping 0.0 $InitStiffnessProportionalDamping 0.0
constraints Transformation
algorithm KrylovNewton
numberer RCM
system ProfileSPD
analysis Transient
"""


def gravity_commands() -> str:
    stages = "".join(
        f"updateMaterialStage -material {tag} -stage 1\n" for tag in range(1, 6)
    )
    return (
        GRAVITY_SETUP
        + 'analyze 3 5.e5\nputs "End of Elastic Phase of Gravity Application"\n'
        + stages
        + 'analyze 5 5.e5\nputs "End of Gravity Application. Starting Dynamic Excitation..."\n'
        + "wipeAnalysis\nsetTime 0.0\n"
    )


@dataclass(frozen=True)
class Phase:
    max_dt: str
    cutback_dt: str
    tol: str
    min_tol: str


SHAKE = Phase("0.05", "0.0005", "5.0e-4", "1e-4")
SETTLE = Phase("0.1", "0.005", "5.0e-4", "5e-4")
DRAIN = Phase("1.", "0.05", "1.0e-3", "1e-3")
SEQUENCES = (
    ("13.", "20.", "120."),
    ("133.", "140.", "239."),
    ("253.", "260.", "360."),
)
SHAKE_MESSAGES = (
    "",
    'puts "Start the second sequence of shaking"\n',
    'puts "Starting the third and sequence of shaking"\n',
)

ADAPTIVE_LOOP = """
while {{$currentTime < {end}}} {{
    test EnergyIncr $disp_incr $iter 2;
    set ok [analyze 1 $dt]
    set currentTime [getTime]
    puts "$ok  dt=$dt  Disp_incr=$disp_incr  #Iter=$iter"
    if {{$ok ==0}} {{
        set i_suc [expr $i_suc+1]
        if {{$disp_incr>1e-2}} {{
            set i_extr_nl [expr $i_extr_nl+1]
        }}
        if {{$i_suc>10}} {{
            set disp_incr [expr $disp_incr/2]
            if {{$disp_incr<{min_tol}}} {{
                set disp_incr {min_tol}
            }}
            if {{$dt<{cutback_dt}}} {{
                set iter 30
            }}
        }}
        if {{$i_suc>20}} {{
            set dt [expr $dt*2]
            if {{$dt>{max_dt}}} {{
                set dt {max_dt}
            }}
        }}
    }}
    if {{$ok !=0}} {{
        set dt [expr $dt/2]
        if {{$i_suc==0}} {{
            set disp_incr [expr $disp_incr*2]
            if {{$disp_incr>0.02}} {{
                set disp_incr 0.02
            }}
        }}
        if {{$dt<{cutback_dt}}} {{
            set iter 200
        }}
        set i_suc 0
    }}
}}
"""


def recorders(suffix: str, interval: str) -> str:
    commands = []
    for name, dofs, response in (
        ("disp", "1 2", "disp"),
        ("accel", "1 2", "accel"),
        ("pore", "3", "vel"),
    ):
        commands.append(
            f"recorder Node -file output_{name}_{suffix}.txt"
            f" -nodeRange 1 4204 -time -dT {interval} -dof {dofs} {response}"
        )
    for response in ("stress", "strain"):
        for point in range(1, 5):
            commands.append(
                f"recorder Element -file {response}_{point}_{suffix}.txt"
                f" -time -dT {interval} -eleRange 1 3932 material {point} {response}"
            )
    return "".join(
        f"set r_{index} [{command}]\n" for index, command in enumerate(commands, 1)
    )


def shaking_commands() -> str:
    parts = [SHAKING_SETUP]
    for seq, ends in enumerate(SEQUENCES, 1):
        for stage, (phase, end) in enumerate(
            zip((SHAKE, SETTLE, DRAIN), ends, strict=True), 1
        ):
            first = seq == stage == 1
            interval = ".05" if stage == 1 and seq > 1 else phase.max_dt
            if stage == 1:
                parts.append(SHAKE_MESSAGES[seq - 1])
            parts.append(recorders(f"{stage}{seq}", interval))
            if first:
                parts.append("set startT [clock seconds]\n")
            parts.append("set ok 0\n")
            if first:
                parts.append("set currentTime 0.0\nset i_extr_nl 0.0\n")
            parts.extend(
                [
                    f"set dt {phase.max_dt}\nset i_suc 0\nset disp_incr {phase.tol}\n",
                    "set iter 30\ntest NormDispIncr $disp_incr $iter 2;\n",
                    "algorithm Newton\n" if first else "",
                    ADAPTIVE_LOOP.format(
                        end=end,
                        max_dt=phase.max_dt,
                        cutback_dt=phase.cutback_dt,
                        min_tol=phase.min_tol,
                    ),
                ]
            )
            # The source also leaves the second drainage recorders active.
            if stage != 3 or seq == 1:
                parts.extend(f"remove recorder $r_{index}\n" for index in range(1, 12))
    parts.append(
        'puts "No of extremely nonlinear steps:"\nputs $i_extr_nl\n'
        'set endT [clock seconds]\nputs "Execution time: [expr $endT-$startT] seconds."\n'
    )
    return "".join(parts)


def modern_newmark(script: str) -> str:
    """Translate the optional Rayleigh arguments removed from Newmark."""
    integrator = "integrator Newmark $gamma [expr pow($gamma+0.5, 2)/4]"
    legacy_gravity = integrator + " 0.00 0.0 0.00 0.0"
    legacy_shaking = re.escape(integrator) + (
        r"\s+\\\n\s*\$massProportionalDamping\s+0\.0\s+"
        r"\$InitStiffnessProportionalDamping\s+0\.0"
    )
    if script.count(legacy_gravity) != 1:
        raise ValueError("expected one legacy gravity integrator")
    script = script.replace(legacy_gravity, integrator)
    script, count = re.subn(
        legacy_shaking,
        "rayleigh $massProportionalDamping 0.0 $InitStiffnessProportionalDamping 0.0\n"
        + integrator,
        script,
    )
    if count != 1:
        raise ValueError("expected one legacy shaking integrator")
    return script


CAP_TIME = "0.15"
CAP_ATTEMPTS = 8


def cap_shaking(script: str) -> str:
    """Keep both full gravity stages and the first 0.15 s of adaptive shaking."""
    first_loop = "while {$currentTime < 13.} {"
    prefix, marker, tail = script.partition(first_loop)
    if not marker:
        raise ValueError("expected the first shaking loop")
    body, marker, _ = tail.partition("remove recorder $r_1")
    if not marker:
        raise ValueError("expected recorder removal after the first shaking loop")
    script = (
        prefix
        + f"set strut_attempts 0\nwhile {{$currentTime < {CAP_TIME}"
        + f" && $strut_attempts < {CAP_ATTEMPTS}}} {{\n    incr strut_attempts"
        + body
        + f'\nif {{$currentTime < {CAP_TIME}}} {{error "Shaking cap was not reached"}}\n'
        + 'puts "Centrifuge shaking time: [getTime]"\n'
    )
    for steps, expected in ((3, 1500000), (5, 4000000)):
        command = f"analyze {steps} 5.e5"
        if script.count(command) != 1:
            raise ValueError(f"expected one gravity command: {command}")
        script = script.replace(
            command,
            f'if {{[{command}] != 0}} {{error "Centrifuge gravity failed"}}\n'
            f'if {{[getTime] != {expected}}} {{error "Centrifuge gravity ended early"}}\n'
            'puts "Centrifuge gravity time: [getTime]"',
        )
    return script
