"""Generate the Model1 benchmark mesh without reading the OpenSees fixture."""

from __future__ import annotations

import json
import lzma
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from itertools import accumulate
from math import sqrt
from pathlib import Path

PARAMS = Path(__file__).with_name("params.json")

# Iteration order follows the geometric commands in each source file.
FAMILIES = (
    ("opensees.tcl", "dispBeamColumn"),
    ("Hbeam.tcl", "dispBeamColumn"),
    ("layershell.tcl", "ShellMITC4"),
    ("truss.tcl", "elasticBeamColumn"),
    ("truss.tcl", "truss"),
)
ARITY = {"dispBeamColumn": 2, "ShellMITC4": 4, "elasticBeamColumn": 2, "truss": 2}

Point = tuple[float, float, float]
# Plane and plan slot. The low three slot bits select reflections/swap.
Location = tuple[int, int]
Family = tuple[str, str]
Element = tuple[str, tuple[int, ...]]


@dataclass
class Geometry:
    nodes: dict[int, Point]
    elements: dict[int, Element]
    diaphragms: dict[int, frozenset[int]]
    element_order: dict[Family, tuple[int, ...]]


def floor_elevations(params: dict) -> list[float]:
    """Return base plus diaphragm elevations, in metres."""
    increments = (
        height for height, count in params["story_heights_dm"] for _ in range(count)
    )
    return [z / 10 for z in accumulate(increments, initial=0)]


def _reflection_mask(x: float, y: float) -> int:
    if x == y:
        return 15
    if x == 0:
        return 53
    return 255


def _linear(z, z0, z1, a, b):
    return a + (b - a) * (z - z0) / (z1 - z0)


def _tracks(groups):
    for first, last, rows in groups:
        for shape, *symmetry in rows:
            yield first, last, shape, symmetry


def _track_positions(tracks, elevations):
    @cache
    def position(track, floor):
        first, last, shape, _ = tracks[track]
        method, *args = shape
        if not isinstance(method, str):
            return float(method), float(args[0])
        if method == "arc":
            radius, x = args
            if x == "diag":
                x = radius / sqrt(2)
                return x, x
            return x, sqrt(radius**2 - x**2)
        if method == "chord":
            x, left, right = args
            a, b = position(left, floor)
            c, d = position(right, floor)
            return x, _linear(x, a, c, b, d)
        if method == "lean":
            x, y0, y1 = args
            return x, _linear(
                elevations[floor], elevations[first], elevations[last], y0, y1
            )
        if method == "diagonal":
            a, b = args
            x = _linear(elevations[floor], elevations[first], elevations[last], a, b)
            return x, x
        if method == "line":
            x0, y0, x1, y1 = args
            return (
                _linear(elevations[floor], elevations[first], elevations[last], x0, x1),
                _linear(elevations[floor], elevations[first], elevations[last], y0, y1),
            )
        if method == "polyline":
            x, y, *knots = args
            start = first
            for stop, a, b in knots:
                if floor <= stop:
                    return (
                        _linear(
                            elevations[floor], elevations[start], elevations[stop], x, a
                        ),
                        _linear(
                            elevations[floor], elevations[start], elevations[stop], y, b
                        ),
                    )
                start, x, y = stop, a, b
        raise ValueError(f"Invalid track shape {shape!r} at floor {floor}")

    return position


def _locations(params: dict) -> dict[Location, Point]:
    floors = floor_elevations(params)
    points = {}
    for name, parity in (("floor_tracks", 0), ("wall_head_tracks", 1)):
        tracks = list(_tracks(params[name]))
        at = (
            floors
            if parity == 0
            else [round(z - params["wall_head_offset"], 4) for z in floors[1:]]
        )

        position = _track_positions(tracks, at)
        for track, (first, last, _, symmetry) in enumerate(tracks):
            overrides = {}
            if len(symmetry) == 2:
                for start, stop, mask in symmetry[1]:
                    overrides.update(dict.fromkeys(range(start, stop + 1), mask))
            for floor in range(first, last + 1):
                x, y = position(track, floor)
                mask = overrides.get(
                    floor, symmetry[0] if symmetry else _reflection_mask(x, y)
                )
                for bit in range(8):
                    if mask & (1 << bit):
                        a, b = (y, x) if bit & 4 else (x, y)
                        a = -a if bit & 1 else a
                        b = -b if bit & 2 else b
                        plane = 2 * floor + parity
                        points[plane, track * 8 + bit] = (a or 0.0, b or 0.0, at[floor])
    return points


class _Reader:
    """Read byte arrays and unsigned or zigzag base-128 integers."""

    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def bytes(self, size):
        end = self.offset + size
        out = self.data[self.offset : end]
        if len(out) != size:
            raise ValueError("Truncated geometry table")
        self.offset = end
        return out

    def uint(self):
        value = 0
        shift = 0
        while True:
            byte = self.data[self.offset]
            self.offset += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value
            shift += 7

    def ints(self, size):
        return [self.uint() for _ in range(size)]

    def signed(self, size):
        return [(i >> 1) ^ -(i & 1) for i in self.ints(size)]

    def deltas(self, size):
        return list(accumulate(self.signed(size)))

    def require_end(self):
        if self.offset != len(self.data):
            raise ValueError("Unexpected trailing geometry data")


def _decode_packet(data: bytes) -> bytes:
    return lzma.decompress(
        data,
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA2, "dict_size": 1 << 20}],
    )


def _element_stencils(reader, family):
    """Expand relative-plane stencils repeated in element tag order."""
    _, element_type = family
    width = ARITY[element_type]
    count, stencils_count, repeat_count, shift_count, ordered = reader.ints(5)
    tags = reader.deltas(count)
    base_planes = reader.deltas(stencils_count)
    shifts = iter(reader.signed(shift_count))
    repeats = [reader.ints(4) for _ in range(repeat_count)]
    transforms = iter(reader.bytes(sum(copies - 1 for _, _, copies, _ in repeats)))
    offsets = reader.ints(stencils_count)
    tracks = iter(reader.signed(stencils_count * width))
    reflections = iter(reader.bytes(stencils_count * (width // 2)))

    stencils = []
    previous = {}
    first_track = 0
    for packed_offsets in offsets:
        symmetries = []
        for _ in range(width // 2):
            packed = next(reflections)
            first = packed % 8
            symmetries.extend((first, first ^ (packed // 8)))
        first_track += next(tracks)
        ds = [next(tracks) for _ in range(width - 1)]
        # Reuse the other endpoints of the last stencil with this first track and span.
        context = first_track, packed_offsets
        guess = previous.get(context, (first_track,) * (width - 1))
        track_numbers = (first_track, *(v + old for v, old in zip(ds, guess)))
        previous[context] = track_numbers[1:]
        nodes = tuple(
            ((packed_offsets >> (3 * i)) & 7, 8 * track + symmetries[i])
            for i, track in enumerate(track_numbers)
        )
        stencils.append(nodes)

    layout = []
    start = 0
    end = 0
    for skip, width, copies, uniform_shift in repeats:
        start = end + skip
        end = start + width
        layout.extend(
            zip(base_planes[start - skip : end], stencils[start - skip : end])
        )
        for _ in range(copies - 1):
            transform = next(transforms)
            shift = next(shifts) if uniform_shift else 0
            for plane, nodes in zip(base_planes[start:end], stencils[start:end]):
                base = plane + shift if uniform_shift else next(shifts)
                layout.append(
                    (base, tuple((offset, slot ^ transform) for offset, slot in nodes))
                )
    layout.extend(zip(base_planes[end:], stencils[end:]))
    if len(layout) != count:
        raise ValueError(f"Invalid stencil expansion for {family!r}")

    elements = {
        tag: tuple((base + offset, slot) for offset, slot in slots)
        for tag, (base, slots) in zip(tags, layout)
    }
    order = (
        tuple(tags[index] for index in reader.deltas(count)) if ordered else tuple(tags)
    )
    reader.require_end()
    return elements, order


def _node_numbering(reader, points, elements, count):
    """Decode the legacy node permutation against generated geometry."""
    modes = reader.bytes(count)
    neighbor_ranks = iter(reader.ints(modes.count(0)))
    track_ranks = iter(reader.ints(modes.count(1)))
    explicit = modes.count(2)
    plane_steps = iter(reader.signed(explicit))
    plane_parities = iter(reader.bytes(explicit))
    slots = reader.bytes(explicit)
    wide_slots = set(reader.ints(reader.uint()))
    reader.require_end()

    adjacency = defaultdict(set)
    for family in elements.values():
        for stencil in family.values():
            for key in stencil:
                adjacency[key].update(other for other in stencil if other != key)
    planes = defaultdict(list)
    tracks = defaultdict(list)
    for key in sorted(points):
        plane, slot = key
        planes[plane].append(key)
        tracks[slot].append(key)

    assigned = {}
    prev_plane = prev_slot = explicit_index = 0
    previous = None
    for number, mode in enumerate(modes, 1):
        if mode == 0:
            nearby = sorted(
                (key for key in adjacency[previous] if key not in assigned),
                key=lambda key: (
                    abs(key[0] - prev_plane),
                    abs(key[1] - prev_slot),
                    key,
                ),
            )
            key = nearby[next(neighbor_ranks)]
        elif mode == 1:
            nearby = sorted(
                tracks[prev_slot],
                key=lambda key: (abs(key[0] - prev_plane), key),
            )
            key = nearby[next(track_ranks)]
        elif mode == 2:
            plane = 2 * (prev_plane // 2 + next(plane_steps)) + next(plane_parities)
            rank = slots[explicit_index] + 256 * (explicit_index in wide_slots)
            explicit_index += 1
            key = planes[plane][rank]
        else:
            raise ValueError(f"Unknown node numbering mode {mode}")
        prev_plane, prev_slot = key
        planes[prev_plane].remove(key)
        tracks[prev_slot].remove(key)
        assigned[key] = number
        previous = key

    if len(assigned) != len(points):
        raise ValueError("Node numbering and parametric layout differ")
    return assigned


def generate(params_path: Path = PARAMS) -> Geometry:
    """Return original IDs, ordered element connectivity, and diaphragm sets."""
    params_path = Path(params_path)
    params = json.loads(params_path.read_text())
    if params["version"] != 1:
        raise ValueError("Unsupported megatall geometry table version")
    data = params_path.with_name(params["legacy_table"]).read_bytes()
    packets = []
    start = 0
    for size in params["packet_bytes"]:
        packets.append(_decode_packet(data[start : start + size]))
        start += size
    if start != len(data):
        raise ValueError("Geometry table size disagrees with the parameter file")

    # Supplemental tracks share the main track schema. Reinsert them at their
    # slot indices before expanding the stencils.
    for index, group in json.loads(packets.pop()):
        params["floor_tracks"].insert(index, group)
    points = _locations(params)

    element_locations = {}
    order = {}
    for family, packet in zip(FAMILIES, packets[:-1], strict=True):
        element_locations[family], order[family] = _element_stencils(
            _Reader(packet), family
        )
    number = _node_numbering(
        _Reader(packets[-1]), points, element_locations, len(points)
    )
    nodes = {tag: points[location] for location, tag in number.items()}
    elements = {
        tag: (family[1], tuple(number[key] for key in element_locations[family][tag]))
        for family, tags in order.items()
        for tag in tags
    }

    diaphragm_rows = []
    for first, last, masks in params["diaphragms"]:
        for floor in range(first, last + 1):
            tags = {
                number[floor * 2, track * 8 + bit]
                for track, mask in masks
                for bit in range(8)
                if mask & (1 << bit)
            }
            roof = floor >= params["roof_diaphragm_start"]
            master = min(
                tag
                for tag in tags
                if max(map(abs, nodes[tag][:2])) > 15
                or (roof and nodes[tag][:2] == (5, 15))
            )
            diaphragm_rows.append((roof, master, frozenset(tags - {master})))

    diaphragms = {master: slaves for _, master, slaves in sorted(diaphragm_rows)}
    return Geometry(nodes, elements, diaphragms, order)
