"""Inspect or export Strut's little-endian sparse solver replay capture."""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from struct import unpack_from

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Pattern:
    columns: NDArray[np.int64]
    rows: NDArray[np.int64]


@dataclass(frozen=True)
class Tangent:
    values: NDArray[np.float64]


@dataclass(frozen=True)
class Solve:
    rhs: NDArray[np.float64]
    solution: NDArray[np.float64]


def read_trace(path: Path) -> Iterator[Pattern | Tangent | Solve]:
    data = np.memmap(path, dtype=np.uint8, mode="r")
    offset = n = nnz = 0
    have_tangent = False

    def take(dtype: str, count: int) -> NDArray:
        nonlocal offset
        if count < 0 or count > (len(data) - offset) // 8:
            raise ValueError(f"{path}: truncated trace at byte {offset}")
        values = np.frombuffer(data, dtype=dtype, count=count, offset=offset)
        offset += 8 * count
        return values

    while offset < len(data):
        if len(data) - offset < 8:
            raise ValueError(f"{path}: incomplete record tag at byte {offset}")
        (kind,) = unpack_from("<Q", data, offset)
        offset += 8
        if kind == 1:
            n, nnz = map(int, take("<u8", 2))
            if n <= 0 or nnz < n:
                raise ValueError(f"{path}: invalid CSC dimensions")
            columns = take("<i8", n + 1)
            rows = take("<i8", nnz)
            if columns[0] != 0 or columns[-1] != nnz or np.any(np.diff(columns) < 1):
                raise ValueError(f"{path}: invalid CSC column pointers")
            have_tangent = False
            yield Pattern(columns, rows)
        elif kind == 2 and n:
            have_tangent = True
            yield Tangent(take("<f8", nnz))
        elif kind == 3 and have_tangent:
            yield Solve(take("<f8", n), take("<f8", n))
        else:
            raise ValueError(
                f"{path}: unexpected record kind {kind} at byte {offset - 8}"
            )


def write_matrix_market(path: Path, pattern: Pattern, tangent: Tangent) -> None:
    columns = np.repeat(np.arange(1, len(pattern.columns)), np.diff(pattern.columns))
    entries = np.empty(
        len(columns), dtype=[("row", "<i8"), ("col", "<i8"), ("value", "<f8")]
    )
    entries["row"] = pattern.rows + 1
    entries["col"] = columns
    entries["value"] = tangent.values
    n = len(pattern.columns) - 1
    with path.open("x") as output:
        output.write(
            f"%%MatrixMarket matrix coordinate real symmetric\n{n} {n} {len(entries)}\n"
        )
        np.savetxt(output, entries, fmt=("%d", "%d", "%.17g"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--matrix", type=int, default=1, help="1-based tangent index for export"
    )
    parser.add_argument(
        "--matrix-market",
        type=Path,
        help="export selected tangent as symmetric CSC entries",
    )
    args = parser.parse_args()
    patterns = tangents = solves = 0
    for record in read_trace(args.trace):
        if isinstance(record, Pattern):
            patterns += 1
            pattern = record
            print(
                f"Pattern {patterns}: {len(record.columns) - 1} DOFs, {len(record.rows)} lower entries"
            )
        elif isinstance(record, Tangent):
            tangents += 1
            if args.matrix_market and tangents == args.matrix:
                write_matrix_market(args.matrix_market, pattern, record)
        else:
            solves += 1
    print(f"{patterns} patterns, {tangents} tangents, {solves} right-hand sides")
    if args.matrix_market and not 1 <= args.matrix <= tangents:
        parser.error(f"--matrix must be between 1 and {tangents}")


if __name__ == "__main__":
    main()
