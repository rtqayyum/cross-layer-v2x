"""Minimal reader for PCL binary-compressed PCD point clouds."""

from __future__ import annotations

import struct

import numpy as np


def _lzf_decompress(source: bytes, expected_size: int) -> bytes:
    output = bytearray()
    index = 0
    while index < len(source):
        control = source[index]
        index += 1
        if control < 32:
            length = control + 1
            output.extend(source[index : index + length])
            index += length
            continue
        length = control >> 5
        reference = len(output) - ((control & 0x1F) << 8) - 1
        if length == 7:
            length += source[index]
            index += 1
        reference -= source[index]
        index += 1
        length += 2
        if reference < 0:
            raise ValueError("invalid LZF back-reference")
        for _ in range(length):
            output.append(output[reference])
            reference += 1
    if len(output) != expected_size:
        raise ValueError(
            f"LZF size mismatch: expected {expected_size}, decoded {len(output)}"
        )
    return bytes(output)


def read_binary_compressed_pcd(payload: bytes) -> dict[str, np.ndarray]:
    marker = b"DATA binary_compressed\n"
    marker_offset = payload.find(marker)
    if marker_offset < 0:
        raise ValueError("only DATA binary_compressed PCD files are supported")
    header_end = marker_offset + len(marker)
    header: dict[str, list[str]] = {}
    for raw_line in payload[:header_end].decode("ascii").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        header[parts[0].upper()] = parts[1:]

    fields = header["FIELDS"]
    sizes = [int(value) for value in header["SIZE"]]
    types = header["TYPE"]
    counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
    points = int(header["POINTS"][0])
    compressed_size, uncompressed_size = struct.unpack_from("<II", payload, header_end)
    compressed = payload[header_end + 8 : header_end + 8 + compressed_size]
    if len(compressed) != compressed_size:
        raise ValueError("truncated PCD compressed payload")
    decoded = _lzf_decompress(compressed, uncompressed_size)

    dtype_map = {
        ("F", 4): "<f4",
        ("F", 8): "<f8",
        ("U", 1): "u1",
        ("U", 2): "<u2",
        ("U", 4): "<u4",
        ("I", 1): "i1",
        ("I", 2): "<i2",
        ("I", 4): "<i4",
    }
    arrays: dict[str, np.ndarray] = {}
    offset = 0
    for field, size, kind, count in zip(fields, sizes, types, counts, strict=True):
        dtype = dtype_map.get((kind, size))
        if dtype is None:
            raise ValueError(f"unsupported PCD field type: {field} {kind}{size}")
        byte_count = points * size * count
        values = np.frombuffer(decoded, dtype=dtype, count=points * count, offset=offset)
        arrays[field] = values.reshape(points, count)[:, 0] if count > 1 else values
        offset += byte_count
    if offset != uncompressed_size:
        raise ValueError("PCD field layout does not match decoded size")
    return arrays

