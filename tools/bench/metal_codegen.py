"""Extract native Apple GPU code and raw compiler metadata from a Metal archive.

This is an archive inspector, not an ISA disassembler. It deliberately does not
interpret instruction bytes with a decoder for a different Apple GPU generation.
Mach-O structures follow the platform headers. The optional metadata field hints
are experimental; see docs/research/m5-native-code.md for their provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


def unpack(data, fmt, offset):
    if offset < 0 or offset + struct.calcsize(fmt) > len(data):
        raise ValueError(f'truncated binary at {offset}')
    return struct.unpack_from(fmt, data, offset)


def region(data, offset, size):
    if offset < 0 or size < 0 or offset + size > len(data):
        raise ValueError(f'invalid binary region {offset}+{size}')
    return data[offset:offset + size]


def macho(data):
    magic, cpu, subtype, _, count, command_bytes = unpack(data, '<6I', 0)
    if magic != 0xFEEDFACF or cpu != 0x1000013:
        raise ValueError('expected a little-endian 64-bit Apple GPU Mach-O')
    end = 32 + command_bytes
    region(data, 32, command_bytes)
    offset, sections, symbols = 32, {}, {}
    for _ in range(count):
        command, size = unpack(data, '<II', offset)
        if size < 8 or offset + size > end:
            raise ValueError('invalid Mach-O load command')
        if command == 0x19:  # LC_SEGMENT_64 / section_64
            nsections, = unpack(data, '<I', offset + 64)
            if 72 + 80 * nsections > size:
                raise ValueError('truncated Mach-O section table')
            for i in range(nsections):
                p = offset + 72 + 80 * i
                name, segment, address, length, file_offset = unpack(data, '<16s16sQQI', p)
                key = (segment.rstrip(b'\0').decode(), name.rstrip(b'\0').decode())
                sections[key] = (address, region(data, file_offset, length))
        elif command == 2:  # LC_SYMTAB / nlist_64
            symoff, nsyms, stroff, strsize = unpack(data, '<4I', offset + 8)
            strings = region(data, stroff, strsize)
            for i in range(nsyms):
                nameoff, _, _, _, value = unpack(data, '<IBBHQ', symoff + i * 16)
                if nameoff >= len(strings):
                    raise ValueError('invalid Mach-O string offset')
                name = strings[nameoff:].split(b'\0', 1)[0].decode()
                symbols[name] = value
        offset += size
    return subtype, sections, symbols


def gpu_image(data):
    if data[:4] == bytes.fromhex('cbfebabe'):
        count, = unpack(data, '>I', 4)
        for i in range(count):
            cpu, _, offset, size, _ = unpack(data, '>5I', 8 + 20 * i)
            if cpu == 0x1000013:
                return gpu_image(region(data, offset, size))
        raise ValueError('archive contains no Apple GPU image')
    _, sections, _ = macho(data)
    if ('__GPU_METADATA', '__compute') in sections:
        return data
    if ('__TEXT', '__compute') in sections:
        return gpu_image(sections['__TEXT', '__compute'][1])
    raise ValueError('archive contains no compute executable')


def metadata_fields(data):
    """Return raw field bytes from the stats table in the metadata FlatBuffer.

    Field widths and meanings are not a public Metal API. Preserve bytes instead
    of treating the neighboring fields of a one-byte flag as a uint32 statistic.
    """
    def table(position):
        distance, = unpack(data, '<i', position)
        vtable = position - distance
        length, object_size = unpack(data, '<HH', vtable)
        if length < 4 or length % 2:
            raise ValueError('invalid metadata vtable')
        offsets = unpack(data, '<' + 'H' * ((length - 4) // 2), vtable + 4)
        present = {i: v for i, v in enumerate(offsets) if v}
        boundaries = sorted(set(present.values()) | {object_size})
        return {i: (position + v, next(b for b in boundaries if b > v) - v)
                for i, v in present.items()}
    root, = unpack(data, '<I', 0)
    pointer, _ = table(root)[0]
    relative, = unpack(data, '<I', pointer)
    fields = table(pointer + relative)
    return {str(i): region(data, offset, size).hex() for i, (offset, size) in fields.items()}


def inspect_archive(path, output):
    gpu = gpu_image(path.read_bytes())
    subtype, sections, symbols = macho(gpu)
    address, text = sections['__TEXT', '__text']
    start = symbols['_agc.main'] - address
    end = min([v - address for v in symbols.values() if v > symbols['_agc.main']] + [len(text)])
    code = region(text, start, end - start)
    fields = metadata_fields(sections['__GPU_METADATA', '__compute'][1])
    output.mkdir(parents=True, exist_ok=True)
    (output / 'compute.gpubin').write_bytes(gpu)
    (output / 'main.bin').write_bytes(code)
    (output / 'metadata.bin').write_bytes(sections['__GPU_METADATA', '__compute'][1])
    result = dict(archive=str(path.resolve()), cpu_subtype=subtype, main_bytes=len(code),
                  main_sha256=hashlib.sha256(code).hexdigest(), raw_metadata_fields=fields,
                  statistic_hints={name: int.from_bytes(bytes.fromhex(fields.get(str(i), '00')), 'little')
                                   for name, i in [('registers_field_0', 0), ('scratch_field_14', 14),
                                                   ('scratch_field_41', 41), ('threadgroup_field_28', 28)]})
    (output / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect_archive(args.archive, args.out)))


if __name__ == '__main__':
    main()
