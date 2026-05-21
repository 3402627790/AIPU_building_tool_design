import argparse
import struct
import numpy as np
import sys
import os
# PKG_DIR = os.path.join(os.path.dirname(__file__), "../../../../src")
# sys.path.insert(0, PKG_DIR)
from pathlib import Path
from AIPUBuilder._C._builder import *  # noqa


def decompress(zip_path, output_dir):
    elf_objs = []
    elf_wts = []
    elf_tabs = []
    elf_files = []

    print(f'Extracting files from {zip_path}...')
    assert os.path.exists(zip_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(zip_path, "rb") as fd:
        # read file head
        num_files = struct.unpack('I', fd.read(4))[0]
        file_info = []
        for _ in range(num_files):
            offset = struct.unpack('Q', fd.read(8))[0]
            file_size = struct.unpack('Q', fd.read(8))[0]
            file_info.append([offset, file_size])

        # read file list
        file_names = []
        fd.seek(file_info[1][0])
        for s in fd.read(file_info[1][1]).decode('utf-8').split('\n'):
            path = s.strip()
            if len(path) > 0:
                file_names.append(path)
        assert len(file_names) == len(file_info)

        # read aipuelf files
        for path, (offset, file_size) in zip(file_names, file_info):
            if path.startswith('aipuelf'):
                fd.seek(offset)
                obj = AIPUELF()
                obj.load_from_buffer(fd.read(file_size))
                elf_objs.append(obj)
                elf_tabs.append(obj.constanthashtable)
                elf_files.append(path)

        def read_file(file_name, dtype):
            for path, (offset, file_size) in zip(file_names, file_info):
                if path == file_name:
                    fd.seek(offset)
                    return np.frombuffer(fd.read(file_size), dtype=dtype)
            raise ValueError(f'File not found: {file_name}')

        merged_wt = read_file('merged_weight.bin', np.uint8)
        wt_offsets = read_file('offset.bin', np.uint64)
        wt_sizes = read_file('size.bin', np.uint64)

    offset_idx = 0
    size_idx = 0
    for tab in elf_tabs:
        wts = []
        for items in tab:
            wt = np.zeros((int(wt_sizes[size_idx]),), dtype=np.uint8)
            size_idx += 1
            for it in items:
                offset = int(wt_offsets[offset_idx])
                offset_idx += 1
                wt[it.bin_offset: it.bin_offset + it.bin_size] = \
                    merged_wt[offset: offset + it.bin_size]
            wts.append(BinSection(wt))
        elf_wts.append(wts)

    for path, obj, wts in zip(elf_files, elf_objs, elf_wts):
        obj.weight = wts[0]
        obj.extra_weight = wts[1:] if len(wts) > 1 else []
        elf_dir = output_dir
        if len(wts) > 1:
            elf_dir = os.path.join(output_dir, Path(path).stem)
            Path(elf_dir).mkdir(parents=False, exist_ok=True)
        print(f'Saving {path} to {os.path.abspath(elf_dir)}...')
        obj.write(os.path.join(elf_dir, path))


def main():
    parser = argparse.ArgumentParser('AIPUELFUNZIP')
    parser.add_argument('zip_path',
                        type=str,
                        default='',
                        help='Path to aipuelf.zip file')
    parser.add_argument('-o', '--output_dir',
                        type=str,
                        default='./',
                        help='Path to output dir')
    opt = parser.parse_args()
    decompress(opt.zip_path, opt.output_dir)


if __name__ == "__main__":
    main()
