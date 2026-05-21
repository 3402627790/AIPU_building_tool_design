import argparse
import shutil
import struct
import numpy as np
import sys
import os
# PKG_DIR = os.path.join(os.path.dirname(__file__), "../../../../src")
# sys.path.insert(0, PKG_DIR)
from pathlib import Path
from AIPUBuilder._C._builder import *  # noqa


def compress(elf_files, output_file):
    elf_wts = []
    elf_tabs = []
    tmp_files = []

    for k, s in enumerate(elf_files):
        path = s.strip()
        print(f'Loading file: {path}...')
        assert os.path.exists(path)
        obj = AIPUELF()
        obj.load(path)
        wts = [obj.weight.numpy()]
        for wt in obj.extra_weight:
            wts.append(wt.numpy())
        obj.remove_aipu_note('weight')
        obj.remove_aipu_note('extra_weight_name')
        obj.extra_weight = []
        elf_wts.append(wts)
        elf_tabs.append(obj.constanthashtable)
        elf_file = f'aipuelf{k}_{Path(path).name}'
        tmp_files.append(elf_file)
        obj.write(elf_file)

    hash_map = dict()
    for wts, tab in zip(elf_wts, elf_tabs):
        for wt, items in zip(wts, tab):
            for it in items:
                key = str(it.hash_code)
                if key not in hash_map:
                    hash_map[key] = [wt, it.bin_offset, it.bin_size, 0]

    wt_file = 'merged_weight.bin'
    tmp_files.append(wt_file)
    offset = 0
    wts = []
    for k, v in hash_map.items():
        v[3] = offset
        offset += v[2]
        wts.append(v[0][v[1]: v[1] + v[2]])
    merged_wt = np.concatenate(wts, axis=0)
    merged_wt.tofile(wt_file)
    print(f'Merged weight size: {offset}')

    wt_offset_file = 'offset.bin'
    wt_size_file = 'size.bin'
    tmp_files.append(wt_offset_file)
    tmp_files.append(wt_size_file)
    wt_offsets = []
    wt_sizes = []
    for wts, tab in zip(elf_wts, elf_tabs):
        for wt, items in zip(wts, tab):
            wt_sizes.append(wt.size)
            for it in items:
                key = str(it.hash_code)
                assert key in hash_map
                wt_offsets.append(hash_map[key][3])
    np.array(wt_offsets, dtype=np.uint64).tofile(wt_offset_file)
    np.array(wt_sizes, dtype=np.uint64).tofile(wt_size_file)

    list_file = 'file_list.bin'
    head_file = 'file_head.bin'
    tmp_files.insert(0, list_file)
    tmp_files.insert(0, head_file)

    with open(list_file, "w") as fd:
        for path in tmp_files:
            fd.write(path + '\n')

    num_files = len(tmp_files)
    head_size = 4 + 16 * num_files
    with open(head_file, "wb") as fd:
        fd.write(struct.pack('I', num_files))
        offset = 0
        for k, path in enumerate(tmp_files):
            file_size = os.stat(path).st_size if k > 0 else head_size
            fd.write(struct.pack('Q', offset))
            fd.write(struct.pack('Q', file_size))
            offset += file_size

    print(f'Compressing files into {output_file}...')
    with open(output_file, "wb") as ofd:
        for path in tmp_files:
            with open(path, "rb") as ifd:
                shutil.copyfileobj(ifd, ofd)
    for path in tmp_files:
        os.remove(path)


def main():
    parser = argparse.ArgumentParser('AIPUELFZIP')
    parser.add_argument('elf_files',
                        type=str,
                        default='',
                        help='Path to aipu.elf files, e.g. "aipu0.elf,aipu1.elf"')
    parser.add_argument('-o', '--output',
                        type=str,
                        default='aipuelf.zip',
                        help='Path to output file')
    opt = parser.parse_args()
    compress(opt.elf_files.split(','), opt.output)


if __name__ == "__main__":
    main()
