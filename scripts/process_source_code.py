# The Compass SDK ("Software") may only be used by a person authorized under and to the extent
# permitted by a subsisting licensing agreement from Arm Technology (China) Co., Ltd. or its affiliates.
# Copyright (C) [2023] Arm Technology (China) Co., Ltd.

import argparse
import os
import re
import time


_TARGET_SOURCE_CODE_SUFFIXES = [
    '.s',
    '.c',
    '.cl',
    '.cc',
    '.cpp',
    '.h',
    '.hpp',
    '.py',
    '.pyi',
    '.sh',
    '.txt',
]


_EXCLUDED_REPO_SUBDIRS = [
    '3rdparty',
    'build',
]


_LICENSE_HEADER_TEMPLATE = \
"""// The Compass SDK ("Software") may only be used by a person authorized under and to the extent
// permitted by a subsisting licensing agreement from Arm Technology (China) Co., Ltd. or its affiliates.
// Copyright (C) [2023{}] Arm Technology (China) Co., Ltd.
"""


_LICENSE_EXCLUDED_FILES = [
    'include/core/fp16/half.h',
    'tools/aipuperfinfo/json.hpp',
    'python/Parser/tf_frozen.py',
]


_AUTHOR_INFO_PATTERN = r'\/\*.*?@Author.*?@Date.*?\*\/'


_OLD_LICENSE_HEADER_PATTERN = r'(\/\/-|\/\*).*?(Copyright|Author).*?Arm China.*?(-|\*\/)\n'


_EMPTY_DESCRIPTION_PATTERN = r'"""\n"""\n'


_NEW_LICENSE_HEADER_PATTERN = r'The Compass SDK.*?Arm Technology \(China\) Co., Ltd.\n'


_INTERNAL_PATH_PATTERN = r'(\/project\/|\/home\/).*?(\'|"| |;|\)|\n)'


def main():
    parser = argparse.ArgumentParser('SourceCodeProcessor')
    parser.add_argument('-r', '--repo_path',
                        type=str,
                        default='',
                        help='Path to AIPU_building_tool_design repo directory')
    opt = parser.parse_args()

    if not opt.repo_path:
        opt.repo_path = os.path.abspath('..')
    print(f'[I] Start to process folder: {opt.repo_path}...')

    source_code_files = []
    source_code_suffixes = []
    for root, dirs, files in os.walk(opt.repo_path):
        is_excluded = False
        for subdir in _EXCLUDED_REPO_SUBDIRS:
            if os.path.join('AIPU_building_tool_design', subdir) in root:
                is_excluded = True
                break
        if is_excluded:
            continue
        for file_name in files:
            file_path = os.path.join(root, file_name)
            for suffix in _TARGET_SOURCE_CODE_SUFFIXES:
                if file_name.lower().endswith(suffix):
                    source_code_files.append(file_path)
                    source_code_suffixes.append(suffix)
                    break

    current_year = time.strftime("%Y", time.localtime())
    for file_path, file_suffix in zip(source_code_files, source_code_suffixes):
        file_content = ''
        is_modified = False
        with open(file_path, 'r') as fd:
            file_content = fd.read()
            # remove old license header
            matchs = list(re.finditer(_OLD_LICENSE_HEADER_PATTERN, file_content, re.DOTALL))
            if matchs:
                if len(matchs) > 1:
                    print(f'[W] Found more than one license statement in file: {file_path}')
                for match in matchs:
                    matched_text = match.group(0)
                    file_content = file_content.replace(matched_text, '')
                is_modified = True

            # remove empty descrition
            matchs = list(re.finditer(_EMPTY_DESCRIPTION_PATTERN, file_content, re.DOTALL))
            if matchs:
                for match in matchs:
                    matched_text = match.group(0)
                    file_content = file_content.replace(matched_text, '')
                is_modified = True

            # remove author information
            matchs = list(re.finditer(_AUTHOR_INFO_PATTERN, file_content, re.DOTALL))
            if matchs:
                if len(matchs) > 1:
                    print(f'[W] Found more than one author statement in file: {file_path}')
                for match in matchs:
                    matched_text = match.group(0)
                    file_content = file_content.replace(matched_text, '')
                is_modified = True

            # process internal path
            matchs = list(re.finditer(_INTERNAL_PATH_PATTERN, file_content))
            if matchs:
                for match in matchs:
                    matched_text = match.group(0)[:-1]
                    print(file_path)
                    print(matched_text + '\n')

            # insert new license header
            is_excluded = False
            for excluded_file in _LICENSE_EXCLUDED_FILES:
                if file_path.endswith(excluded_file):
                    is_excluded = True
                    break
            if not is_excluded:
                matchs = list(re.finditer(_NEW_LICENSE_HEADER_PATTERN, file_content, re.DOTALL))
                if not matchs:
                    if current_year > '2023':
                        license_header = _LICENSE_HEADER_TEMPLATE.format('-' + current_year)
                    else:
                        license_header = _LICENSE_HEADER_TEMPLATE.format('')
                    if file_suffix in ['.s']:
                        license_header = license_header.replace('//', ';')
                    elif file_suffix in ['.py', '.pyi', '.sh', '.txt']:
                        license_header = license_header.replace('//', '#')
                    file_content = license_header + '\n' + file_content
                    is_modified = True

        if is_modified:
            with open(file_path, 'w') as fd:
                fd.write(file_content)


if __name__ == '__main__':
    main()
