import os
import sys
import re
from pathlib import Path

_EXCLUDED_REPO_SUBDIRS = [
    '3rdparty',
    'build',
]


def remove_comments(content):
    """
    移除C++代码中的注释（保留字符串内容）
    :param content: 源代码字符串
    :return: 无注释的代码
    """
    output = []          # 输出字符列表
    in_block_comment = False  # 是否在块注释中
    in_line_comment = False   # 是否在单行注释中
    in_string = False         # 是否在字符串中（支持单/双引号）
    quote_char = None         # 当前字符串的引号类型（' 或 "）

    i = 0
    while i < len(content):
        # 处理字符串（优先于注释）
        if not in_block_comment and not in_line_comment:
            if content[i] in ('"', "'"):
                if in_string and content[i] == quote_char:
                    # 字符串结束（忽略转义符情况）
                    if i > 0 and content[i-1] != '\\':
                        in_string = False
                        quote_char = None
                elif not in_string:
                    in_string = True
                    quote_char = content[i]
                output.append(content[i])
                i += 1
                continue

        # 处理注释（仅在非字符串状态下）
        if not in_string:
            # 检测块注释开始：/*
            if i + 1 < len(content) and content[i] == '/' and content[i+1] == '*':
                in_block_comment = True
                i += 2  # 跳过两个字符
                continue
            # 检测块注释结束：*/
            if in_block_comment and i + 1 < len(content) and content[i] == '*' and content[i+1] == '/':
                in_block_comment = False
                i += 2
                continue
            # 检测单行注释：//
            if i + 1 < len(content) and content[i] == '/' and content[i+1] == '/':
                in_line_comment = True
                i += 2
                continue
            # 单行注释遇到换行符结束
            if in_line_comment and content[i] == '\n':
                in_line_comment = False

        # 非注释内容且不在字符串中时，直接跳过
        if in_block_comment or in_line_comment:
            i += 1
            continue

        # 非注释内容，添加到输出
        output.append(content[i])
        i += 1

    return ''.join(output)


def process_file(file_path):
    """处理单个文件，生成备份并移除注释"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 备份原始文件（保留权限和时间戳）
        backup_path = file_path + '.bak'
        if not os.path.exists(backup_path):
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(content)

        # 移除注释并写回原文件
        cleaned_content = remove_comments(content)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)

        print(f"处理成功: {file_path}")
    except Exception as e:
        print(f"处理失败: {file_path} - {str(e)}")


def process_directory(directory):
    """递归处理目录下所有C++文件"""
    path = Path(directory)
    for file_path in path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in {'.cpp', '.h', '.hpp', '.cc', '.hh'} and not any(subdir in str(file_path) for subdir in _EXCLUDED_REPO_SUBDIRS):
            process_file(str(file_path))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python remove_cpp_comments.py <目录或文件路径>")
        sys.exit(1)

    target = sys.argv[1]
    if os.path.isfile(target):
        process_file(target)
    elif os.path.isdir(target):
        process_directory(target)
    else:
        print(f"错误: 路径不存在 {target}")
