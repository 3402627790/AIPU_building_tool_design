import subprocess
import shlex
import os
import sys

cur_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.abspath(os.path.join(cur_dir, "../.."))

_test_modules = set()


def get_modules():
    return _test_modules


def test_modules(options):
    print("num of modules", len(get_modules()))
    for m in list(get_modules()):
        m(options)


def changed(file_or_dir="", default=False):

    def dec(func):
        def f(*args, **kwargs):
            if f in _test_modules:
                _test_modules.remove(f)
            return func(*args, **kwargs)
        _test_modules.add(f)
        return f

    def empty(func):
        def f(*args, **kwargs):
            pass
        return f
    if len(file_or_dir) == 0:
        return dec

    if "-f" in sys.argv:    # test full test case
        return dec
    elif "--ignore-change" in sys.argv:
        if default:         # test force only
            return dec
        return empty        # ignore change in this repo
    # else: drive by git change

    files = []
    if isinstance(file_or_dir, str):
        files = [file_or_dir]
    elif isinstance(file_or_dir, list):
        files = file_or_dir
    files = [f.strip().strip("/") for f in files]
    cmd = "/arm/tools/git/git/2.17.1/rhe7-x86_64/bin/git diff --name-only HEAD HEAD~1"
    cmd = shlex.split(cmd)
    output = subprocess.check_output(cmd, cwd=project_dir, preexec_fn=os.setsid).decode("utf-8")
    for file in output.split("\n"):
        for target in files:
            if target in file:  # run the test
                return dec

    return empty  # nothing to do
