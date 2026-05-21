import warnings
import time
import threading
import functools
from AIPUBuilder.logger import WARN, DEBUG
import sys


def is_elf_file(fd):
    if isinstance(fd, str):
        fd = open(fd, 'rb')
    magic = b"\x7fELF"
    pos = fd.tell()
    fd.seek(0)
    data = fd.read(4)
    fd.seek(pos)
    if data == magic:
        return True
    return False


def Deprecated(*args):
    import traceback
    msg = ""
    if len(args) == 1 and not callable(args[0]):
        msg = args[0]

    def decorator(func):
        def ret(*args, **kwargs):
            stacks = traceback.format_stack()
            message = "[Deprecated] function: %s is deprecated. " % (
                func.__name__) + msg + "\n" + stacks[-2]
            warnings.warn(
                message,
                DeprecationWarning
            )
            WARN(message)
            return func(*args, **kwargs)
        return ret
    if callable(args[0]):
        return decorator(args[0])
    return decorator


def timeit(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        ret = func(*args, **kwargs)
        DEBUG("[timeit] %s costs %f msec." %
              (func.__name__, (time.time() - t0) * 1000.))
        return ret
    return wrapper


def _monitoring(t: threading.Thread, time_step=1, stop=None):
    def get_current_frameinfo(f):
        code = f.f_code
        return f"{code.co_filename}:{f.f_lineno}"
    # return a list of call stack,with filename and lino or function

    def get_call_stack(f):
        ret = []
        while f is not None and f.f_back is not None:
            ret.append(get_current_frameinfo(f))
            f = f.f_back
        return ret

    f = sys._current_frames().get(t.ident, None)
    last_call_stack = get_call_stack(f)
    if stop is None:
        stop = [False]
    while t.is_alive() and not stop[0]:
        time.sleep(time_step)
        f = sys._current_frames().get(t.ident, None)
        current_call_stack = get_call_stack(f)
        if len(current_call_stack) == len(last_call_stack):
            # we not compare current line
            info = True
            for i in range(1, len(current_call_stack)):
                if current_call_stack[i] != last_call_stack[i]:
                    info = False
                    break
            if info and len(current_call_stack) > 0:
                print(f"[runing ...] {current_call_stack[0]}")
        last_call_stack = current_call_stack
    # print("monitor done.....")


def enable_monitoring(time_step=1):
    t = threading.Thread(target=_monitoring, args=(
        threading.current_thread(), time_step))
    t.start()
    return t


class MonitorStatus(object):
    def __init__(self, time_step=1):
        self.stop = [False]
        self.monitor = threading.Thread(target=_monitoring, args=(
            threading.current_thread(), time_step, self.stop))

    def __enter__(self):
        self.monitor.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop[0] = True
        while self.monitor.is_alive():
            time.sleep(0.05)
        # print("exit monitor status")


INTRINSIC_SPECIAL_ATTRIBUTES = frozenset(
    (
        "__qualname__",
        "__module__",
        "__metaclass__",
        "__dict__",
        "__weakref__",
        "__class__",
        "__subclasshook__",
        "__name__",
        "__doc__",
        '__members__',
    )
)


class RaiseCRuntimeError(object):
    """
    Raise a C runtime error in Python.
    """

    def __init__(self):
        self.messages = []

        def raise_error(msg):
            print(msg)
            self.messages.append(msg)
        self.cb = raise_error
        self.cb_name = "_RaiseCRuntimeError"

    def __enter__(self):
        from AIPUBuilder._C._core import add_error_callback
        add_error_callback(self.cb_name, self.cb)

    def __exit__(self, exc_type, exc_value, traceback):
        from AIPUBuilder._C._core import remove_error_callback
        remove_error_callback(self.cb_name)
        if len(self.messages):
            raise RuntimeError("\n".join(self.messages))


class DisableLogger(object):
    """
    Disable logger temporarily.
    """

    def __init__(self):
        self.backup_level = None

    def __enter__(self):
        from AIPUBuilder.logger import get_logger_level, set_logger_level
        self.backup_level = get_logger_level()
        set_logger_level(8)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        from AIPUBuilder.logger import get_logger_level, set_logger_level
        set_logger_level(self.backup_level)

# inside from lsst.utils


def continue_class(cls):

    orig = getattr(sys.modules[cls.__module__], cls.__name__)
    for name in dir(cls):
        attr = cls.__dict__.get(name, None) or getattr(cls, name)
        if not (name.startswith("__") and (attr is getattr(object, name, None) or name in INTRINSIC_SPECIAL_ATTRIBUTES)):
            setattr(orig, name, attr)
    return orig


def wrap_class_method(cls, methods, wrapper):
    orig = getattr(sys.modules[cls.__module__], cls.__name__)
    for name in methods:
        attr = cls.__dict__.get(name, None) or getattr(cls, name, None)
        # print(name,attr)
        if not (name.startswith("__") and (attr is getattr(object, name, None) or name in INTRINSIC_SPECIAL_ATTRIBUTES)):
            setattr(orig, name, wrapper(attr))
        # attr = cls.__dict__.get(name, None) or getattr(cls, name,None)
        # print(name,attr)


def raise_rt_error(cls):
    class Wrapped(cls):  # 继承导入的类
        @functools.wraps(cls.__init__)
        def __init__(self, *args, **kwargs):
            with RaiseCRuntimeError():
                super().__init__(*args, **kwargs)

        def __getattribute__(self, name):
            attr = super().__getattribute__(name)
            if callable(attr) and (not name.startswith("__")):  # 排除特殊方法
                def wrapper(*args, **kwargs):
                    with RaiseCRuntimeError():
                        return attr(*args, **kwargs)
                return wrapper
            return attr
    Wrapped.__name__ = cls.__name__
    Wrapped.__doc__ = cls.__doc__
    Wrapped.__module__ = cls.__module__
    return Wrapped
