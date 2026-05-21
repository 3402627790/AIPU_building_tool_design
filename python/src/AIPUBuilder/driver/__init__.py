import os
import fcntl
from AIPUBuilder._C._driver import *
from ._driver import AIPUJob, AIPUModel  # override the original AIPUJob and AIPUModel from _C._driver
__VERSION__ = "0.1"
__build_number__ = None
if __build_number__ is not None:
    __VERSION__ = __VERSION__+"."+str(__build_number__)


GET_RUNNING_JOB_THREAD_ID = 0x80804119
AIPU_IOCTL_GET_RUNNING_JOB_THREAD_ID = 0x80804119
AIPU_IOCTL_QUERY_CAP = 0x81a84100
AIPU_IOCTL_QUERY_PARTITION_CAP = 0x80804101
AIPU_IOCTL_GET_DRIVER_VERSION = 0x80084112
AIPU_IOCTL_ENABLE_TICK_COUNTER = 0x410d
AIPU_IOCTL_DISABLE_TICK_COUNTER = 0x410c


def is_run_on_aipu():
    return is_running_on_aipu()


def is_running_on_aipu():
    '''
    Check if the code is running on an AIPU device.

    Returns:
       bool: True if running on an AIPU device, False otherwise.
    '''
    return os.path.exists("/dev/aipu")


def get_cur_freq():
    '''
    Get the current frequency of AIPU.

    Returns:
        int: The current frequency in Hz, or -1 if the frequency cannot be determined.
    '''
    with os.popen("cat /sys/class/devfreq/*.aipu/cur_freq") as f:
        return int(f.read().strip())
    return -1
