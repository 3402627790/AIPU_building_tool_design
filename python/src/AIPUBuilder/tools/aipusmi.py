import fcntl
import os
import struct
import time
import sys
import numpy as np
from collections import deque
from threading import Thread
import argparse
from AIPUBuilder._C._driver import *
from AIPUBuilder.utils import continue_class
__VERSION__ = "0.1.0"


def get_command_by_pid(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            cmdline = f.read().decode("utf-8").split("\x00")  # 分割参数
            cmdline = [arg for arg in cmdline if arg]  # 过滤空字符串
            return " ".join(cmdline)
    except FileNotFoundError:
        return f""
    except PermissionError:
        return f"no permission"


GET_RUNNING_JOB_THREAD_ID = 0x80804119
AIPU_IOCTL_GET_RUNNING_JOB_THREAD_ID = 0x80804119
AIPU_IOCTL_QUERY_CAP = 0x80c84100
AIPU_IOCTL_QUERY_PARTITION_CAP = 0x80804101
AIPU_IOCTL_GET_DRIVER_VERSION = 0x80084112
AIPU_IOCTL_ENABLE_TICK_COUNTER = 0x410d
AIPU_IOCTL_DISABLE_TICK_COUNTER = 0x410c
AIPU_IOCTL_GET_CLUSTER_STATUS = 0xc00c4118


def table_line(s, pads=' ', split='|'):
    s = s.replace('\t', ' ' * 4)
    if len(s) < 87:
        s += pads * (87 - len(s))
    # print(s,len(s))
    return split + s + split + '\n'


def table_line_cols(cols, widths, pads=' ', split='|'):
    s = split
    for i in range(len(cols)):
        s += cols[i]
        s += pads * (widths[i] - len(cols[i]))
        s += split
    s += '\n'
    return s


def get_aipu_freq():
    with os.popen("cat /sys/class/devfreq/*.aipu/cur_freq") as f:
        return int(f.read().strip())
    return -1


@continue_class
class AIPUClusterCap:
    def __repr__(self):
        return f"AIPUClusterCap(core_cnt={self.core_cnt}, en_core_cnt={self.en_core_cnt}, tec_cnt={self.tec_cnt})"

    def __str__(self) -> str:
        return self.__repr__()


@continue_class
class AIPUPartitionCap:

    def __repr__(self) -> str:
        return f"AIPUPartionCap(pid={self.id}, arch={self.arch}, version={self.version}, config={self.config}, bus_addr={self.reg_base:#x}, clusters={self.clusters})"

    def __str__(self) -> str:
        return self.__repr__()


@continue_class
class AIPUCap:

    @property
    def name(self):
        if self.partition_cap:
            arch = self.partition_cap.version
            name = "AIPU "
            if arch == 5:
                name += "X2_120"
            elif arch == 6:
                name += "X3_130"
            cluster = self.partition_cap.clusters[0]
            name += str(cluster.tec_cnt)
            if cluster.core_cnt > 0:
                name += f"MP{cluster.core_cnt}"
            return name
        return "AIPU not Presented"

    @property
    def busid(self):
        return f"{self.partition_cap.reg_base:#x}" if self.partition_cap else "Bus ID not available"

    def __repr__(self):
        return f"AIPUCap(partition_cnt={self.partition_cnt}, asid={self.asid_base}, is_homogeneous={self.is_homogeneous}, gm0_size={self.gm0_size}, gm1_size={self.gm1_size}, partition={self.partition_cap})"

    def __str__(self) -> str:
        return self.__repr__()

    def pretty_print(self):
        line = '+---------------------------------------------------------------------------------------+\n'
        version = get_driver_version()

        versions = ' AIPUSMI Version:%s' % __VERSION__ + \
            '     Driver Version: %s' % version
        ret = line + table_line(versions)
        widths = [32, 20, 16, 16]
        ret += table_line_cols(['-', '-', '-', '-'], widths, '-', '+')
        ret += table_line_cols([' NPU Name', ' BUS-ID', ' NPU-Util', ' BUS-Util'],
                               widths) + table_line_cols(['=', '=', '=', '='], widths, '=', '+')
        return ret


class NPUUsageMonitor:
    def __init__(self, device="/dev/aipu", freq=20):
        self.npu_usage = deque(maxlen=max(freq, 100))
        self.bus_usage = deque(maxlen=max(freq, 100))
        # self.npu_core_=set()
        self.device = open(device, 'rb')
        self.sleep_time = 1. / freq * 0.95  # give some times for calucation
        self.core_num = 3
        self.npu_usage.append([1] * self.core_num)
        # self.min = 3
        self.aipu = get_aipu_cap()
        self.aipu_name = self.aipu.name
        self.aipu_bus = self.aipu.busid
        self.header = self.aipu.pretty_print()
        self.__stop__ = False

    def _get_data(self):
        s = get_cluster_status(0)
        return [s.status, s.core_status]

    def start_collection(self):
        mask = (1 << self.core_num) - 1

        def collect():
            while True:
                data = self._get_data()
                self.bus_usage.append((data[0] >> 6) & 1)
                s = []
                for i in range(0, self.core_num):
                    s .append((data[1] >> i) & 1)
                # self.npu_core_.add(data[1])
                self.npu_usage.append(s)
                time.sleep(self.sleep_time)
                if self.__stop__:
                    break
        self.thread = Thread(target=collect)
        self.thread.start()

    def stop_collection(self):
        self.__stop__ = True

    def get_average(self):
        if not self.npu_usage:
            return 0.0, 0.0
        # self.min = min(min(self.npu_usage),self.min)
        usage = np.array(self.npu_usage, dtype=np.int32)
        s = (1. - np.mean(usage, axis=0)) * 100
        bus = 0
        if len(self.bus_usage) > 0:
            bus = (1. - sum(self.bus_usage) / len(self.bus_usage)) * 100
        return s, bus

    def get_running_job_thread(self):
        return get_running_job_thread_id()

    def mesage(self):
        cores, bus = self.get_average()
        npu = np.mean(cores)
        freq = get_aipu_freq() / 1000 / 1000

        ret = self.header + table_line_cols([f"{self.aipu_name} @{freq}MHz",
                                             self.aipu_bus, f"{npu:.2f}%", f"{bus:.2f}%"], [32, 20, 16, 16])
        for i, core in enumerate(cores):
            ret += table_line_cols([f"    Core {i}",
                                    "", f"{core:.2f}%", ""], [32, 20, 16, 16])
        ret += table_line("=", '=', '+')
        ret += '+---------------------------------------------------------------------------------------+\n'
        ret += table_line(" Processes:")
        widths = [16, 70]
        ret += table_line_cols([" PID", " Process name"], widths)
        ret += table_line_cols(["=", "="], widths, '=', '+')

        def cut_cmd(cmd):
            if len(cmd) > 70:
                return cmd[:57] + "..." + cmd[-10:]
            return cmd
        for pid in self.get_running_job_thread():
            try:
                process = get_command_by_pid(pid)
                if len(process):
                    ret += table_line_cols([str(pid),
                                            cut_cmd(process)], widths)
            except Exception:
                ret += table_line_cols([pid,
                                        "no premission to access process info"], widths)
        ret += table_line("=", '=', '+')
        return ret


def main():
    args = argparse.ArgumentParser()
    args.add_argument('-v', '--version', action='version',
                      version='%(prog)s version ' + __VERSION__)
    args.add_argument('-l', '--listen', type=float, default=-1,
                      help='frequency of monitoring the npu status.')
    args.add_argument('-s', '--sampling', type=int, default=200,
                      help='the sampling rate of the monitor.')
    options = args.parse_args(sys.argv[1:])

    npusmi = NPUUsageMonitor(freq=options.sampling)
    npusmi.start_collection()

    def listenning(f):
        try:
            while True:
                print("\033[H\033[J", end="")
                print(npusmi.mesage())
                time.sleep(f)

        except KeyboardInterrupt:
            npusmi.stop_collection()
    if options.listen > 0:
        listenning(options.listen)
    else:
        time.sleep(0.5)  # for get data
        print(npusmi.mesage())
        npusmi.stop_collection()


if __name__ == "__main__":
    main()
