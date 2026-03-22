import os
import signal
import subprocess
import shlex
import threading
from collections import OrderedDict
import time
import re


class JobManager(object):

    def __init__(self, max_jobs=1024, max_running=16, timeout=1):
        # a job is a dict
        self.start_time = time.time()
        self.max_running = max_running
        self.jobs = OrderedDict()
        self.running_jobs = []
        self.finished = []
        self.passed = []
        self.failed = []
        self.timeout = timeout
        self.__stop__ = False
        self._enable_bsub = False
        self.mt = threading.Thread(target=self.monitoring)

    def run(self):
        self.mt.start()
        self.mt.join()

    @property
    def pass_num(self):
        return len(self.passed)

    @property
    def fail_num(self):
        return len(self.failed)

    def add_job(self, name, cmd, timeout=1800, core=1, log_to_file=True, flags="-----"):
        # flags for pretty print result
        pretty_name = "|" + flags.ljust(len("H S D")) + "|"
        pretty_name += name.ljust(50) + "|"

        self.jobs[name] = (cmd, timeout, core, log_to_file, pretty_name)

    def _kill_all_jobs(self):
        for job in self.running_jobs:
            pid = job["pid"]
            print("killing pid:%d" % (pid.pid))
            os.killpg(os.getpgid(pid.pid), signal.SIGTERM)

    def monitoring(self):
        no_job_time = 0

        if self._enable_bsub:
            self.max_running = 160  # at least 10 host to run case

        print("|FLAGS|" + "NAME".ljust(50) + "|RESULT")

        while True:
            try:
                if self.__stop__:
                    self._kill_all_jobs()
                    break
                for job_dict in self.running_jobs:
                    job = job_dict["pid"]
                    timeout = job_dict["timeout"]

                    # if using bsub, REAL timeout using -W option
                    # enlarge timeout for pending time
                    if self._enable_bsub:
                        timeout *= 10

                    t0 = job_dict["start_time"]
                    ret = job.poll()
                    if ret is not None:
                        self.running_jobs.remove(job_dict)
                        job_dict["cost_time"] = time.time() - t0
                        if ret == 0:
                            self.passed.append(job_dict)
                            result = "(pass)"
                        else:
                            self.failed.append(job_dict)
                            result = "(fail) <-- :("
                        try:
                            with open(job_dict["log_file"], "r") as f:
                                log = f.read()
                                ct = re.findall(
                                    r"compile time:\s*([\d\.]+)\s*s", log)
                                cm = re.findall(r"Total cycles estimated by cost model is:\s+([\d\.]+)", log)
                                if len(ct) != 0:
                                    result += f" compiling time={ct[0]}s"
                                if len(cm) != 0:
                                    result += f" estimated cycles={cm[-1]}"
                        except Exception as e:
                            # result += str(e)
                            pass

                        print("%sfinish, cost_time = %-3.2fs %s"
                              % (job_dict["pretty_name"], job_dict["cost_time"], result), flush=True)
                    elif timeout > 0 and time.time() - t0 > timeout:
                        job_dict["cost_time"] = time.time() - t0
                        print("%stimeout, cost_time = %-3.2fs %s"
                              % (job_dict["pretty_name"], job_dict["cost_time"], "(timeout) <-- :("), flush=True)
                        os.killpg(os.getpgid(job.pid), signal.SIGTERM)
                        self.failed.append(job_dict)
                        self.running_jobs.remove(job_dict)

                if len(self.jobs) > 0 and len(self.running_jobs) < self.max_running:
                    name = list(self.jobs.keys())[0]
                    cmd, timeout, core, log_to_file, pretty_name = self.jobs.pop(
                        name)

                    cmd = shlex.split(cmd)

                    if timeout > 0:  # second -> min
                        bsub_timeout = int((timeout + 59) / 60)
                    else:
                        bsub_timeout = 60  # min

                    if self._enable_bsub:
                        bsub_cmd = [
                            # submit to aisw queue:
                            "/project/ai/scratch02/regai02/scripts/Jenkinsfile/submit.sh",
                            # args of bsub:
                            "-n", str(core),                    # core required
                            "-W", str(bsub_timeout),            # timeout
                            # setup environment for every case:
                            "./run.sh"
                        ] + cmd
                    else:
                        bsub_cmd = cmd

                    # redirect logs to file
                    if log_to_file:
                        log = open(self.job_log_file_name(name), "w")
                        pid = subprocess.Popen(
                            bsub_cmd, preexec_fn=os.setsid, stdout=log, stderr=log)
                    else:
                        pid = subprocess.Popen(bsub_cmd, preexec_fn=os.setsid)

                    pid.name = name
                    running_job = dict()
                    running_job["name"] = name
                    running_job["cmd"] = ' '. join(cmd)
                    running_job["pretty_name"] = pretty_name
                    running_job["pid"] = pid
                    running_job["timeout"] = timeout
                    running_job["start_time"] = time.time()
                    running_job["log_file"] = self.job_log_file_name(name)
                    self.running_jobs.append(running_job)

                if len(self.running_jobs) == 0:
                    if no_job_time == 0:
                        no_job_time = time.time()
                    if time.time() - no_job_time > self.timeout:
                        break
                elif no_job_time != 0:
                    no_job_time = 0
                time.sleep(0.5)
            except (Exception, KeyboardInterrupt) as e:
                self._kill_all_jobs()
                raise e

        print("|FLAGS|" + "NAME".ljust(50) + "|RESULT")
        print("|FLAGS: [L M H]: priority")
        print("|       [S]: check outputs of Simulator")
        print("|       [D]: check outputs of Dumper")
        print("|PASSED %s FAILED %s" % (len(self.passed), len(self.failed)))

        # print some log of failed cases
        for i, failed_job in enumerate(self.failed):
            failed_log_name = self.job_log_file_name(failed_job['name'])
            print(("[%d] %s" % (i, failed_log_name)).center(80, "="))
            print("..... tail of log file:", flush=True)
            subprocess.run(["tail", failed_log_name])
            print("=" * 80)
            print(("rerun this case: " + failed_job["cmd"]).center(80, "="))
            print(("[%d] %s" % (i, failed_log_name)).center(80, "="))
            print("")

        print("|PASSED %s FAILED %s" % (len(self.passed), len(self.failed)))

        # tar logs
        logs_files = []
        for root, dirs, files in os.walk('.'):
            for f in files:
                if f.endswith("log.txt"):
                    logs_files.append(os.path.join(root, f))

        if len(logs_files) > 0:
            subprocess.run(["tar", "czf", "logs.tgz"] + logs_files)

        if len(self.failed) > 0:
            print(
                "Tips: find the full log 'CASE_NAME_log.txt' at >>> log.tgz <<<, you can download it by Jenkins")
        if "BUILD_URL" in os.environ:
            print(("%s/artifact/logs.tgz" %
                   os.environ["BUILD_URL"]).center(80, '='))

    def stop(self):
        self.__stop__ = True
        print("Waiting monitoring stop")
        while self.mt.isAlive():
            time.sleep(0.1)
        for job in self.running_jobs:
            job.kill()

    def enable_bsub(self):
        self._enable_bsub = True

    def disable_bsub(self):
        self._enable_bsub = False

    def job_log_file_name(self, name):
        illegal_char_pattern = r'[\\/:*?"<>|] '
        name = re.sub(illegal_char_pattern, '_', name)
        name = name.strip('. ') + "_log.txt"
        if len(name) > 255:
            name = name[:50] + "_" + name[-180:]
        return name


if __name__ == "__main__":
    jm = JobManager()
    job = "aipurun -v"
    for i in range(5):
        jm.add_job(i, job)
