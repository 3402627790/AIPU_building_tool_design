from .logger import *
from .configs import *

import subprocess
import json
import os
import shutil
import numpy as np
import csv
from AIPUBuilder.driver import *  # noqa
from AIPUBuilder.core import *  # noqa
from AIPUBuilder._C._builder import *  # noqa
import multiprocessing
import time
import sys
import signal


class Driver(object):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.cnt = 0
        self.phase = 0
        self.try_idx = 0
        self.sim_idx = 0
        self.action_idx = 0
        # self.btn_idx = 0
        # self.strg_idx = 0
        # self.move_idx = 0
        self.loop_idx = 0
        self.folder = "strategy_dump"
        # self.data_load = data_load
        # self.strategy_dump = data_dump
        # self.ir = ir
        # self.gb = gb
        # self.wt = wt
        # self.inputs = inputs
        # self.target = target
        # self.iteration = iteration
        self.csv = [["#", "strg#", "sim#", "cycle", "strg file", "data file"]]
        self.strategy_file = ""
        self.prior_in_gm_file = ""
        self.skip = False

    def try_strategy(self, strategy, prior_in_gm):
        file_path = "tune_strategy_tmp.json"
        new_file_path = "tune_strategy_" + str(self.phase) + "_" + str(self.try_idx) + "_" + str(
            self.sim_idx) + "_" + str(self.action_idx) + "_" + str(self.loop_idx) + ".json"
        file2_path = "prior_in_gm.json"
        new_file2_path = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as jsonfile:
                tune_data = json.load(jsonfile)
                # logger.info("tune_data " + str(tune_data))
                logger.info("strategy " + str(strategy))
                sg_ids = []
                # for layer, value in strategy.items():
                last_split = -1
                last_num = -1
                last_parallel = -1
                last_sg_id = 0
                for layer in tune_data:
                    # logger.info("layer " + layer)
                    # if layer in tune_data:
                    new_start = 0
                    if layer in strategy:
                        # logger.info("tune layer " + layer)
                        nested_data = tune_data[layer]
                        # logger.info("tune layer " + layer + " value " + str(value))
                        # print(type(nested_data))
                        value = strategy[layer]
                        new_start = (value >> 31) & 0x1
                        if isinstance(nested_data, dict):
                            nested_data["split"] = (value) & 0xFF
                            nested_data["num"] = (value >> 8) & 0xFF
                            nested_data["parallel"] = (value >> 16) & 0xFF
                            # logger.info("tuned layer " + layer + " nested_data " + str(nested_data))
                        else:
                            logger.error("error")
                    nested_data = tune_data[layer]
                    # logger.info("nested_data " + str(nested_data))
                    if isinstance(nested_data, dict):
                        split = nested_data["split"]
                        num = nested_data["num"]
                        parallel = nested_data["parallel"]
                        # logger.info("split " + str(split) + " last split " + str(last_split))
                        sg_id = nested_data["sg"]
                        if split != last_split or num != last_num or parallel != last_parallel or new_start == 1:
                            if sg_id in sg_ids:
                                while sg_id in sg_ids:
                                    sg_id += 1
                            sg_ids.append(sg_id)
                            # logger.info("should use new sg id " + str(sg_id))
                            nested_data["sg"] = sg_id
                        else:
                            sg_id = last_sg_id
                            # logger.info("should use last sg id " + str(sg_id))
                            nested_data["sg"] = sg_id
                        last_split = split
                        last_num = num
                        last_parallel = parallel
                        last_sg_id = sg_id

                with open(new_file_path, 'w', encoding='utf-8') as new_jsonfile:
                    json.dump(tune_data, new_jsonfile, indent=4)
                    logger.info("Modified data saved to " + new_file_path)
                    self.strategy_file = new_file_path
        except FileNotFoundError:
            print(f"The file '{file_path}' was not found.")
        except json.JSONDecodeError:
            print(f"Invalid JSON format in file '{file_path}'.")
        except Exception as e:
            print(f"An error occurred: {e}")
        
        data_filename = "data_" + str(self.phase) + "_" + str(self.try_idx) + "_" + str(
            self.sim_idx) + "_" + str(self.action_idx) + "_" + str(self.loop_idx) + ".json"

        if get_config("strategy_dump"):
            dest = os.path.join(self.folder, new_file_path)
            if os.path.exists(dest):
                os.remove(dest)
            shutil.move(new_file_path, self.folder)

        if self.phase == 2:
            new_file2_path = "prior_in_gm_" + str(self.phase) + "_" + str(self.try_idx) + "_" + str(
                self.sim_idx) + "_" + str(self.action_idx) + "_" + str(self.loop_idx) + ".json"
            
            try:
                with open(file2_path, 'r', encoding='utf-8') as jsonfile:
                    tune_data = json.load(jsonfile)
                    # logger.info("tune_data " + str(tune_data))
                    logger.info("prior_in_gm " + str(prior_in_gm))
                    for layer in tune_data:
                        # logger.info("layer " + layer)
                        # if layer in tune_data:
                        if layer in prior_in_gm:
                            # logger.info("tune layer " + layer)
                            nested_data = tune_data[layer]
                            # logger.info("tune layer " + layer + " value " + str(value))
                            # print(type(nested_data))
                            if isinstance(nested_data, dict):
                                nested_data["prior_in_gm"] = 0
                                # logger.info("tuned layer " + layer + " nested_data " + str(nested_data))
                            else:
                                logger.error("error")
                   
                    with open(new_file2_path, 'w', encoding='utf-8') as new_jsonfile:
                        json.dump(tune_data, new_jsonfile, indent=4)
                        logger.info("Modified data saved to " + new_file2_path)
                        self.prior_in_gm_file = new_file2_path
            except FileNotFoundError:
                print(f"The file '{file2_path}' was not found.")
            except json.JSONDecodeError:
                print(f"Invalid JSON format in file '{file2_path}'.")
            except Exception as e:
                print(f"An error occurred: {e}")

            data, cycle = self.get_data(data_filename, new_file_path, new_file2_path,
                                    "--load_balance_strategy 0 --branch_reorder_strategy -1 --closer_sg_opt --dump_tune_strategy --dump_prior_in_gm")
            if get_config("strategy_dump"):
                dest = os.path.join(self.folder, new_file2_path)
                if os.path.exists(dest):
                    os.remove(dest)
                shutil.move(new_file2_path, self.folder)

        else:
            data, cycle = self.get_data(data_filename, new_file_path, new_file2_path,
                                    "--load_balance_strategy 0 --branch_reorder_strategy -1 --closer_sg_opt --dump_tune_strategy --dump_prior_in_gm")



        new_row = [self.cnt, self.sim_idx, self.action_idx, cycle, new_file_path, data_filename]
        self.csv.append(new_row)
        self.cnt += 1

        return data, cycle

    def get_data(self, data_filename, strategy, prior_in_gm, option):
        # logger.info("data_filename " + data_filename)
        logger.info("folder " + self.folder)
        dest = os.path.join(self.folder, data_filename)
        cycles = []
        if get_config("data_load"):
            with open(dest, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if data is not None:
                    profile_data = data["profile_data"]
                    cycle = profile_data["Total Cycles"]
                    cycles.append(cycle)
        else:
            aipu = self.run_gb(strategy, prior_in_gm, option)
            if aipu is not None:
                for i in range(get_config("iteration")):
                    logger.info("Running iteration " + str(i))
                    data = self.__run_runtime(aipu)

                    # while(self.__abnormal_check(data)):
                    #     logger.info("abnormal data run again!")
                    #     prof = self.__run_runtime(aipu)
                    #     data = self.get_prof(prof, aipu)
                    if get_config("strategy_dump"):
                        with open(data_filename, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=4)
                        if os.path.exists(dest):
                            os.remove(dest)
                        shutil.move(data_filename, dest)

                    if data is not None:
                        profile_data = data["profile_data"]
                        cycle = profile_data["Total Cycles"]
                        cycles.append(cycle)
            else:
                data = None

        if len(cycles) > 0:
            cycle = round(sum(cycles) / len(cycles))
        else:
            cycle = -1

        return data, cycle

    def run_gb(self, strategy, prior_in_gm, option):
        if self.skip:
            logger.info("skip")
            return None
        out = "aipu.bin"
        gb = get_config("gb")
        ir = get_config("ir")
        wt = get_config("wt")
        target = get_config("target")

        if strategy == "":
            tune_option = f"--dump_tune_strategy --dump_prior_in_gm"
        else:
            if self.phase == 2 and prior_in_gm != "":
                tune_option = f"--load_tune_strategy {strategy} --load_prior_in_gm {prior_in_gm}"
            else:
                tune_option = f"--load_tune_strategy {strategy}"

        

        if option == "":
            cmd = f"{gb} {ir} -w {wt} --wdc_rate 0 --target {target} -p --fast_perf {tune_option} -o {out} --show_short_name"
        else:
            cmd = f"{gb} {ir} -w {wt} --wdc_rate 0 --target {target} -p --fast_perf {tune_option} -o {out} --show_short_name {option}"
        returncode = self.__run_command(cmd)
        if returncode != 0:
            logger.info("gb failed")
            return None
        return out

    def __worker(self, aipu, return_dict):
        try:
            inputs = get_config("inputs").split(',')
            job = AIPUJob(aipu)
            input_list = []
            for i in range(len(inputs)):
                with open(inputs[i], 'rb') as f:
                    numpy_array = np.frombuffer(f.read(), dtype=job.model.inputs[i].dtype.np)
                writable_array = numpy_array.copy()
                inp = Tensor(writable_array)
                input_list.append(inp)
            out = job.run(input_list)
            prof = job.get_profdata()
            data = self.get_prof(prof, aipu)
            return_dict['success'] = True
            return_dict['data'] = data
        except Exception as e:
            # import traceback
            # logger.error("Exception occurred:")
            # logger.error(traceback.format_exc())  # 打印完整的错误栈
            logger.info("exception")
            return_dict['success'] = False
            return_dict['error'] = str(e)

    def __run_with_timeout(self, aipu, timeout):
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        p = multiprocessing.Process(target=self.__worker, args=(aipu, return_dict))

        p.start()

        start_time = time.time()
        while p.is_alive():
            if time.time() - start_time > timeout:
                print("Timeout! Terminating job process...")
                # p.terminate()
                os.kill(p.pid, signal.SIGTERM)  # 发送 SIGKILL
                p.join()
                return None, "Timeout"
            time.sleep(0.2)

        p.join()

        if not return_dict:
            return None, "Crash"

        if return_dict.get('success'):
            return return_dict.get('data'), None
        else:
            return None, return_dict.get('error', 'Unknown error')

    def __run_runtime(self, aipu):
        # output = subprocess.run("lsmod|grep aipu", check=True, shell=True)
        # logger.info("before runtime")
        try:
            enable_tick_counter()
        except:
            logger.warning("enable_tick_conter failed")
        data, error = self.__run_with_timeout(aipu, 10)
        if error is not None:
            logger.info("error: " + error)
            logger.info("Trying to reload AIPU driver...")
            # time.sleep(2)
            # output = subprocess.run("lsmod|grep aipu", check=True, shell=True)
            # logger.info("after runtime")

            self.reset()

            # if error == "Timeout":
            #     logger.info("will exit PGO due to timeout, may need to reset NPU")
            #     subprocess.run("sudo rmmod aipu",shell=True)
            #     subprocess.run("sudo insmod /home/micche01/debug_250416/aipu.ko",shell=True)
            #     # sys.exit(1)

            # if error == "Crash":
            #     logger.info("will exit PGO due to crash, may need to reset NPU")
            #     subprocess.run("sudo rmmod aipu",shell=True)
            #     subprocess.run("sudo insmod /home/micche01/debug_250416/aipu.ko",shell=True)
            #     # sys.exit(1)

            return None
        else:
            # output = subprocess.run("lsmod|grep aipu", check=True, shell=True)
            # logger.info("after runtime")
            logger.info('success')
            return data
        # prof = job.get_profdata()
        # return prof

    def get_prof(self, prof, aipu):
        if prof is None:
            return None

        if len(prof) != 0:
            # gen prof report
            from AIPUBuilder.Profiler.get_profiler_data import get_profiler_data
            import json
            elf = AIPUELF()
            elf.load(aipu)
            gjson = json.loads(elf.graphjson)
            del elf
            data = get_profiler_data(gjson, prof[0].np.view(
                "uint8"), frequency=get_cur_freq() // 1000000)

        return data

    def gen_csv(self):
        with open("pgo.csv", mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerows(self.csv)

    def __abnormal_check(self, data):
        profile_data = data["profile_data"]
        subgraphs = profile_data["subgraphs"]
        start_list = []
        end_list = []
        for subgraph in subgraphs:
            nodes = subgraph["nodes"]
            level = subgraph["Level"]
            for node in nodes:
                start = node["start"]
                end = node["end"]
                # logger.info("level " + str(level) + " node " + str(node["Layer Name"]) + " start " + str(start) + " end " + str(end))
                if level > len(start_list):
                    start_list.append(start)
                    end_list.append(end)
                else:
                    value = start_list[level-1]
                    if start < value:
                        start_list[level-1] = start
                    value = end_list[level-1]
                    if end > value:
                        end_list[level-1] = end
                # break
            # logger.info("level " + str(level) + " start " + str(start_list[level-1]) + " end " + str(end_list[level-1]))
        for i in range(1, len(start_list)):
            # logger.info("abnormal check " + str(i) + " between start and end " + str(start_list[i] - end_list[i-1]))
            if start_list[i] - end_list[i-1] > 1000000000:
                return True
            # else:
            #     logger.info("normal data diff " + str(start_list[i] - start_list[i-1]))

        return False

    def reset(self):
        try:
            device_soft_reset()
        except Exception as e:
            logger.error(f"Failed to perform soft reset: {e}")
            sys.exit(1)

        # try:
        #     subprocess.run(["sudo", "rmmod", "aipu"], check=True)
        # except subprocess.CalledProcessError:
        #     logger.error("Failed to rmmod aipu.")
        #     # sys.exit(1)
        #     # subprocess.run(["sudo", "rmmod", "-f", "aipu"], check=True)

        # try:
        #     subprocess.run([
        #         "sudo", "insmod",
        #         "/project/ai/scratch01/kylhua01/Compass_NPU_Driver/Linux/bin/sky1/debug/aipu.ko"
        #     ], check=True)
        # except subprocess.CalledProcessError:
        #     logger.error("Failed to insmod aipu. will exit")

    def skip_run(self):
        self.skip = True

    def __run_command(self, command):
        logger.info("run_command:" + command)

        try:
            result = subprocess.run(command, check=True, shell=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, timeout=300)

        except subprocess.CalledProcessError as e:
            logger.info("run_command:" + command + " failed")
            return -1

        except subprocess.TimeoutExpired:
            logger.info("run_command:" + command + " time out")
            return -1
        # os.system(command)
        return result.returncode
