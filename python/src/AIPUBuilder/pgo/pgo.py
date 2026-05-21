
import subprocess
import os
import glob
import json
import shutil
import ast
import re
import numpy as np

from .logger import *
from .mcts import *
from .driver import *
from .configs import *


CORE_NUM = 3
INIT_LAYER_NUM = 10
DEFAULT_CHANCE = 2


class GB_State(GameState):
    def __init__(self, data, cycle, last_cycle, strategy, strategy_file, prior_in_gm, prior_in_gm_file, nodes, btn_node, tuned_set, driver, action, try_idx, sim_idx, phase, chance):
        self.data = data
        self.cycle = cycle
        self.last_cycle = last_cycle
        self.strategy = strategy
        self.strategy_file = strategy_file
        self.prior_in_gm = prior_in_gm
        self.prior_in_gm_file = prior_in_gm_file
        self.nodes = nodes
        self.btn_node = btn_node
        self.tuned_set = tuned_set
        self.driver = driver
        self.action = action
        self.try_idx = try_idx
        self.sim_idx = sim_idx
        self.terminal = False
        self.phase = phase
        self.chance = chance

        # logger.info("strategy: " + str(self.strategy))

    def get_possible_actions(self):

        # split
        # 1 n
        # 2 h
        # 3 c
        nxt_split = []
        nxt_num = 0
        nxt_start = 0
        nxt_end = 0
        self.terminal = False

        logger.info("last cycle: " + str(self.last_cycle) + " cycle " + str(self.cycle))
        if self.last_cycle > 0 and self.cycle > 0 and self.cycle < self.last_cycle:
            logger.info("no better")

        level_idx = self.btn_node[5]
        sgs = self.__get_exe_sg(level_idx)
        logger.info("exe sg: " + str(len(sgs)))


        if self.action == 0:
            logger.info("current action 0")
            if self.phase == 1:
                nxt_parallel = 1
                nxt_direction = 0
                nxt_num = 0
                start,end = self.__get_parallel_range(sgs[nxt_num])
                logger.info(str(start) + " " + str(end))
                if start != -1 and end != -1:
                    nxt_start = start
                    nxt_end = end
                    nxt_split.append(0)
            elif self.phase == 2:
                nxt_parallel = 1
                nxt_direction = 1
                nxt_num = 0
                start,end = self.__get_sg_node()
                logger.info(str(start) + " " + str(end))
                if start != -1 and end != -1:
                    nxt_start = start
                    nxt_end = end
                    nxt_split.append(0)
            else:
                nxt_parallel = 0
                nxt_direction = 0
                nxt_num = 3
                start, end = self.__get_tiling_range(0)
                logger.info(str(start) + " " + str(end))
                if start != -1 and end != -1:
                    nxt_start = start
                    nxt_end = end
                    nxt_split.append(1)

                start, end = self.__get_tiling_range(1)
                logger.info(str(start) + " " + str(end))
                if start != -1 and end != -1:
                    nxt_start = start
                    nxt_end = end
                    nxt_split.append(2)
                    nxt_split.append(5)

                start, end = self.__get_tiling_range(3)
                logger.info(str(start) + " " + str(end))
                if start != -1 and end != -1:
                    nxt_start = start
                    nxt_end = end
                    nxt_split.append(3)

                if not nxt_split:
                    nxt_start = 0
                    nxt_end = 0
                    nxt_split.append(0)
                    self.terminal = True
        else:
            cur_start, cur_end, cur_split, cur_num, cur_direction, cur_parallel = self.__decode_params(self.action)
            logger.info("current action parallel " + str(cur_parallel))
            if cur_parallel == 1:
                if cur_direction == 0:
                    num = cur_split << 3 | cur_num
                    num = num + 1 #next sg num
                    if num >= len(sgs):
                        self.terminal = True
                    else:
                        nxt_direction = cur_direction
                        nxt_parallel = 1
                        start,end = self.__get_parallel_range(sgs[num])
                        nxt_start = start
                        nxt_end = end
                        nxt_num = num & 0x7
                        nxt_num_h = num >> 3
                        nxt_split.append(nxt_num_h)
                        if start == -1 or end == -1:
                            self.terminal = True
                else:
                    nxt_direction = cur_direction
                    nxt_start = cur_start
                    nxt_end = cur_end
                    nxt_split = [cur_split]
                    nxt_num = cur_num
                    nxt_parallel = cur_parallel
                    self.terminal = True
            else:
                logger.info("cur direction " + str(cur_direction))
                if self.last_cycle != -1 and self.cycle > self.last_cycle:
                    if self.chance == 0:
                        if cur_direction == 0:
                            logger.info("DEBUG should change direction")
                            cur_direction = 1
                            self.chance = DEFAULT_CHANCE
                            self.terminal = False
                        else:
                            logger.info("DEBUG should stop")
                            self.terminal = True
                            nxt_direction = cur_direction
                            nxt_start = cur_start
                            nxt_end = cur_end
                            nxt_split = [cur_split]
                            nxt_num = cur_num
                            nxt_parallel = cur_parallel
                    else:
                        self.chance = self.chance - 1
                        logger.info("DEBUG chance " + str(self.chance))

                if self.terminal == False:
                    nxt_direction = self.__get_nxt_direction(cur_direction, cur_start, cur_end, cur_split)
                    if cur_direction != nxt_direction:
                        self.chance = DEFAULT_CHANCE
                    logger.info("DEBUG nxt direction " + str(nxt_direction))
                    if nxt_direction == 0:
                        nxt_start = cur_start
                        nxt_end = cur_end + 1
                        nxt_split = [cur_split]
                        nxt_num = cur_num
                        nxt_parallel = cur_parallel
                        self.terminal = False
                    elif nxt_direction == 1:
                        nxt_start = cur_start - 1
                        nxt_end = cur_end
                        nxt_split = [cur_split]
                        nxt_num = cur_num
                        nxt_parallel = cur_parallel
                        self.terminal = False
                    else:
                        self.terminal = True
                        nxt_direction = cur_direction
                        nxt_start = cur_start
                        nxt_end = cur_end
                        nxt_split = [cur_split]
                        nxt_num = cur_num
                        nxt_parallel = cur_parallel


        combo = []

        for sp in nxt_split:
            logger.info("get next action: " + str(self.__encode_params(nxt_start, nxt_end, sp, nxt_num, nxt_direction, nxt_parallel)) + " from " + str(nxt_start) + " " + str(nxt_end) + " " + str(sp) + " " + str(nxt_num) + " " + str(nxt_direction) + " " + str(nxt_parallel))
            combo.append(self.__encode_params(nxt_start, nxt_end, sp, nxt_num, nxt_direction, nxt_parallel))
        # logger.info("action: " + str(combo))
        # sys.exit()

        return combo

    def is_terminal(self):
        if self.terminal:
            return True

        return False

    def get_reward(self):
        return self.cycle

    def take_action(self, action):
        self.driver.phase = self.phase
        new_strategy = self.strategy
        new_prior_in_gm = self.prior_in_gm

        if action != 0:
            start, end, split, num, direction, parallel = self.__decode_params(action)
            logger.info("take action: " + str(action) + " " + str(start) +
                        " " + str(end) + " " + str(split) + " " + str(num) + " " + str(direction) + " " + str(parallel))
            self.driver.try_idx = self.try_idx
            self.driver.sim_idx = self.sim_idx
            self.driver.action_idx = action
            # logger.info("current strategy: " + str(self.strategy))
            if self.phase == 2:
                new_prior_in_gm = self.__gen_prior_in_gm(parallel, direction, split, num, start, end)
                logger.info("new prior in gm: " + str(new_prior_in_gm))
            else:
                new_strategy = self.__gen_strategy(parallel, direction, split, num, start, end)
                logger.info("new strategy: " + str(new_strategy))
            
        data, cycle = self.driver.try_strategy(new_strategy,new_prior_in_gm)
        # logger.info("strategy file: " + self.strategy_file)
        logger.info("try strategy cycle: " + str(cycle) + " cycle: " + str(self.cycle) + " last cycle: " + str(self.last_cycle))
        # display_data(data)
        new_state = GB_State(data, cycle, self.cycle, new_strategy, self.driver.strategy_file, new_prior_in_gm, self.driver.prior_in_gm_file,
                             self.nodes, self.btn_node, self.tuned_set, self.driver, action, self.try_idx, self.sim_idx + 1, self.phase, self.chance)
        return new_state

    def _is_winner(self, player):
        return False

    def __get_nxt_sg(self, cur_sg):
        if cur_sg+1 == len(self.level):
            return -1

        for i in range(cur_sg+1, len(self.level)):
            sg_id, sg = self.level[i]
            load = self.__get_sg_load(sg)
            if load > 2000:
                return i
        return -1

    def __get_sg_load(self, sg):
        load = 0
        for layer, load_cyc, tiling, sg_name, sg_idx, level_idx, sgs, gm in sg:
            load = load + load_cyc

        return load

    def __encode_params(self, start, end, split, num, direction, parallel):
        if parallel > 1: 
            logger.info("parallel error")
        if direction > 1:
            logger.info("direction error")
        if num > 7:
            logger.info("num error")
        if split > 7:
            logger.info("split error")
        if end > 4095:
            logger.info("end error")
        if start > 4095:
            logger.info("start error")

        return parallel << 31 | direction << 30 | num << 27 | split << 24 | end << 12 | start

    # 31:31 parallel      
    # 30:30 direction/gm
    # 27:29 split number
    # 24:26 split dim
    # 12:23 end node
    # 0:11 start node
    def __decode_params(self, encoded_value):
        parallel = (encoded_value >> 31) & 0x1
        direction = (encoded_value >> 30) & 0x1
        num = (encoded_value >> 27) & 0x7
        split = (encoded_value >> 24) & 0x7
        end = (encoded_value >> 12) & 0xFFF
        start = encoded_value & 0xFFF
        return start, end, split, num, direction, parallel

    def __get_nxt_direction(self, direction, start, end, split):
        #0 goes down
        #1 goes up
        #-1 end
        if direction == 0:
            if end < len(self.nodes)-1 and self.__check_tiling(self.nodes[end+1],split):
                return 0
            elif start > 0 and self.__check_tiling(self.nodes[start-1],split):
                return 1
        else:
            if start > 0 and self.__check_tiling(self.nodes[start-1],split):
                return 1
            elif end < len(self.nodes)-1 and self.__check_tiling(self.nodes[end+1],split):
                return -1
        return -1

    def __get_sg_node(self):
        start = -1
        end = -1
        for i, node in enumerate(self.nodes):
            if self.btn_node[0] == node[0]:
                start = i
                end = i
                break
        return start,end

    def __get_parallel_range(self, sg_idx):
        start = -1
        end = -1
        for i, node in enumerate(self.nodes):
            if self.btn_node[0] == node[0]:
                # if self.__check_tiling(node, 0):
                if node[0] not in self.tuned_set:
                    start = i
                    end = i
                    break
        

        # for element in self.btn_node:
        #     logger.info(str(element) + " ")

        level_idx = self.btn_node[5]


        start = -1
        end = -1
        # logger.info("try to find the sg")
        found = False
        for i, node in enumerate(self.nodes):
            if node[5] == level_idx and node[4] == sg_idx and node[0] not in self.tuned_set:
                end = i
                if not found:
                    found = True
                    start = i
                # logger.info(node[0] + " " + str(node[4]) + " " + str(node[1]) + " update start " + str(start) + " end " + str(end))
            else:
                if found:
                    break
                else:
                    found = False
        return start,end


    def __get_exe_sg(self, level_idx):
        sgs = []
        first_sg = self.btn_node[4]
        for node in self.nodes:
            if node[5] == level_idx and node[1] > 0 and node[4] not in sgs:
                if node[4] == first_sg:
                    sgs.insert(0,node[4])
                else:
                    sgs.append(node[4])
        return sgs

    def __check_tiling(self, node, split):
        tiling = node[2]
        if split == 0:
            if tiling == 0:
                return True
            else:
                return False
        elif split == 1 or split == 4: #batch axis = 0
            axis = 0
        elif split == 2 or split == 5: #h
            axis = 1
        elif split == 3 or split == 6: #c
            axis = 3
        else:
            logger.error("error")

        if (tiling >> axis) & 0x1 > 0:
            return True
        else:
            return False

    def __find_continue_split_nodes(self, sg, cur_start, cur_end, axis):
        if cur_end > len(sg):
            cur_end = len(sg)

        start = -1
        end = cur_end
        for i, item in enumerate(sg[cur_start:cur_end]):
            layer, load_cyc, tiling, sg_name, sg_idx, level_idx, sgs, gm = item
            if ((tiling >> axis) & 0x1) and (start == -1):
                start = i

            if not ((tiling >> axis) & 0x1) and (start != -1) and (end == cur_end - 1):
                end = i-1
            # if end != num - 1:
            #     break
        return start, end
    def __gen_prior_in_gm(self, parallel, direction, split, num, start, end):
        new_prior_in_gm = self.prior_in_gm.copy()
        for i, layer in enumerate(self.nodes):
            if i >= start and i <= end:
                value = parallel & 0xFF
                logger.info("layer: " + layer[3] + " value " + str(value) + " parallel " + str(parallel))
                new_prior_in_gm[layer[3]] = value
        return new_prior_in_gm

    def __gen_strategy(self, parallel, direction, split, num, start, end):
        # logger.info("gen new strategy: " + str(start) + " " + str(end) + " " + str(parallel) + " " + str(direction))
        new_strategy = self.strategy.copy()

        # logger.info("sg " + str(sg) + " sg_id " + str(sgs[sg]))
        if parallel == 1:
            if direction == 0:
                logger.info("parallel")
                real_num = split << 3 | num
                for i, layer in enumerate(self.nodes):
                    if i >= start and i <= end:
                        value = ((real_num & 0xFF) << 8) | ((parallel & 0xFF) << 16)
                        # logger.info("layer: " + layer[0] + " value " + str(value) + " parallel " + str(parallel))
                        new_strategy[layer[0]] = value
        else:
            for i, layer in enumerate(self.nodes):
                if i >= start and i <= end:
                    if i == start:
                        value = ((0x1 << 31) | (num & 0xFF) << 8) | ((split & 0xFF))
                    else:
                        value = ((num & 0xFF) << 8) | ((split & 0xFF))
                    # logger.info("layer: " + layer[0] + " value " + str(value) + " split " + str(split) + " num " + str(num))
                    new_strategy[layer[0]] = value

        return new_strategy

    def __get_parallel_layers(self, start, end):
        layers = []
        node = self.nodes[start]
        sg_idx = node[4]
        level_idx = node[5]
        for i, layer in enumerate(self.nodes):
            if layer[4] != sg_idx and layer[5] == level_idx:
                layers.append(layer)
        return layers


    def __get_index(self):
        for i, node in enumerate(self.nodes):
           if self.btn_node[0] == node[0]:
                return i

        return -1

    def __get_tiling_range(self,axis):
        # logger.info("len " + str(len(self.nodes)))
        start = -1
        end = -1
        for i, node in enumerate(self.nodes):
            # logger.info("node " + str(node))
            # logger.info("self.btn_node " + self.btn_node[0])
            if self.btn_node[0] == node[0] and (node[2] >> axis) & 0x1 > 0:
                start = i
                end = i

        return start,end


class PGO(object):
    # def __init__(self):
    #     self.folder = "strategy_dump"
    #     self.data_load = False
    #     self.strategy_dump = False

    def case(self, strategy_file):
        logger.info("PGO case")

        driver_instance = Driver()
        data, cycle = driver_instance.get_data(
            "data_init.json", strategy_file, "", get_config("option"))
        logger.info("total cycle: " + str(cycle))

        with open("data_init.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            logger.info("data saved to " + "data_init.json")

        display_data(data)

        data_filename = "data.json"
        with open(data_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
            logger.info("data saved to " + data_filename)

    def mcts(self):
        logger.info("mcts start")
        dump_folder = "strategy_dump"
        if get_config("strategy_dump"):
            try:
                os.makedirs(dump_folder, exist_ok=True)
                logger.info(dump_folder + " created successfully")
            except Exception as e:
                logger.error(e)

        logger.info("folder " + dump_folder)
        driver_instance = Driver()
        driver_instance.reset()

        btn_node = None
        # tuned_list = {}
        strategy = {}
        strategy_file = "tune_strategy_tmp.json"

        prior_in_gm = {}
        prior_in_gm_file = "prior_in_gm.json"

        start_time = time.time()
        data, cycle = driver_instance.get_data("data_init.json", "", "", "--load_balance_strategy -1")
        shutil.move("tune_strategy.json", strategy_file)
        end_time = time.time()

        # display_data(data)
        # tuned_list = self.__init_tuned_list(data)
        logger.info("init cycle " + str(cycle))

        parallel = {}

        # sg_list = self.__init_sg_list(data)

        levels = self.__init_levels(strategy_file, data, parallel)
        parallel = self.__update_sg_parallel(levels)
        # for i, level in enumerate(levels):
        #     logger.info("level " + str(i))
        #     for sg_id, layers in level:
        #         logger.info("--SG " + str(sg_id))
        #         last_sg_name = ""
        #         for layer_name, load_cyc, tiling, sg_name, sg_idx, level_idx, sgs, gm in layers:
        #             if sg_name == last_sg_name:
        #                 load_cyc = 0
        #             logger.info("--*-- " + sg_name + " " + layer_name + " " + str(load_cyc) + " " + str(tiling))
        #             last_sg_name = sg_name
        # logger.info(" ")

        # sys.exit()
        


        data, cycle = driver_instance.get_data(strategy_file, "", "", "--single_core_opt --branch_reorder_strategy -1")
        levels = self.__init_levels(strategy_file, data, parallel)
        btn_nodes = self.__init_btn_nodes(levels)
        exe_nodes = self.__init_exe_nodes(levels)#need more investigate



        if get_config("preview"):
            driver_instance.skip_run()

        
        # btn_node = self.__get_bottleneck_node(data, tuned_list)
        tuned_set = set()
        tuned_set_1 = set()
        content = ""
        loop = get_config("loop")
        logger.info("will try " + str(loop) + " loop")
        # for i in range(loop):
        i = 0

        # again = False
        phase = 0
        while True:
            if phase == 2:
                btn_node = self.__find_btn_gm(btn_nodes, strategy_file, tuned_set)
            else:
                btn_node = self.__find_btn(btn_nodes, tuned_set, phase)

            if btn_node == None or i >= loop:
                logger.info("no more btn node or max loop reached")
                if phase == 0:
                    tuned_set = set()
                    logger.info("phase 0 optimized cycle " + str(cycle) + " change to phase 1")
                    i = 0
                    phase = 1
                    continue
                elif phase == 1:
                    tuned_set = set()
                    logger.info("phase 1 optimized cycle " + str(cycle) + " change to phase 2")
                    i = 0
                    phase = 2
                    parallel = {}
                    levels = self.__init_levels(strategy_file, data, parallel)
                    btn_nodes = self.__init_btn_nodes(levels)
                    exe_nodes = self.__init_exe_nodes(levels)
                    continue
                else: #phase = 2
                    logger.info("phase 2 optimized cycle " + str(cycle))
                    break
                    
            # if phase == 0 and btn_node[2] == 0:
            #     logger.info(btn_node[0] + "skip for phase 0")
            #     tuned_set.add(btn_node[0])
            #     continue
            # if phase == 1 and btn_node[2] != 0:
            #     logger.info(btn_node[0] + "skip for phase 1")
            #     tuned_set.add(btn_node[0])
            #     continue
        
            i += 1
            logger.info("------------------phase " + str(phase) + " start handle node " + str(i) + " " + btn_node[0] + " ---------------------")
            driver_instance.try_idx = i
            init_state = GB_State(data, cycle, -1, strategy, strategy_file, prior_in_gm, prior_in_gm_file, exe_nodes, btn_node, tuned_set, driver_instance, 0, i, 0, phase, DEFAULT_CHANCE)

            mcts = MCTS(init_state)
            logger.info("init mcts")
            
            mcts.search()

            # if self.arg_map["preview"]:
            #     state_num = mcts.get_state_num()
            #     level_state[i] = state_num
            # else:
            best_state = mcts.get_best_state()
            logger.info("best_state")
            cycle = best_state.cycle
            data = best_state.data
            logger.info("best cycle " + str(best_state.get_reward()))
            logger.info("best strategy " + str(best_state.strategy))
            logger.info("best strategy file " + best_state.strategy_file)

            keep_files = {best_state.strategy_file, "tune_strategy.json", "tune_strategy_tmp.json"}
            for file_name in glob.glob("tune_strategy*"):
                if os.path.isfile(file_name):
                    if file_name not in keep_files:
                        base_name = os.path.basename(file_name)
                        if base_name not in keep_files:
                            os.remove(file_name)

            strategy = best_state.strategy
            strategy_file = best_state.strategy_file
            if phase == 2:
                tuned_set.add(btn_node[0])
                prior_in_gm = best_state.prior_in_gm
                prior_in_gm_file = best_state.prior_in_gm_file
                logger.info("best prior_in_gm " + str(best_state.prior_in_gm))
                logger.info("best prior_in_gm file " + best_state.prior_in_gm_file)

                keep_files = {best_state.prior_in_gm_file}
                for file_name in glob.glob("prior_in_gm*"):
                    if os.path.isfile(file_name):
                        if file_name not in keep_files:
                            base_name = os.path.basename(file_name)
                            if base_name not in keep_files:
                                os.remove(file_name)
            else:
                self.__update_tuned_set(tuned_set, strategy, btn_node[0], phase)
            logger.info("updated tuned list " + str(tuned_set))
            logger.info("------------------end handle node " + str(i) + " " + btn_node[0] + " ---------------------")

        if get_config("preview"):
            logger.info("PGO preview done")

            profile_data = data["profile_data"]
            subgraphs = profile_data["subgraphs"]
            last_level = 0
            layer_cyc = 0
            for subgraph in subgraphs:
                level = subgraph["Level"]

                if level > last_level:
                    last_level = level
                    logger.info("level: " + str(level))
                    layer_cyc = 0
                nodes = subgraph["nodes"]
                for node in nodes:
                    n_cycle = node["Cycles"]
                    layer_cyc = layer_cyc + n_cycle
                logger.info("sg nodes: " + str(len(nodes)) +
                            " cycle: " + str(n_cycle))
                # if level in level_state:
                #     est_time = level_state[level] * (end_time - start_time)
                #     logger.info("level: " + str(level) + " will try strategy: " + str(
                #         level_state[level]) + " estimated tuning time: " + str(est_time))
                # else:
                #     logger.info("level: " + str(level) +
                #                 " will have no tuning")
        else:
            content = "best cycle: " + str(cycle)
            fp_rd = data["profile_data"]["AIPU Read Bytes"]
            fp_wr = data["profile_data"]["AIPU Write Bytes"]
            content += f", AIPU Read Bytes: {fp_rd}, AIPU Write Bytes: {fp_wr}"
            driver_instance.gen_csv()
            aipu = driver_instance.run_gb(strategy_file, prior_in_gm_file, "--load_balance_strategy 0 --branch_reorder_strategy -1 --closer_sg_opt")
            if os.path.exists(aipu):
                new_file = get_config("aipu")
                new_dir = new_file.strip('aipu.bin')
                if new_dir and not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                    shutil.copy(aipu, new_file)
                else:
                    shutil.move(aipu, new_file)
                logger.info("best aipu bin file: " + new_file)
                content = content + " aipubin: " + new_file
                with open('best_perf.json', 'w') as f:
                    json.dump(data, f)
            else:
                logger.error("aipu.bin doesn't exist")
            # logger.info("tuned list " + str(tuned_list))
            logger.info("PGO done")
            get_config("output")

            with open(get_config("output"), 'w') as f:
                f.write(content)

        return 0

    # def __get_btn_node(self, sg):
    #     max_cycle = 0
    #     max_node = None
    #     for key in sg:
    #         cycle = sg[key]
    #         if cycle > max_cycle:
    #             max_cycle = cycle
    #             max_node = key
    #     return max_node

    # def __get_bottleneck_node(self, data, tuned_list):
    #     self.btn_idx += 1
    #     profile_data = data["profile_data"]
    #     sgs = profile_data["subgraphs"]
    #     max_cycle = 0
    #     max_node = None
    #     for sg in sgs:
    #         nodes = sg["nodes"]
    #         for node in nodes:
    #             layer_name = node["Layer Name"]
    #             cycle = node["Cycles"]
    #             if layer_name in tuned_list and tuned_list[layer_name] == -1 and cycle > max_cycle:
    #                 max_cycle = cycle
    #                 max_node = layer_name
    #             # logger.info("layer " + layer_name + " " + str(cycle))
    #     return max_node

    def __find_btn(self, btn_nodes, tuned_set, phase):
        for node in btn_nodes:
            if node[0] not in tuned_set:
                if phase == 0 and node[2] > 0:
                    return node
                if phase == 1 and node[6] > 1:
                    return node
        return None
            
    def __find_btn_gm(self, btn_nodes, strategy_file, tuned_set):
        with open(strategy_file, 'r', encoding='utf-8') as f:
            tune_init = json.load(f)
            if tune_init is not None:
                for node in btn_nodes:
                    if node[0] not in tuned_set:
                        for layer_name, info in tune_init.items():
                            gm = info['gm']
                            if node[0] == layer_name and gm == 1:
                                return node
        return None
            


    def __get_sg_from_node(self, data, node):
        profile_data = data["profile_data"]
        sgs = profile_data["subgraphs"]
        is_sg = False
        btn_sg = []
        for sg in sgs:
            nodes = sg["nodes"]
            is_sg = False
            for n in nodes:
                btn_sg.append(n["Layer Name"])
                if n["Layer Name"] == node:
                    is_sg = True
            if is_sg:
                break
            else:
                btn_sg.clear()
        return btn_sg

    def __init_levels(self, tune_strategy_file, data, parallel):
        levels = []
        logger.info("init levels")
        node_map = {}
        if data is not None:
            profile_data = data["profile_data"]
            subgraphs = profile_data["subgraphs"]
            last_level = 0
            level_core = []
            for subgraph in subgraphs:
                nodes = subgraph["nodes"]
                for node in nodes:
                    n_cycle = node["Cycles"]
                    n_name = node["Layer Name"]
                    node_map[n_name] = n_cycle

        # for name, cycle in node_map.items():
            # logger.info("layer " + name + " cycle " + str(cycle))

        with open(tune_strategy_file, 'r', encoding='utf-8') as f:
            tune_init = json.load(f)
            if tune_init is not None:
                for layer_name, info in tune_init.items():
                    level_id = info['level']
                    sg_id = info['sg']
                    tiling = info['tiling']
                    sg_name = info['sg_name']
                    gm = info['gm']
                    load_cyc = 0
                    if sg_name in node_map:
                        load_cyc = node_map[sg_name]
                    # else:
                    #     logger.info("empty load " + layer_name)
                    logger.info(layer_name + " id " + str(sg_id))
                    if not parallel:
                        sg_idx = -1
                        level_idx = -1
                        sgs = -1
                    else:
                        if layer_name in parallel:
                            sg_idx, level_idx, sgs = parallel[layer_name]
                            # logger.info("parallel " + layer_name + " " + str(sgs))
                        else:
                            logger.error("no " + layer_name)
                            sg_idx = -1
                            level_idx = -1
                            sgs = -1
                            sys.exit()

                    # 如果 level_id 超出当前 levels 长度，扩展它
                    while len(levels) <= level_id:
                        # 每个 level 是一个 list of (sg_id, layers)
                        levels.append([])

                    current_level = levels[level_id]

                    # 查看是否已经有这个 sg_id 存在
                    found = False
                    for sg_group in current_level:
                        if sg_group[0] == sg_id:
                            sg_group[1].append(
                                (layer_name, load_cyc, tiling, sg_name, sg_idx, level_idx, sgs, gm))
                            found = True
                            break

                    # 如果没有，新建一个 sg 组
                    if not found:
                        current_level.append(
                            [sg_id, [(layer_name, load_cyc, tiling, sg_name, sg_idx, level_idx, sgs, gm)]])

        for level in levels:
            level.sort(
                key=lambda sg_group: sum(
                    layer_info[1] for layer_info in sg_group[1]),
                reverse=True
            )

        return levels


    def __init_btn_nodes(self, levels):
        nodes = []
        for i, level in enumerate(levels):
            logger.info("level " + str(i))
            for sg_id, layers in level:
                logger.info("--SG " + str(sg_id))
                for layer, load_cyc, tiling, sg_name, sg_idx, level_idx, sgs, gm in layers:
                    logger.info("--*-- " + sg_name + " " + layer + " " + str(load_cyc) + " " + str(tiling) + " " + str(sg_idx) + " " + str(level_idx) + " " + str(sgs))
                    nodes.append((layer, load_cyc, tiling, sg_name, sg_idx, level_idx, sgs, gm))
        
        
        nodes.sort(key=lambda x: x[1], reverse=True)
        logger.info(" after sorted")
        for layer, load_cyc, tiling, sg_name, sg_idx, level_idx, sgs, gm in nodes:
            logger.info(layer + " " + str(load_cyc) + " " + str(tiling) + " level: " + str(level_idx))

        return nodes
        
    def __init_exe_nodes(self, levels):
        level_map = {}
        for i, level in enumerate(levels):
            for sg_id, layers in level:
                for item in layers:
                    layer, load_cyc, tiling, sg_name, sg_idx, level_idx, sgs, gm = item
                    if level_idx not in level_map:
                        level_map[level_idx] = []
                    level_map[level_idx].append(item)
                    
        merged_list = [item for sublist in level_map.values() for item in sublist]



        for layer, load_cyc, tiling, sg_name, sg_idx, level_idx, sgs, gm in merged_list:
            logger.info("--*-- " + layer + " " + str(load_cyc) + " " + str(tiling) + " " + str(sg_idx) + " " + str(level_idx))

        return merged_list


    def __update_tuned_set(self, tuned_set, strategy, layer_name, phase):
        tuned_set.add(layer_name)
        if phase == 1:
            for key in strategy:
                value = strategy[key]
                parallel = (value >> 16) & 0xFF
                if parallel != 0:
                    tuned_set.add(key)
                

    def __update_non_tuned_set(self, tuned_set, tuned_set_1):
        non_tuned_set = set()
        for key in tuned_set:
            if key not in tuned_set_1:
                non_tuned_set.add(key)
        return non_tuned_set

    def __get_max_load(self, level):
        max_load = 0
        for sg_id, sg in level:
            load = 0
            for layer, load_cyc, tiling, sg_name, sg_idx, level_idx, sgs, gm in sg:
                load = load + load_cyc
            max_load = max(max_load, load)

        return max_load

    def __update_sg_parallel(self,levels):
        # logger.info("update sg parallel")
        parallel = {}
        level_group = {}
        for i, level in enumerate(levels):
            sg_s = set()
            for sg_id, layers in level:
                # for layer, load_cyc, tiling, sg_name, sg_idx, level_idx, sgs, gm in layers:
                # logger.info(str(i) + " add id " + str(sg_id))
                sg_s.add(sg_id)
            # logger.info("level " + str(i) + " has sgs " + str(len(sg_s)))
            level_group[i] = len(sg_s)


        for i, level in enumerate(levels):
            for sg_id, layers in level:
                for layer, load_cyc, tiling, sg_name, sg_idx, slevel_idx, sgs, gm in layers:
                    # logger.info(layer + " " + str(sg_id) + " level " + str(i) + " has sg " + str(level_group[i]))
                    parallel[layer] = sg_id, i, level_group[i]
        return parallel



def display_data(data):
    if data is not None:
        logger.info("display profiler")
        logger.info("")
        profile_data = data["profile_data"]
        subgraphs = profile_data["subgraphs"]
        last_level = 0
        level_core = [0] * CORE_NUM
        for subgraph in subgraphs:
            nodes = subgraph["nodes"]
            for node in nodes:
                n_cycle = node["Cycles"]
                n_name = node["Layer Name"]
                level = node["Level"]
                core = node["core_id"]
                task = node["Task ID"]
                mem_str = node["Output Memory"]
                mems = re.split(r'\]\s*,\s*\[', mem_str)
                shape_str = node["Output Shapes"]
                shapes = re.split(r'\]\s*,\s*\[', shape_str)

                if level > last_level:
                    if len(level_core) > 0:
                        max_value = max(level_core)
                        logger.info("level: " + str(last_level) +
                                    " cycle " + str(max_value))
                        logger.info("")
                        logger.info("level: " + str(level))
                        level_core = [0] * CORE_NUM
                
                level_core[core] = level_core[core] + n_cycle
                logger.info("core" + str(core) + " task" + str(task) + " " + n_name + " " + str(n_cycle) + " " + mems[0] + " " + shapes[0])
                last_level = level
        if len(level_core) > 0:
            max_value = max(level_core)
            logger.info("level: " + str(last_level) +
                        " cycle " + str(max_value))
            logger.info("")

