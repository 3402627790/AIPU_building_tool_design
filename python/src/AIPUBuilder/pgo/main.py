import argparse
import configparser
import time
import subprocess
import multiprocessing
import datetime
import os


from AIPUBuilder.driver import *  # noqa
from AIPUBuilder.pgo import PGO
from .configs import *

def main():
    parser = argparse.ArgumentParser(description="Print a message based on configuration.")
    parser.add_argument('-c', '--config-file', type=str, required=True, help="Path to the configuration file.")
    
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)
    settings = config['Settings']
    loop = settings.getint('loop', fallback=1)

    ir = settings.get('ir', fallback='default')
    wt = settings.get('wt', fallback='default')
    inputs = settings.get('inputs', fallback='default')
    target = settings.get('target', fallback='default')
    gb = settings.get('gb', fallback='default')
    strategy_dump = settings.getboolean('strategy_dump', fallback=False)
    data_load = settings.getboolean('data_load', fallback=False)
    data_dump = settings.getboolean('data_dump', fallback=False)
    iteration = settings.getint('iteration', fallback=1)
    option = settings.get('option', fallback='')
    strategy_file = settings.get('strategy_file', fallback='')
    mcts = settings.getboolean('mcts', fallback=False)
    output = settings.get('output', fallback='default')
    preview = settings.getboolean('preview', fallback=False)
    aipu = settings.get('aipu', fallback='default')
    # dumper = settings.get('dumper',fallback='default')
    # sim = settings.get('sim',fallback='default')
    # profiler = settings.get('profiler',fallback='default')
    # schedule = settings.get('schedule',fallback='default')
    arg_map = {}
    arg_map["ir"] = ir
    arg_map["wt"] = wt
    arg_map["inputs"] = inputs
    arg_map["target"] = target
    arg_map["gb"] = gb
    arg_map["loop"] = loop
    arg_map["strategy_dump"] = strategy_dump
    arg_map["data_load"] = data_load
    arg_map["data_dump"] = data_dump
    arg_map["iteration"] = iteration
    arg_map["option"] = option
    arg_map["output"] = output
    arg_map["preview"] = preview
    arg_map["aipu"] = aipu


    set_configs(arg_map)
    pgo = PGO()

    if mcts:

        now_time = datetime.datetime.now()
        formatted_time = now_time.strftime("%Y-%m-%d %H:%M:%S")
        with open(output, 'w') as f:
            f.write(" time: " + str(formatted_time) + '\n')

        start_time = time.time()
        pgo.mcts()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Program time: {elapsed_time:.2f} s")
        with open(output, 'a') as f:
            f.write(" time: " + str(elapsed_time) + '\n')
    else:
        pgo.case(strategy_file)

    return

def check_aipu_available():
    with multiprocessing.Manager() as manager:
        result = manager.dict()
        p = multiprocessing.Process(target=_worker_check_status, args=(result,))
        p.start()
        p.join()
        return result.get('status')

def _worker_check_status(result):
    try:
        status = get_device_status()
        result['status'] = status
    except Exception as e:
        result['error'] = str(e)


if __name__ == "__main__":
    # subprocess.run("lsmod|grep aipu", check=True, shell=True)
    status = check_aipu_available()
    # subprocess.run("lsmod|grep aipu", check=True, shell=True)
    if status == 0:
        main()
    else:
        print("Device not ready")

    # print(get_core_number())
    # print(get_device_version())
    # print(get_cur_freq())
