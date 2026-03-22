import sys
import subprocess
import shelve
import struct
import argparse
import os
import tempfile
import time
import random
import re

bin_aipurun = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../../build/tools/aipurun")
bin_aipugb = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../../build/tools/aipugb")
bin_aipudumper = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "../../build/tools/aipudumper")


def run_cmd(cmd, wkdir, case_name, log_name, out_info):
    log_file = log_name
    log = open(log_file, "w+")
    log.write(" ".join(cmd) + "\n")
    retcode = subprocess.call(cmd, cwd=wkdir, stdout=log, stderr=log)

    if retcode != 0:
        print(
            f'\n{case_name.ljust(50)} run {out_info} error!! log: file://{log_file}')
        log.close()
        sys.exit(-1)
    else:
        log.close()


def verify(bsample, bref, out):
    success = True
    maxe = 0
    with open(out, "w") as o:
        o.write("offset\tsample\treference\tdifference\n")
        sample = struct.unpack("B" * len(bsample), bsample)
        ref = struct.unpack("B" * len(bref), bref)
        e_cnt = 0
        for i in range(min(len(sample), len(ref))):
            e = abs(sample[i] - ref[i])

            if e > 3:
                e_cnt += 1
                maxe = max(maxe, e)
                success = False

            if i < 20 or e > 0:  # print head & fist 100 error
                o.write("%#x\t%d\t%d\t%d\n" %
                        (i, sample[i], ref[i], abs(sample[i] - ref[i])))

            if e_cnt > 100:  # too many error
                o.write("...")
                break

        maxe = 0
        if not success:
            # try int8
            o.write("compare in int8\n")
            success = True
            sample = struct.unpack("b" * len(bsample), bsample)
            ref = struct.unpack("b" * len(bref), bref)
            e_cnt = 0
            for i in range(min(len(sample), len(ref))):
                e = abs(sample[i] - ref[i])

                if e > 3:
                    e_cnt += 1
                    maxe = max(maxe, e)
                    success = False

                if i < 20 or e > 0:  # print head & fist 100 error
                    o.write("%#x\t%d\t%d\t%d\n" %
                            (i, sample[i], ref[i], abs(sample[i] - ref[i])))

                if e_cnt > 100:  # too many error
                    o.write("...")
                    break

    return success, maxe


def compare(ref, output, log_name):
    ref_infos = []
    log_file = log_name
    log = open(log_file, "w+")
    with open(ref, "rb") as ref_file:
        output_cnt = struct.unpack('i', ref_file.read(4))[0]
        log.write("%d output(s) in this model" % output_cnt)
        for i in range(0, output_cnt):
            ref_infos.append({
                'index': struct.unpack('i', ref_file.read(4))[0],
                'offset': struct.unpack('i', ref_file.read(4))[0],
                'size': struct.unpack('i', ref_file.read(4))[0]
            })

    for ref_info in ref_infos:
        log.write("Verifing output %s" % ref)

        # output.bin, output.bin1, output.bin2...
        if (0 == ref_info['index']):
            output_data_path = output
        else:
            output_data_path = output + str(ref_info['index'])

        log.write("(%s-%s)" % (ref, output_data_path))
        with open(ref, "rb") as ref_file:
            ref_file.seek(ref_info['offset'])
            bref = ref_file.read(ref_info['size'])
            with open(output_data_path, "rb") as fout:
                bsample = fout.read()

                diff_log_path = log_path.replace(
                    ".txt", "_output_%s_diff_log.txt" % ref_info['index'])
                ok, maxe = verify(bsample, bref, diff_log_path)
                if not ok:
                    print(
                        f'\n{case_name.ljust(50)} mismatch, log: file://{diff_log_path}')
                    sys.exit(-1)
    log.close()


def run_builder(graph,
                weight=None,
                target="Z1_0904",
                log=None,
                check_tiling=False,
                check_ds=False,
                check_wt_split=False,
                check_ocm=False,
                wkdir=None):
    case_name = f"{os.path.basename(graph)}[{target}]"

    cmd = [bin_aipugb, graph]
    if weight is not None:
        cmd.append("-w")
        cmd.append(weight)
    cmd.append("-o")
    output = os.path.join(wkdir, str(time.time()) + "_aipu.bin")
    cmd.append(output)
    cmd.append("--target")
    cmd.append(target)
    # cmd.append("-D")
    # cmd.append("--disable-qformat")
    # cmd.append("--new-section")
    # cmd.append("--new-interface")
    # print(" ".join(cmd))

    if check_tiling:
        cmd.append("--tiling")
        cmd.append("fps")

    if check_ds:
        cmd.append("--enable_ds")

    if check_wt_split:
        cmd.append("--max_wt_size")
        cmd.append("3")

    if check_ocm:
        cmd.append("--ocm_addr")
        cmd.append("0xa0017000")
        cmd.append("--ocm_size")
        cmd.append("0x9000")

    log_file = "run_log.txt"
    if log is None:
        log = open("run_log.txt", "w")
    else:
        log_file = log
        log = open(log, "a")

    log.write(" ".join(cmd) + "\n")
    # pid = subprocess.Popen(cmd, stdout=log, stderr=log)
    retcode = subprocess.call(cmd, cwd=wkdir, stdout=log, stderr=log)
    # retcode = pid.wait()
    if retcode != 0:
        print(f'\n{case_name.ljust(50)} aipugb error!! log: file://{log_file}')
        log.close()
        # with open(log_file, "r") as f:
        #     print(f.read())
        sys.exit(-1)
    log.close()
    return output


def run_simulator(ref,
                  graph,
                  input,
                  weight=None,
                  target="Z1_0904",
                  log=None,
                  simulator=None,
                  check_output=True,
                  check_tiling=False,
                  check_ds=False,
                  input_shape=None,
                  max_input_shape=None,
                  check_wt_split=False,
                  check_ocm=False,
                  wkdir=None):
    case_name = f"{os.path.basename(graph)}[{target}]"

    cmd = [bin_aipurun, graph, "-i", input]
    if weight is not None:
        cmd.append("-w")
        cmd.append(weight)
    cmd.append("-o")
    output = os.path.join(wkdir, str(time.time()) + target + "_output.bin")
    cmd.append(output)
    cmd.append("--target")
    cmd.append(target)

    # due to -p will gen many files will degrade sanity test performance, we only run with X3 single core on AIB
    if "X3" in target and "MP" not in target and "ai_benchmark" in graph:
        cmd.append("-p")  # add -p by default to get cost model cycles

    # avoid mismatch
    if "ai_benchmark" in graph:
        cmd.append("--op_plugin_options")
        cmd.append("AIPU_32BIT_BIAS=1,AIPU_AVG_POOL_16BIT_SCALE=1")

    if check_tiling:
        cmd.append("--tiling")
        cmd.append("fps")

    if check_ds:
        cmd.append("--enable_ds")
        if input_shape is not None:
            cmd.append("--input_shape")
            cmd.append(input_shape)
        if max_input_shape is not None:
            cmd.append("--max_input_shape")
            cmd.append(max_input_shape)

    if check_wt_split:
        cmd.append("--max_wt_size")
        cmd.append("3")

    if check_ocm:
        cmd.append("--ocm_addr")
        cmd.append("0xa0017000")
        cmd.append("--ocm_size")
        cmd.append("0x9000")

    log_file = "run_log.txt"
    if log is None:
        log = open("run_log.txt", "w")
    else:
        log_file = log
        log = open(log, "a")
    if simulator is not None:
        cmd.append("--simulator")
        cmd.append(simulator)
    log.write(" ".join(cmd) + "\n")
    # pid = subprocess.Popen(cmd, stdout=log, stderr=log)
    retcode = subprocess.call(cmd, cwd=wkdir, stdout=log, stderr=log)
    # retcode = pid.wait()
    if retcode != 0:
        print(f'\n{case_name.ljust(50)} aipurun error!! log: file://{log_file}')
        log.close()
        # with open(log_file, "r") as f:
        #     print(f.read())
        sys.exit(-1)

    # Parser ref_output bin
    if check_output:
        ref_infos = []
        with open(ref, "rb") as ref_file:
            output_cnt = struct.unpack('i', ref_file.read(4))[0]
            log.write("%d output(s) in this model" % output_cnt)
            for i in range(0, output_cnt):
                ref_infos.append({
                    'index': struct.unpack('i', ref_file.read(4))[0],
                    'offset': struct.unpack('i', ref_file.read(4))[0],
                    'size': struct.unpack('i', ref_file.read(4))[0]
                })

        for ref_info in ref_infos:
            log.write("Verifing output %s" % ref)

            # output.bin, output.bin1, output.bin2...
            if (0 == ref_info['index']):
                output_data_path = output
            else:
                output_data_path = output + str(ref_info['index'])

            log.write("(%s-%s)" % (ref, output_data_path))
            with open(ref, "rb") as ref_file:
                ref_file.seek(ref_info['offset'])
                bref = ref_file.read(ref_info['size'])
                with open(output_data_path, "rb") as fout:
                    bsample = fout.read()

                    diff_log_path = log_file.replace(
                        ".txt", "_output_%s_diff_log.txt" % ref_info['index'])
                    ok, maxe = verify(bsample, bref, diff_log_path)
                    if not ok:
                        print(
                            f'\n{case_name.ljust(50)} mismatch, log: file://{diff_log_path}')
                        # print(graph,"max error is ",maxe)
                        # with open(log_file,"r") as f:
                        #     print(f.read())
                        # print(diff_log_path)
                        # with open(diff_log_path,"r") as f:
                        #     print(f.read())
                        sys.exit(-1)
    else:
        log.write("skipped output binary check")

    # clean up if ok
    os.remove(output)

    # sys.exit(0)


def run_aipudumper(graph,
                   input,
                   output,
                   target="Z1_0904",
                   simulator=None,
                   log=None,
                   ref=None,
                   wkdir=None):
    case_name = f"{os.path.basename(graph)}[{target}]"

    # Step1  generate aipudumper addr.ini
    cmd = [bin_aipudumper, output, "-i", input,
           "--dump_addr_cfg", "--ddr_data_gather"]

    core_map = {
        'MP2': '2',
        'MP3': '3',
        'MP4': '4'
    }
    multi_core = next((v for k, v in core_map.items() if k in target), '')
    if multi_core:
        cmd.append("--multi_core")
        cmd.append(multi_core)

    target_dumper_path = os.path.join(wkdir, target)
    if not os.path.exists(target_dumper_path):

        os.makedirs(target_dumper_path)
        print(f"Created directory: {target_dumper_path}")
    else:
        print(f"Directory already exists: {target_dumper_path}")

    log_file = f"{case_name}_aipudumper_generate_addr_log.txt"
    run_cmd(cmd, target_dumper_path, case_name, log_file, f"{case_name}_Step1_generate aipudumper addr.ini")

    # Step2  modify addr_offset_cfg
    offset_shell_path = os.path.join(os.path.abspath(os.path.join((os.path.dirname(
        __file__)), "../../../")), "AIPU_common/top_test/end_to_end_test/addr_cfg_offset.sh")
    addr_cfg = os.path.join(wkdir, target, 'addr_cfg_default.ini')

    offset_base_ini_asid0 = 0  # set asid0 addr to 0x0
    offset_base_ini_hex_asid0 = hex(offset_base_ini_asid0 * 4)
    offset_base_ini_asid1 = random.randint(0,
                                           589824)  # range(0, 0x9000) align_bytes 4k 8B 32B --> range to 0x900000000
    offset_base_ini_hex_asid1 = hex(offset_base_ini_asid1 * 4 + 8388608)  # + 4G bytes 0x800000
    offset_base_ini_offset = random.randint(0,
                                            65536)  # range(0, 0x10000) align_bytes 4k 8B 32B --> range to 0x40000000
    if 'Z1' in target:
        offset_base_ini_offset = random.randint(0,
                                                16) * 4096  # range(0, 0x1000) align_bytes 1M --> range to 0x40000000
    offset_base_ini_hex_offset = hex(offset_base_ini_offset * 4)
    if 'X2' in target or 'X3' in target:
        cmd = ["bash", offset_shell_path, addr_cfg, "all"]
        cmd.append(str(offset_base_ini_hex_offset) + "000")
        cmd.append(str(offset_base_ini_hex_asid0) + "000")
        cmd.append(str(offset_base_ini_hex_asid1) + "000")
    else:
        cmd = ["bash"]
        cmd.append(offset_shell_path)
        cmd.append(addr_cfg)
        cmd.append("base")
        cmd.append(str(offset_base_ini_hex_offset) + "000")

    log_file = f"{case_name}_aipudumper_modify addr_offset_cfg.txt"
    run_cmd(cmd, target_dumper_path, case_name, log_file, f"{case_name}_Step2_modify addr_offset_cfg")
    ini_file = os.path.join(wkdir, target, f"addr_cfg_offset_bac.ini")

    # Step3 use modified addr.ini to run simulator
    dumper_output = os.path.join(wkdir, str(
        time.time()) + target + "_dumper_output.bin")
    cmd = [bin_aipudumper, output, "-i", input,
           "-o", dumper_output, "--simulator", simulator]
    if multi_core:
        cmd.append("--multi_core")
        cmd.append(multi_core)
    cmd.append("--load_addr_cfg")
    cmd.append(ini_file)

    log_file = f"{case_name}_aipudumper_rerun_simulator.txt"
    run_cmd(cmd, target_dumper_path, case_name, log_file, 'Step3 use modified addr.ini to run simulator')
    compare(ref, dumper_output, f"{case_name}_step3_compare_log.txt")

    # Step4  check disable_gm parameter
    dumper_output = os.path.join(wkdir, str(time.time()) + target + "_dumper_disable_gm_output.bin")
    cmd = [bin_aipudumper, output, "-i", input,
           "-o", dumper_output, "--simulator", simulator]

    if multi_core:
        cmd.append("--multi_core")
        cmd.append(multi_core)

    cmd.append("--disable_gm")

    log_file = f"{case_name}_aipudumper_disbale_gm_simulator.txt"
    run_cmd(cmd, target_dumper_path, case_name, log_file, 'Step4  check disable_gm parameter')
    compare(ref, dumper_output, f"{case_name}_step4_compare_log.txt")

    # remove output, temp and addr.ini file
    # os.remove(output)
    # os.remove(os.path.join(wkdir, 'addr_cfg*'))
    # os.remove(os.path.join(wkdir, 'temp*'))
    # sys.exit(0)


def main():
    args = argparse.ArgumentParser()
    args.add_argument("graph", type=str)
    args.add_argument("-i", type=str)
    args.add_argument("-w", type=str, default=None)
    args.add_argument("-r", type=str)
    args.add_argument("-dumper_ref", type=str, default=None)
    args.add_argument("-log", type=str, default=None)
    args.add_argument("-s", type=str, default=None)
    args.add_argument("-t", type=str, default=None)
    args.add_argument("-wkdir", type=str, default=None)
    args.add_argument("--check_output", action="store_true")
    args.add_argument("--check_dumper", action="store_true")
    args.add_argument("--check_tiling", action="store_true")
    args.add_argument("--check_ds", action="store_true")
    args.add_argument("--input_shape", type=str, default=None)
    args.add_argument("--max_input_shape", type=str, default=None)
    args.add_argument("--check_wt_split", action="store_true")
    args.add_argument("--check_ocm", action="store_true")
    options = args.parse_args(sys.argv[1:])
    temp_dir = None
    if options.wkdir is None:
        temp_dir = tempfile.TemporaryDirectory()
        options.wkdir = os.path.abspath(temp_dir.name)
    output = run_builder(options.graph,
                         options.w,
                         options.t,
                         options.log,
                         options.check_tiling,
                         options.check_ds,
                         options.check_wt_split,
                         options.check_ocm,
                         options.wkdir)
    if options.check_output:
        run_simulator(options.r,
                      options.graph,
                      options.i,
                      options.w,
                      options.t,
                      options.log,
                      options.s,
                      options.check_output,
                      options.check_tiling,
                      options.check_ds,
                      options.input_shape,
                      options.max_input_shape,
                      options.check_wt_split,
                      options.check_ocm,
                      options.wkdir)
    if options.check_dumper:
        run_aipudumper(options.graph,
                       options.i,
                       output,
                       options.t,
                       options.s,
                       options.log,
                       options.dumper_ref,
                       options.wkdir)
    if temp_dir is not None:
        temp_dir.cleanup()
    try:
        if len(options.log):
            with open(options.log, "r") as f:
                log = f.read()
                ct = re.findall(r"compile time:\s+[\d\.]+\s+s", log)
                cm = re.findall(r"Total cycles estimated by cost model is:\s+[\d\.]+", log)
                if len(ct) != 0:
                    print(f"{ct[0]}")
                if len(cm) != 0:
                    print(f"{cm[-1]}")
    except Exception as e:
        print(str(e))


if __name__ == "__main__":
    main()
