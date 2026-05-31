import argparse
import xml.etree.ElementTree as ET
import os
import re
import struct

DIR = os.path.dirname(os.path.abspath(__file__))


def xml_to_dict(element):
    node = {}

    for key, value in element.attrib.items():
        node["@" + key] = value.strip()

    children = list(element)
    if children:
        child_data = {}
        for child in children:
            if child.tag in child_data:
                if not isinstance(child_data[child.tag], list):
                    child_data[child.tag] = [child_data[child.tag]]
                child_data[child.tag].append(xml_to_dict(child))
            else:
                child_data[child.tag] = xml_to_dict(child)

        if element.text and element.text.strip():
            node["#text"] = element.text.strip()
        node.update(child_data)
    elif element.text:
        node["#text"] = element.text

    return node


def xml_file_to_dict(path):
    xml_data = open(path, "r").read()
    root = ET.fromstring(xml_data)

    dict_data = xml_to_dict(root)
    return dict_data


class Field(object):

    def __init__(self, dic):
        self.name = dic['@name']
        self.width = int(dic['@width'])
        self.rw = dic['@rw']
        self.default = int(dic['@init'], 16)
        self.bits = [int(i) for i in re.findall("\d+", dic['@bits'])]

    def __str__(self):
        return f"{self.name}[{self.bits}]"

    def __repr__(self):
        return self.__str__()

    def to_c(self):
        bits = self.bits[0] - self.bits[1] + 1
        if bits == 32:
            return f'uint32_t {self.name};'
        return f'uint32_t {self.name} : {bits};'


class Reg(object):

    def __init__(self, dic):
        self.addr = int(dic["@addr"], 16)
        self.name = dic['@name']
        self.width = dic['@width']
        self.rw = dic['@rw']
        self.default = int(dic['@init'], 16)
        self.type = 0  # 0 for act, 1 for param, 2 for ctrl
        field = dic["FIELD"]
        if isinstance(field, dict):
            field = [field]
        self.field = sorted([Field(i) for i in field],
                            key=lambda x: x.bits[1], reverse=True)

    def __str__(self):
        return f"{self.name}@0x{self.addr:x}"

    def __repr__(self):
        return self.__str__()

    def to_c(self, indent=0):
        ind = "    " * indent
        pad = ''
        if self.field[-1].bits[1] != 0:
            pad = f'\n{ind}        uint32_t _pad : {self.field[-1].bits[1]};\n'
        fields = f'\n        {ind}'.join([i.to_c() for i in self.field]) + pad
        return f'''
{ind}union {self.name}
{ind}{{
    {ind}struct{{
        {ind}{fields}
    {ind}}} bits;
    {ind}uint32_t reg = {self.default};
{ind}}};
        '''


def walk_to_regs(regs):
    ret = dict()

    def walk(ret, regs):
        if isinstance(regs, str):
            return
        if 'REG' in regs:
            cnt = 1
            offset = 0
            if '@set_cnt' in regs:
                cnt = int(regs['@set_cnt'])
                offset = int(regs['@addr_offset'], 16)
            if isinstance(regs['REG'], dict):
                ret[regs['@name']] = Reg(regs['REG'])
                ret[f"{regs['@name']}_info"] = {"cnt": cnt, "offset": offset}
            else:
                ret[regs['@name']] = [Reg(i) for i in regs['REG']]
                ret[f"{regs['@name']}_info"] = {"cnt": cnt, "offset": offset}
            return
        if isinstance(regs, list):
            for i in regs:
                walk(ret, i)
            return
        for k in regs:
            if isinstance(regs[k], list):
                for i in regs[k]:
                    walk(ret, i)
            else:
                walk(ret, regs[k])
    walk(ret, regs)
    return ret


def flat_regs(regs):
    ret = []

    def walk(ret, regs):
        if isinstance(regs, Reg):
            if regs not in ret:
                ret.append(regs)
            return
        if isinstance(regs, list):
            for i in regs:
                walk(ret, i)
        elif isinstance(regs, dict):
            for i in regs:
                walk(ret, regs[i])
    walk(ret, regs)
    return ret


def split_act_param_ctrl(mem, g, arch):
    # split by address
    # address is in order of act,param,ctrl
    if g not in ["MTPS", "MTP", "ITPS", "ITP", "PTP", "WRB", "UNB"]:
        return mem, [], []
    # split = []
    # mem = sorted(mem, key=lambda x: x.addr)
    # last = mem[0].addr
    # for n, m in enumerate(mem):
    #     if m.addr > last + 1:
    #         split.append(n)
    #     last = m.addr
    # assert len(split) == 2
    # return mem[:split[0]], mem[split[0]:split[1]], mem[split[1]:]
    mem_table = [
        ["MTPS_ACT", 0, [None, None]],
        ["MTPS_PARAM", 1, [None, None]],
        ["MTPS_CTRL", 2, [0x170, 0x170]],

        ["MTP_ACT", 0, [0x100, 0x106]],
        ["MTP_PARAM", 1, [0x110, 0x115]],
        ["MTP_CTRL", 2, [0x120, 0x14e]],

        ["ITPS_ACT", 0, [0x200, 0x204]],
        ["ITPS_PARAM", 1, [0x210, 0x211]],
        ["ITPS_CTRL", 2, [0x220, 0x239]],

        ["ITP_ACT", 0, [0x300, 0x305]],
        ["ITP_PARAM", 1, [0x310, 0x31b]],
        ["ITP_CTRL", 2, [0x320, 0x351]],

        ["PTP_ACT", 0, [0x700, 0x705]],
        ["PTP_PARAM", 1, [0x710, 0x716]],
        ["PTP_CTRL", 2, [0x720, 0x749]],

        ["WRB_ACT", 0, [0x30, 0x35]],
        ["WRB_PARAM", 1, [None, None]],
        ["WRB_CTRL", 2, [0x40, 0x79]],

        ["UNB_ACT", 0, [None, None]],
        ["UNB_PARAM", 1, [None, None]],
        ["UNB_CTRL", 2, [0x20, 0x24]],
    ]
    mem = sorted(mem, key=lambda x: x.addr)
    act, param, ctrl = [], [], []
    for m in mem:
        for key, type, (low, high) in mem_table:
            if low is None and high is None:
                continue
            if m.addr >= low and m.addr <= high:
                if key.endswith("_ACT"):
                    act.append(m)
                elif key.endswith("_PARAM"):
                    param.append(m)
                elif key.endswith("_CTRL"):
                    ctrl.append(m)
                m.type = type
                break
    assert len(act) + len(param) + len(ctrl) == len(mem)
    return act, param, ctrl


def gen_aiff_regs_c(regs, arch, indent=0):
    '''
    it will generate a super class with all AIFF registers
    '''
    ind = "    " * indent
    aiff_mem = ''
    ret = '''//autogenerated DO NOT modify!!!!
# include <cinttypes>
# include <vector>

'''
    for g in regs:
        if g in ["SYS"]:
            continue
        if g.endswith('_info'):
            name = g.replace("_info", "")
            info = regs[g]
            if info['cnt'] == 1:
                aiff_mem += f"\t{name}Reg {name.lower()};\n"
            else:
                for i in range(info['cnt']):
                    aiff_mem += f"\t{name}Reg {name.lower()}{i};\n"
            continue
        mem = regs[g]
        if not isinstance(mem, list):
            mem = [mem]

        act, param, ctrl = split_act_param_ctrl(mem, g, arch)
        fields = f"\n{ind}    ".join(
            [f"{m.to_c(indent+1)}\n{ind}    union {m.name} {m.name.lower()};" for m in mem])
        base = 0
        if len(ctrl):
            base = min([m.addr for m in ctrl])
        ctrl_desc = f"\n{ind}       ".join(
            [f"ret[{(m.addr - base) + 1}] = {m.name.lower()}.reg;" for m in ctrl]
        )
        param_desc = f"\n{ind}       ".join(
            [f"ret[{(m.addr - param[0].addr)}] = {m.name.lower()}.reg;" for m in param]
        )
        act_desc = f"\n{ind}       ".join(
            [f"ret[{(m.addr - act[0].addr)}] = {m.name.lower()}.reg;" for m in act]
        )

        ret += f'''
{ind} struct {g}Reg
{ind} {{
    {ind}{fields}
    {ind}uint32_t base = 0;
    {ind}{g}Reg(uint32_t base = 0):base(base){{}}
    {ind}std::vector<uint32_t> gen_ctrl()
    {ind}{{
    {ind}   std::vector<uint32_t> ret({len(ctrl)+1});
    {ind}   ret[0] = {base+(len(mem)<<16)} + base;
    {ind}   {ctrl_desc}
    {ind}   return ret;
    {ind}}}
    {ind}std::vector<uint32_t> gen_param()
    {ind}{{
    {ind}   std::vector<uint32_t> ret({len(param)});
    {ind}   {param_desc}
    {ind}   return ret;
    {ind}}}
    {ind}std::vector<uint32_t> gen_act()
    {ind}{{
    {ind}   std::vector<uint32_t> ret({len(act)});
    {ind}   {act_desc}
    {ind}   return ret;
    {ind}}}
{ind} }};
        '''

    ret += f'''
struct AIFFReg
{{
{aiff_mem};
}};
    '''
    with open("aiff_reg.h", "w") as f:
        f.write(ret)
    return ret


def gen_regs_c(regs, path="regs.h"):
    with open(path, "w") as f:
        f.write("""//autogenerated DO NOT modify!!!!
# include <cinttypes>

""")
        f.write(
            "enum class Reg{\n" + ",\n".join([i.name for i in regs]) + "};\n")
        f.write(
            'constexpr const uint32_t RegValue[]={' + ",".join([str(i.default) for i in regs]) + '};\n')
        f.write(
            'constexpr const uint32_t RegAddr[]={' + ",".join([str(i.addr) for i in regs]) + '};\n')
        f.write("\n".join([i.to_c() for i in regs]))


def gen_regs_map(regs, arch, path):
    ret = '''//autogenerated DO NOT modify!!!!
# include <cinttypes>
# include <vector>
# include "base_reg.h"

namespace ffc
{
namespace HAL
{
    AIFFRegs gen_%s_regs()
    {
        return std::map<std::string, AIFFGroupRegs>({
            %s
        });
    }

}//namespace HAL
}//namespace ffc
'''
    groups = []
    for g in regs:
        if g in ["SYS"]:
            continue
        if g.endswith('_info'):
            name = g.replace("_info", "")
            info = regs[g]
            continue
        else:
            info = regs[f"{g}_info"]
        mem = regs[g]
        if not isinstance(mem, list):
            mem = [mem]

        act, param, ctrl = split_act_param_ctrl(mem, g, arch)

        def format_reg(r):
            return ",\n".join([f'{{"{i.name}", {(i.width<<16) +i.bits[1]}}}' for i in r.field])

        def format_grp(act, param, ctrl, base=0):
            ret = []
            ret += [
                f'{{"{r.name}", Register({r.default},{r.addr+base},{{{format_reg(r)}}}, {r.type})}}' for r in act]
            ret += [
                f'{{"{r.name}", Register({r.default},{r.addr+base},{{{format_reg(r)}}}, {r.type})}}' for r in param]
            ret += [
                f'{{"{r.name}", Register({r.default},{r.addr+base},{{{format_reg(r)}}}, {r.type})}}' for r in ctrl]
            return ",\n".join(ret)

        if info["cnt"] != 1:
            offset = 0
            for i in range(info['cnt']):
                groups.append(
                    f'{{"{g}{i}",AIFFGroupRegs({{{format_grp(act, param, ctrl, offset)}}})}}')
                offset += info['offset']
        else:
            groups.append(
                f'{{"{g}",AIFFGroupRegs({{{format_grp(act, param, ctrl)}}})}}')
    with open(path, "w") as f:
        f.write(ret % (arch, ",\n".join(groups)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RegisterGenerator')
    parser.add_argument('-c', '--cfg_path',
                        type=str,
                        default='/project/ai/scratch01/neowan01/git/simulator_new/configs/x3/system_registers.xml',
                        help='Path to simulator config file')
    parser.add_argument('-a', '--arch',
                        type=str,
                        default='X3P',
                        help='Hardware arch, e.g. "X3P"')
    opt = parser.parse_args()
    regs = xml_file_to_dict(opt.cfg_path)["AIFF"]["REGISTER"]["GROUP"]
    ret = walk_to_regs(regs)
    arch = opt.arch.lower()
    path = os.path.join(DIR, f'{arch}_regs.cpp')
    gen_regs_map(ret, arch, path)
    # print(gen_aiff_regs_c(ret, arch))
    # regs = flat_regs(ret)
    # gen_regs_c(regs)
