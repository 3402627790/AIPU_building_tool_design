import xml.etree.ElementTree as ET
import os
import re
import struct


_DIR = os.path.dirname(os.path.realpath(__file__))


def xml_to_dict(element):
    node = {}

    for key, value in element.attrib.items():
        node["@" + key] = value

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


paths = ["/project/ai/scratch01/neowan01/git/zhouyi_arch_common/decoder/configs/zhouyi_kun/isa.xml",
         "/project/ai/scratch01/neowan01/git/zhouyi_arch_common/decoder/configs/zhouyi_kun_x2p/isa.xml",
         "/project/ai/scratch01/neowan01/git/zhouyi_arch_common/decoder/configs/zhouyi_kun_x3/isa.xml",
         ]


def get_ins_dict(path):

    xml_data = open(path, "r").read()
    root = ET.fromstring(xml_data)

    dict_data = xml_to_dict(root)
    return dict_data


def merge_dict(d0, d1):
    ret = d0
    for i in d1:
        if i in d0:
            if type(d0[i]) == type(d1[i]):
                if isinstance(d0[i], list):
                    ret[i] += d1[i]
                elif isinstance(d0[i], dict):
                    ret[i] = merge_dict(d0[i], d1[i])
        else:
            ret[i] = d1[i]
    return ret


dict_data = dict()
for p in paths:
    dict_data = merge_dict(dict_data, get_ins_dict(p))


def gen_opcode(d, path=os.path.join(_DIR, "../src/asm/asmopcode.h"), pyins=os.path.join(_DIR, "../python/src/AIPUBuilder/asm/instruction.pyi")):
    '''
    enum class Instruction
    { 
        bl,
        br,
        mov,
        nop,
        wfe,
    };

    const std::map<uint32_t, Instruction> instructionMap=
    {
        {0xff7f0c40,Instruction::bl},
        {0xff7f0c40,Instruction::br},
        {0xff7f0c40,Instruction::mov},
        {0xff7f0c40,Instruction::nop},
        {0xff7f0c40,Instruction::wfe},
    };
    '''
    inst = d["inst"]
    map = {i["@cmd"].upper(): i for i in inst}
    # maxi=0
    # for i in inst:
    #     inp=i["@inputs"]
    #     s= len(inp.split(","))
    #     maxi=max(maxi,s)
    # print(maxi)

    def gen_fmt(ins, bytes=True):
        class Op:
            def __init__(self, type, size, shift, target=False):
                self.type = type
                self.size = size
                self.shift = shift
                self.target = target

            @property
            def encode(self):
                return struct.pack("H", (int(self.target) << 14) | (self.type << 10) | (self.size << 5) | self.shift)
        inp = ins["@inputs"].strip()
        target = ins["@target"]
        inps = inp.split(",")
        if len(inp) == 0:
            inps = []
        fmt = re.compile(r'([a-zA-z\d+_]+)\[(\d+):(\d+)\]')
        ret = []
        for i in inps:
            is_signed = False
            res = fmt.findall(i)
            if len(res) == 0:
                raise Exception("unknown format!!!" + i)
            res = res[0]
            shift = int(res[2])
            size = int(res[1]) - shift + 1
            t = 0
            if '_' in res[0]:
                t = 6
            elif 'imm' in res[0]:
                t = 5
                if not 'uimm' in res[0]:
                    is_signed = True
            elif "mode" in res[0] or "size" in res[0]:
                t = 5
            elif res[0].startswith("R"):
                t = 0
            elif res[0].startswith("F"):
                t = 1
            elif res[0].startswith("T"):
                t = 2
            elif res[0].startswith("P"):
                t = 3
            elif res[0].startswith("S"):
                t = 4

            else:
                raise Exception("xxxxxx")
            ret.append(Op(t, size, shift, target == res[0] or is_signed))
        while len(ret) < 4:
            ret.append(Op(0, 0, 0))
        if bytes:
            buf = b"".join([i.encode for i in ret])
            return struct.unpack("L", buf)[0]
        return ret

    def format_slot(slots: str):
        slots = slots.replace("[", "").replace("]", "").split(",")
        slots = [int(i) for i in slots if len(i) != 0]
        # 4bits for each slot, 0 is not valid, 1 is valid
        ret = 0
        for i in slots:
            ret = ret | (1 << i)
        return ret
    # gen_fmt(inst[1155 - 6])
    # groups=dict()
    # for k in map:
    #     base= int(map[k]["@code_base"],16)
    #     mask = int(map[k]["@base_mask"],16)
    #     add_new = True
    #     for k in groups:
    #         if k&base == base:
    #             add_new = False
    #             groups[k].append(base)
    #             break
    #     if add_new:
    #         groups[base]=[base]
    with open(path, "w") as f:
        f.write("""//autogenerated DO NOT modify!!!!

""")
        f.write(
            "enum class InstructionType{\n" + ",\n".join(map.keys()) + "};\n")
        f.write(
            'constexpr const uint32_t BaseCode[]={' + ",".join([f'{map[k]["@code_base"]}' for k in map.keys()]) + '};\n')
        f.write(
            'constexpr const uint32_t BaseMask[]={' + ",".join([f'{map[k]["@base_mask"]}' for k in map.keys()]) + '};\n')
        f.write('constexpr const char* Mnemonic[]={' + ",".join(
            [f'"{map[k]["@syntax"]}"' for k in map.keys()]) + '};\n')
        f.write('constexpr const char* InstructionTypeName[]={' + ",".join(
            [f'"{k}"' for k in map.keys()]) + '};\n')
        f.write(
            'constexpr const uint64_t FormatCoding[]={' + ",".join([f'{gen_fmt(map[k])}' for k in map.keys()]) + '};\n')
        f.write(
            'constexpr const uint8_t VLIWSlots[]={' + ",".join([f'{format_slot(map[k]["@slot"])}' for k in map.keys()]) + '};\n')
        # f.write('const int Mnemonic[]={'+",\n".join([f"{map[k]["@mnemonic"]}" for k in map.keys()])+'};\n')

    template = '''
class %s:
    def __init__(self,*args,**kwargs) -> None: ...
    '''
    with open(pyins, "w") as f:
        f.write("""#autogenerated DO NOT modify!!!!

""")
        f.write("\n".join([template % k for k in map.keys()]))


gen_opcode(dict_data)
