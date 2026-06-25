from xml.etree import ElementTree as ET
from AIPUBuilder.core import Target, Profiler
from ...logger import WARN

frequency = 1000
def set_frequency(freq):
    global frequency
    frequency = freq

class NormalTableItemGen:
    def __init__(self, header):
        self.header = header
    def gen_header(self):
        th = ET.Element("th")
        th.text = self.header
        return th
    def gen_content(self, func, node_info, *args):
        td = ET.Element("td")
        td.text = func(node_info, *args)
        if "content" not in node_info:
            node_info["content"] = dict()
        node_info["content"][self.gen_header().text] = td.text
        return td

class NormalEventItem(NormalTableItemGen):
    def __init__(self, header, eid):
        super(NormalEventItem, self).__init__(header)
        self.eid = eid

    def gen_content(self, node_info, *args):
        def func(node_info):
            return str(node_info[self.eid])
        return super().gen_content(func, node_info, *args)

class LayerNameItem(NormalTableItemGen):
    def __init__(self):
        super(LayerNameItem, self).__init__("Layer name")

    def gen_content(self, node_info, *args):
        def func(node_info):
            return str(node_info["name"])
        return super().gen_content(func, node_info, *args)

class PluginNameItem(NormalTableItemGen):
    def __init__(self):
        super(PluginNameItem, self).__init__("Plugin name")

    def gen_content(self, node_info, *args):
        def func(node_info):
            return str(node_info["plugin"])
        return super().gen_content(func, node_info, *args)

class OperatorIDsItem(NormalTableItemGen):
    def __init__(self):
        super(OperatorIDsItem, self).__init__("Operator IDs")

    def gen_content(self, node_info, *args):
        def func(node_info):
            return str(node_info["operator_id"])
        return super().gen_content(func, node_info, *args)

class TimeCostItem(NormalTableItemGen):
    def __init__(self):
        pass

    def gen_header(self):
        global frequency
        th = ET.Element("th")
        th.text = f"Time cost(@{frequency}MHZ, ms)"
        self.header = th.text
        return th

    def gen_content(self, node_info, *args):
        def func(node_info):
            global frequency
            val = node_info[0] / frequency / 1000
            return "%.3f" % val
        return super().gen_content(func, node_info, *args)

class CycleBaseUtilizationItem(NormalTableItemGen):
    def __init__(self, unit, eid):
        super(CycleBaseUtilizationItem, self).__init__(f"{unit} utilization")
        self.eid = eid

    def gen_content(self, node_info, *args):
        def func(node_info):
            if node_info[0] == 0:
                return "nan"
            val = node_info[self.eid] / node_info[0] * 100
            val = "%.3f" % val
            return val + "%"
        return super().gen_content(func, node_info, *args)


class BytesBaseItem(NormalTableItemGen):
    def __init__(self, header, eid):
        super(BytesBaseItem, self).__init__(header)
        self.eid = eid

    def gen_content(self, node_info, *args):
        def func(node_info):
            return str(node_info[self.eid] * Target().axi_bandwidth // 8)
        return super().gen_content(func, node_info, *args)


class OpsItem(NormalTableItemGen):
    def __init__(self):
        super(OpsItem, self).__init__("Algorithm Ops(Million)")

    def gen_content(self, node_info, *args):
        def func(node_info):
            val = node_info["ops_count"] / 1000000
            val = "%.3f" % val
            return val
        return super().gen_content(func, node_info, *args)

class OpsPercentItem(NormalTableItemGen):
    def __init__(self):
        super(OpsPercentItem, self).__init__("Op percentiage")

    def gen_content(self, node_info, *args):
        def func(node_info):
            if node_info["total_ops"] == 0:
                return "nan"
            val = node_info["ops_count"] / node_info["total_ops"] * 100
            val = "%.3f" % val
            return val + "%"
        return super().gen_content(func, node_info, *args)


class AverageBandWidthItem(NormalTableItemGen):
    def __init__(self):
        pass

    def gen_header(self):
        global frequency
        th = ET.Element("th")
        th.text = f"Average bandwidth(@{frequency}MHz, MB/s)"
        self.header = th.text
        return th

    def gen_content(self, node_info, *args):
        def func(node_info):
            val = (node_info[120] + node_info[123]) * Target().axi_bandwidth // 8
            cycle = node_info[0]
            if cycle == 0:
                return "nan"
            cycle_time = cycle / frequency
            val =  val / cycle_time
            val = "%.3f" % val
            return val
        return super().gen_content(func, node_info, *args)

class MACUtilizationItem(NormalTableItemGen):
    def __init__(self):
        super(MACUtilizationItem, self).__init__(f"MAC_utilization")

    def gen_content(self, node_info, *args):
        def func(node_info):
            target = Target().name
            check_on_aiff = True
            if target[:2] == "Z1":
                if int(node_info[91]) > 0:
                    check_on_aiff = True
                else:
                    check_on_aiff = False
            else:
                if node_info[256] + node_info[257] > 0:
                    check_on_aiff = True
                else:
                    check_on_aiff = False
            if not check_on_aiff:
                return "slash_ignore"
            else:
                if node_info[0] == 0:
                    return "nan"
                val = node_info["ops_count"] / (Target().ff_mac_num * node_info[0] * 2) * 100
            val = "%.3f" % val
            return val + "%"
        return super().gen_content(func, node_info, *args)

class AddBytesBasedItem(NormalTableItemGen):
    def __init__(self, header, eids):
        super(AddBytesBasedItem, self).__init__(header)
        self.eids = eids


    def gen_content(self, node_info, *args):
        def func(node_info):
            sum_val = 0
            for eid in self.eids:
                sum_val += node_info[eid] * Target().axi_bandwidth // 8
            return str(sum_val)
        return super().gen_content(func, node_info, *args)


class AddBytesFixedWidthBasedItem(NormalTableItemGen):
    def __init__(self, header, eids, width):
        super(AddBytesFixedWidthBasedItem, self).__init__(header)
        self.eids = eids
        self.width = width


    def gen_content(self, node_info, *args):
        def func(node_info):
            sum_val = 0
            for eid in self.eids:
                sum_val += node_info[eid] * self.width
            return str(sum_val)
        return super().gen_content(func, node_info, *args)

class InputShapesItem(NormalTableItemGen):
    def __init__(self):
        super(InputShapesItem, self).__init__(f"Input Shapes")

    def gen_content(self, node_info, *args):
        def func(node_info):
            inputs = node_info["inputs"]
            shapes = []
            for inp in inputs:
                shapes.append(inputs[inp]["shape"])
            return ",".join(shapes)
        return super().gen_content(func, node_info, *args)

class SummaryTableItemGen:
    def __init__(self, header):
        self.header = header
    def gen_header(self):
        td = ET.Element("td")
        td.text = self.header
        return td
    def gen_content(self, func, node_infos, *args):
        td = ET.Element("td")
        td.text = func(node_infos, *args)
        for node_info in node_infos:
            if "summary_content" not in node_info:
                node_info["summary_content"] = dict()
            node_info["summary_content"][self.gen_header().text] = td.text
        return td


class SummaryAverageUtilizationGen(SummaryTableItemGen):
    def __init__(self, unit):
        super(SummaryAverageUtilizationGen, self).__init__(f"average {unit} utilization")
        self.unit = unit

    def gen_content(self, node_infos, *args):
        def func(node_infos):
            sum_val = 0
            cycles = int(float(node_infos[0]["summary_content"]["total cycles"]))
            for node_info in node_infos:
                sum_val += int(node_info["content"][f"{self.unit} busy cycle"])
            if cycles == 0:
                return "nan"
            val = sum_val / cycles * 100
            val = "%.3f" % val
            return val + "%"
        return super().gen_content(func, node_infos, *args)

class SummarySumItem(SummaryTableItemGen):
    def __init__(self, header, eids):
        super(SummarySumItem, self).__init__(header)
        self.eids = eids if isinstance(eids, list) else [eids]

    def gen_content(self, node_infos, *args):
        def func(node_infos):
            sum_val = 0

            for node_info in node_infos:
                for eid in self.eids:
                    try:
                        number = int(node_info["content"][eid])
                    except ValueError:
                        number = float(node_info["content"][eid])
                    sum_val += number
            if isinstance(sum_val, int):
                return str(sum_val)
            else:
                return "%.3f" % sum_val
        return super().gen_content(func, node_infos, *args)

class SummaryFpsItem(SummaryTableItemGen):
    def __init__(self):
        super(SummaryFpsItem, self).__init__("fps")

    def gen_content(self, node_infos, *args):
        def func(node_infos):
            cycles = int(float(node_infos[0]["summary_content"]["total cycles"]))
            global frequency
            if cycles == 0:
                return "nan"
            fps = frequency * 1000000 / cycles
            fps = "%.3f" % fps
            return f"{fps}@({frequency}MHz)"
        return super().gen_content(func, node_infos, *args)

class SummaryCostTimeItem(SummaryTableItemGen):
    def __init__(self):
        super(SummaryCostTimeItem, self).__init__("total time")

    def gen_content(self, node_infos, *args):
        def func(node_infos):
            cycles = int(float(node_infos[0]["summary_content"]["total cycles"]))
            global frequency
            ms = cycles / frequency / 1000
            return f"{ms}ms(@{frequency}MHz)"
        return super().gen_content(func, node_infos, *args)

class SummaryInputShapesGen(SummaryTableItemGen):
    def __init__(self):
        super(SummaryInputShapesGen, self).__init__("model input shapes")

    def gen_content(self, node_infos, *args):
        def func(node_infos):
            return node_infos[0]["model input shapes"]
        return super().gen_content(func, node_infos, *args)


class PlatformTargetItem(SummaryTableItemGen):
    def __init__(self):
        super(PlatformTargetItem, self).__init__("platform target")

    def gen_content(self, node_infos, *args):
        def func(node_infos):
            target = Target().name
            return target
        return super().gen_content(func, node_infos, *args)


BASE_TITLE_ITEM_CONFIG = {
    "layer_name" : LayerNameItem(),
    "layer_plugin" : PluginNameItem(),
    "operator_ids" : OperatorIDsItem(),
    "count_cycle" : NormalEventItem("Count cycle", 0),
    "time_cost" : TimeCostItem(),
    "TPC_credit_cycle" : NormalEventItem(f"TPC credit cycle", 64),
    "AIPU_read_bytes" : BytesBaseItem("AIPU read bytes", 120),
    "AIPU_write_bytes" : BytesBaseItem("AIPU write bytes", 123),
    "HWA_busy_cycle" : NormalEventItem(f"HWA busy cycle", 91),
    "HWA_utilization" : CycleBaseUtilizationItem("HWA", 91),

    "MTP0_busy_cycle" : NormalEventItem(f"MTP0 busy cycle", 256),
    "MTP0_utilization" : CycleBaseUtilizationItem("MTP0", 256),
    "MTP1_busy_cycle" : NormalEventItem(f"MTP1 busy cycle", 257),
    "MTP1_utilization" : CycleBaseUtilizationItem("MTP1", 257),
    "ITP_busy_cycle" : NormalEventItem(f"ITP busy cycle", 258),
    "ITP_utilization" : CycleBaseUtilizationItem("ITP", 258),
    "PTP_busy_cycle" : NormalEventItem(f"PTP busy cycle", 259),
    "PTP_utilization" : CycleBaseUtilizationItem("PTP", 259),

    "DMA_busy_cycle" : NormalEventItem(f"DMA busy cycle", 108),
    "DMA_utilization" : CycleBaseUtilizationItem("DMA", 108),
    "AIFF_busy_cycle" : NormalEventItem(f"AIFF busy cycle", 82),
    "AIFF_utilization" : CycleBaseUtilizationItem("AIFF", 82),

    "AIFF_weight_read_bytes" : AddBytesFixedWidthBasedItem("AIFF_weight_read_bytes", [84, 86], 32),

    "DMA_read_bytes" : BytesBaseItem("DMA_read_bytes", 110),
    "DMA_write_bytes" : BytesBaseItem("DMA_write_bytes", 112),

    "algorithm_ops" : OpsItem(),
    "op_percentiage" : OpsPercentItem(),
    "average_bandwidth" : AverageBandWidthItem(),
    "MAC_utilization" : MACUtilizationItem(),
    "Input_Shapes" : InputShapesItem(),
    "RDMA_WT_RD_CNT" : NormalEventItem("AIFF Weight Read Wait Cycle", 265),
    "RDMA_ACT_RD_CNT" : NormalEventItem("AIFF Input Read Wait Cycle", 266),
    "WDMA_ACT_WR_CNT" : NormalEventItem("AIFF Output Write Wait Cycle", 267),

    ######################################
    "summary target" : PlatformTargetItem(),
    "summary ops" : SummarySumItem("total ops(Mops)", ["Algorithm Ops(Million)"]),
    "summary cycles" : SummarySumItem("total cycles", ["Count cycle"]),
    "summary time" : SummaryCostTimeItem(),
    "summary fps" : SummaryFpsItem(),
    "summary AIPU read bytes" : SummarySumItem("total AIPU read bytes", ["AIPU read bytes"]),
    "summary AIPU write bytes" : SummarySumItem("total AIPU write bytes", ["AIPU write bytes"]),
    "summary AIPU read + write bytes" : SummarySumItem("total AIPU read + write bytes", ["AIPU read bytes", "AIPU write bytes"]),
    "summary DMA read bytes" : SummarySumItem("total DMA read bytes", ["DMA_read_bytes"]),
    "summary DMA write bytes" : SummarySumItem("total DMA write bytes", ["DMA_write_bytes"]),
    "summary DMA read + write bytes" : SummarySumItem("total DMA read + write bytes", ["DMA_read_bytes", "DMA_write_bytes"]),

    "summary average HWA" : SummaryAverageUtilizationGen("HWA"),
    "summary average DMA" : SummaryAverageUtilizationGen("DMA"),
    "summary average AIFF" : SummaryAverageUtilizationGen("AIFF"),
    "summary average MTP0" : SummaryAverageUtilizationGen("MTP0"),
    "summary average MTP1" : SummaryAverageUtilizationGen("MTP1"),
    "summary average ITP" : SummaryAverageUtilizationGen("ITP"),
    "summary average PTP" : SummaryAverageUtilizationGen("PTP"),
    "summary Input Shapes" : SummaryInputShapesGen(),
}



class BaseConfig:
    def __init__(self, units = [], footprints = [], *args):
        self.config = ["layer_name", "layer_plugin", "count_cycle", "time_cost"]
        if isinstance(units, str):
            units = [units]
        if isinstance(footprints, str):
            footprints = [footprints]
        fp = []
        for unit in footprints:
            fp.append(f"{unit}_read_bytes")
            fp.append(f"{unit}_write_bytes")
        footprints = fp
        added = []
        for unit in units:
            if unit != "TPC":
                added += [f"{unit}_busy_cycle", f"{unit}_utilization"]
            else:
                added += [f"{unit}_credit_cycle"]
        self.config = self.config + added + footprints + list(args)
        if "algorithm_ops" not in self.config:
            self.config += ["algorithm_ops"]

class Z2BaseConfig(BaseConfig):
    def __init__(self, units = [], footprints = [], *args):
        super(Z2BaseConfig, self).__init__(units + ["MTP0", "MTP1", "ITP", "PTP"], footprints, *args)
        if "algorithm_ops" not in self.config:
            self.config += ["algorithm_ops"]


class ExtBaseConfig(BaseConfig):
    def __init__(self, units = [], footprints = [], *args):
        super(ExtBaseConfig, self).__init__(units, footprints, *args)
        self.config = self.config[0:1] + ["operator_ids"] + self.config[1:]
        footprint = ['AIPU_read_bytes', 'AIPU_write_bytes']
        if 'AIPU_read_bytes' in self.config and 'AIPU_write_bytes' in self.config:
            self.config = [config for config in self.config if config not in footprint]
            self.config = self.config[0:5] + footprint + self.config[5:]

class ExtConfig(ExtBaseConfig):
    def __init__(self, units = [], footprints = [], *args):
        super(ExtConfig, self).__init__(units, footprints, *args)
        self.config = self.config + ["op_percentiage", "average_bandwidth", "MAC_utilization"]

class ExtNoBWConfig(ExtBaseConfig):
    def __init__(self, units = [], footprints = [], *args):
        super(ExtNoBWConfig, self).__init__(units, footprints, *args)
        self.config = self.config + ["op_percentiage", "MAC_utilization"]

class ExtNoMACConfig(ExtBaseConfig):
    def __init__(self, units = [], footprints = [], *args):
        super(ExtNoMACConfig, self).__init__(units, footprints, *args)
        self.config = self.config + ["op_percentiage", "average_bandwidth"]

class Z2ExtConfig(ExtConfig):
    def __init__(self, units = [], footprints = [], *args):
        super(Z2ExtConfig, self).__init__(units + ["MTP0", "MTP1", "ITP", "PTP"], footprints, *args)

class Z2ExtNoBWConfig(ExtNoBWConfig):
    def __init__(self, units = [], footprints = [], *args):
        super(Z2ExtNoBWConfig, self).__init__(units + ["MTP0", "MTP1", "ITP", "PTP"], footprints, *args)

class BaseSummaryConfig:
    def __init__(self, units = [], footprints = [], *args):
        self.config = ["target", "ops", "cycles", "time", "fps"]
        units = ["average " + unit for unit in units]
        rw = []
        for unit in footprints:
            rw.append(f"{unit} read bytes")
            rw.append(f"{unit} write bytes")
            rw.append(f"{unit} read + write bytes")
        self.config = self.config + units + rw + list(args)
        self.config = ["summary " + unit for unit in self.config]

class ExtSummaryConfig(BaseSummaryConfig):
    def __init__(self, units = [], footprints = [], *args):
        super(ExtSummaryConfig, self).__init__(units, footprints, *args)

class Z2SummaryConfig(BaseSummaryConfig):
    def __init__(self, units = [], footprints = [], *args):
        super(Z2SummaryConfig, self).__init__(units + ["MTP0", "MTP1", "ITP", "PTP"], footprints, *args)

class Z2SummaryExtConfig(ExtSummaryConfig):
    def __init__(self, units = [], footprints = [], *args):
        super(Z2SummaryExtConfig, self).__init__(units + ["MTP0", "MTP1", "ITP", "PTP"], footprints, *args)


MAIN_TABLE_CONFIG = {
    "Z1" : {
        "default" : {
            "0" : BaseConfig(["HWA"], ["AIPU"], "Input_Shapes").config,
            "1" : ExtConfig(["HWA"], ["AIPU"], "Input_Shapes").config
        },
        "TPC" : {
            "0" : BaseConfig(["TPC"], ["AIPU"]).config,
            "1" : ExtNoMACConfig(["TPC"], ["AIPU"]).config
        },
    },
    "Z2" : {
        "default" : {
            "0" : Z2BaseConfig(["TPC"], ["AIPU"], "Input_Shapes").config,
            "1" : Z2ExtConfig(["TPC"], ["AIPU"], "Input_Shapes").config,
        },
        "AIFF" : {
            "0" : Z2BaseConfig(["AIFF"], [], "AIFF_weight_read_bytes").config,
            "1" : Z2ExtNoBWConfig(["AIFF"], [], "AIFF_weight_read_bytes").config,
        },
        "TPC" : {
            "0" : Z2BaseConfig(["TPC"], ["AIPU"]).config,
            "1" : Z2ExtConfig(["TPC"], ["AIPU"]).config,
        },
        "DMA" : {
            "0" : Z2BaseConfig(["DMA"], ["DMA"]).config,
            "1" : Z2ExtNoBWConfig(["DMA"], ["DMA"]).config,
        },
    },
    "X1" : {
        "default" : {
            "0" : Z2BaseConfig(["TPC", "AIFF", "DMA"], ["DMA", "AIPU"], "RDMA_WT_RD_CNT", "RDMA_ACT_RD_CNT", "WDMA_ACT_WR_CNT", "Input_Shapes").config,
            "1" : Z2ExtConfig(["TPC", "AIFF", "DMA"], ["DMA", "AIPU"], "RDMA_WT_RD_CNT", "RDMA_ACT_RD_CNT", "WDMA_ACT_WR_CNT", "Input_Shapes").config,
        },
        "AIFF" : {
            "0" : Z2BaseConfig(["TPC", "AIFF", "DMA"], ["DMA", "AIPU"], "RDMA_WT_RD_CNT", "RDMA_ACT_RD_CNT", "WDMA_ACT_WR_CNT").config,
            "1" : Z2ExtConfig(["TPC", "AIFF", "DMA"], ["DMA", "AIPU"], "RDMA_WT_RD_CNT", "RDMA_ACT_RD_CNT", "WDMA_ACT_WR_CNT").config,
        },
        "TPC" : {
            "0" : Z2BaseConfig(["TPC", "AIFF", "DMA"], ["DMA", "AIPU"], "RDMA_WT_RD_CNT", "RDMA_ACT_RD_CNT", "WDMA_ACT_WR_CNT").config,
            "1" : Z2ExtConfig(["TPC", "AIFF", "DMA"], ["DMA", "AIPU"], "RDMA_WT_RD_CNT", "RDMA_ACT_RD_CNT", "WDMA_ACT_WR_CNT").config,
        },
        "DMA" : {
            "0" : Z2BaseConfig(["TPC", "AIFF", "DMA"], ["DMA", "AIPU"], "RDMA_WT_RD_CNT", "RDMA_ACT_RD_CNT", "WDMA_ACT_WR_CNT").config,
            "1" : Z2ExtConfig(["TPC", "AIFF", "DMA"], ["DMA", "AIPU"], "RDMA_WT_RD_CNT", "RDMA_ACT_RD_CNT", "WDMA_ACT_WR_CNT").config,
        },
    }
}

SUMMARY_TABLE_CONFIG = {
    "Z1" : {
        "default" : {
            "0" : BaseSummaryConfig(["HWA"], ["AIPU"]).config,
            "1" : ExtSummaryConfig(["HWA"], ["AIPU"]).config
        },
        "TPC" : {
            "0" : BaseSummaryConfig([], ["AIPU"]).config,
            "1" : ExtSummaryConfig([], ["AIPU"]).config
        },
    },
    "Z2" : {
        "default" : {
            "0" : Z2SummaryConfig([], ["AIPU"]).config,
            "1" : Z2SummaryExtConfig([], ["AIPU"]).config,
        },
        "AIFF" : {
            "0" : Z2SummaryConfig(["AIFF"], []).config,
            "1" : Z2SummaryExtConfig(["AIFF"], []).config,
        },
        "TPC" : {
            "0" : Z2SummaryConfig([], ["AIPU"]).config,
            "1" : Z2SummaryExtConfig([], ["AIPU"]).config,
        },
        "DMA" : {
            "0" : Z2SummaryConfig(["DMA"], ["DMA"]).config,
            "1" : Z2SummaryExtConfig(["DMA"], ["DMA"]).config,
        },
    },
    "X1" : {
        "default" : {
            "0" : Z2SummaryConfig(["AIFF", "DMA"], ["DMA", "AIPU"], "Input Shapes").config,
            "1" : Z2SummaryExtConfig(["AIFF", "DMA"], ["DMA", "AIPU"], "Input Shapes").config,
        },
        "AIFF" : {
            "0" : Z2SummaryConfig(["AIFF", "DMA"], ["DMA", "AIPU"]).config,
            "1" : Z2SummaryExtConfig(["AIFF", "DMA"], ["DMA", "AIPU"]).config,
        },
        "TPC" : {
            "0" : Z2SummaryConfig(["AIFF", "DMA"], ["DMA", "AIPU"]).config,
            "1" : Z2SummaryExtConfig(["AIFF", "DMA"], ["DMA", "AIPU"]).config,
        },
        "DMA" : {
            "0" : Z2SummaryConfig(["AIFF", "DMA"], ["DMA", "AIPU"]).config,
            "1" : Z2SummaryExtConfig(["AIFF", "DMA"], ["DMA", "AIPU"]).config,
        },
    }
}



def get_configs(graph_json, verbose):
    target = Target().name
    platform = target[:2]
    if target == "Z3_1204" or target[:2] == "X1":
        platform = "X1"
    elif platform == "Z3":
        platform = "Z2"
    info = graph_json["profile"]
    events = info["profile events"]

    if events == "default":
        prof_unit = info["prof_unit"]
        if prof_unit not in MAIN_TABLE_CONFIG[platform]:
            WARN(f"[Profiler] unkown prof_unit {prof_unit}, use default")
            configs = MAIN_TABLE_CONFIG[platform]["default"]
            summary = SUMMARY_TABLE_CONFIG[platform]["default"]
        else:
            configs = MAIN_TABLE_CONFIG[platform][prof_unit]
            summary = SUMMARY_TABLE_CONFIG[platform][prof_unit]
        configs = configs[str(verbose)]
        summary = summary[str(verbose)]
        return [[BASE_TITLE_ITEM_CONFIG[config] for config in configs], [BASE_TITLE_ITEM_CONFIG[config] for config in summary]]
    else:
        if isinstance(events, str):
            events = events.strip().split(",")
        events = [int(e) for e in events]
        events.sort()
        events = [int(eid) for eid in events]
        configs = [LayerNameItem()]
        profiler = Profiler()
        for eid in events:
            configs.append(NormalEventItem(profiler.event_description(eid), eid))
        return [configs, []]
