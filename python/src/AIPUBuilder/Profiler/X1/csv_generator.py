from xml.etree import ElementTree as ET
import pandas as pd
from AIPUBuilder.core import Target, Profiler
from .profiler_config import get_configs
from ..js_script import *

class CsvGenerator:
    def gen(self, graph_json, nodes_info, report_name, verbose):
        from openpyxl import Workbook
        from openpyxl.utils.dataframe import dataframe_to_rows
        profile_info = graph_json["profile"]
        prof_unit = profile_info["prof_unit"]
        if verbose > 1 and prof_unit != "default":
            WARN("[Profiler] only prof_unit is default can set verbose > 1, set verbose to 1")
            verbose = 1

        configs = get_configs(graph_json, verbose)[0]

        for node_info in nodes_info:
            for config in configs:
                config.gen_content(node_info)

        table = {}
        for node_info in nodes_info:
            content = node_info["content"]
            for item in content:
                if item not in table:
                    table[item] = list()
                table[item].append(node_info["content"][item])
        dataframe = pd.DataFrame(table)

        wb = Workbook()
        ws = wb.active
        ws.title = "Layer Data"

        for row in dataframe_to_rows(dataframe, index=False, header=True):
            ws.append(row)
        wb.save(report_name)