from xml.etree import ElementTree as ET
from AIPUBuilder.core import Target, Profiler
from .profiler_config import get_configs
from ..js_script import *
from ..css_style_script import *
import json
import sys
import os
import networkx as nx

def construct_graph(graph_json):
    nodes = graph_json["actual_nodes"]
    tensors = graph_json["tensors"]
    graph = nx.DiGraph()
    tensor_map = dict()
    node_id = dict()
    for id, node in enumerate(nodes):
        name = node["layer node name"]
        node_id[name] = id

        plugin = node["plugin"] if "plugin" in node else ""
        internal = node["internal_nodes"] if "internal_nodes" in node else dict()
        graph.add_node(name, id=id, op_type = node["op_type"], plugin = plugin, internal = internal)

        outputs = node["outputs"] if node["outputs"] != "null" and node["outputs"] is not None else []
        for out in outputs:
            tensor_map[out] = name


    for node in nodes:
        inputs = node["inputs"] if node["inputs"] != "null" and node["inputs"] is not None else []
        name = node["layer node name"]
        for inp in inputs:
            tensor = tensors[inp]
            src = tensor_map[inp]
            dst = name
            graph.add_edge(src, dst, src = node_id[src], dst = node_id[dst], name = inp, dtype = tensor["dtype"], shape = tensor["shape"])

    return graph


def generate_draw_graph_text(graph, nodes_info):
    dataset = dict()
    dataset["nodes"] = list()
    dataset["edges"] = list()

    infos = dict()
    for info in nodes_info:
        infos[info["name"]] = info["content"]

    for n in graph.nodes:
        node = graph.nodes[n]

        node_dict = dict()
        node_dict["id"] = node["id"]
        node_dict["label"] = n
        node_dict["shape"] = "rect"
        node_dict["internal"] = node["internal"] if "internal" in node else dict()
        if n in infos:
            node_dict["style"] = "fill:#f2f2f2;stroke:#F7DC6F"
        else:
            node_dict["style"] = "fill:#f200f2;stroke:#F7DC6F"
        dataset["nodes"].append(node_dict)

    for e in graph.edges:
        edge = graph.edges[e]
        src = edge["src"]
        dst = edge["dst"]
        edge_dict = dict()
        edge_dict["source"] = src
        edge_dict["target"] = dst
        edge_dict["name"] = edge["name"]
        edge_dict["dtype"] = edge["dtype"]
        edge_dict["shape"] = edge["shape"]
        dataset["edges"].append(edge_dict)

    scritp_str = f"""let dataset={json.dumps(dataset)}\n"""
    scritp_str = scritp_str + "function draw_graph(){\n"

    scritp_str = scritp_str + r'''
    let g = new dagreD3.graphlib.Graph();
    g.setGraph({
    rankdir:'TB'
    });
    dataset.nodes.forEach(item => {
        g.setNode(item.id, {
        label: item.label,
        labelStyle: "font-size:medium",
        internal: item.internal,
        shape: item.shape,
        style: item.style
        })
    })
    dataset.edges.forEach(item => {
        g.setEdge(item.source,item.target, {
            style: "fill:none;stroke:#333;stroke-width:2px",
            dtype: item.dtype,
            shape: item.shape,
            name: item.name
        })
    })
    let render = new dagreD3.render();
    let svg = d3.select("svg");
    let svgGroup = svg.append('g');
    render(svgGroup, g);
    svg.attr("width", g.graph().width + 100);
    var xCenterOffset = (svg.attr("width") - g.graph().width) / 2;
    svgGroup.attr("transform", "translate(" + xCenterOffset + ", 50)");
    svg.attr("height", g.graph().height + 40);

    svgGroup.selectAll("g.node").on('mouseover', function () {
        d3.select(".nodes")
            .attr("style", "cursor: pointer")
    });
    svgGroup.selectAll("g.node").on('click', function (v) {
        let sidebar = d3.select("#sidebar")
        sidebar.attr("style", "right:10px;top:2px")
        sidebar.selectAll("tr").remove();
        internal = g.node(v).internal;
        for(var inner_node in internal) {
            inner_info = internal[inner_node]
            params = inner_info["params"]
            sidebar.append("tr").append("td").attr("colspan", "3").append("B").text(inner_info["op_type"]);
            var i = 0;
            var tr;
            for(var key in params)
            {
                if(i % 3 == 0)
                {
                    tr = sidebar.append("tr");
                }
                tr.append("td").text(key + ":" + params[key]);
                i++;
            }
            i = i % 3;
            for(i = i % 3;i < 3 && i != 0; i++)
            {
                tr.append("td").text("");
            }
        }
        d3.select("#sidebar").classed("hidden", false);
    });
    };
    '''
    return scritp_str



class HtmlGenerator:
    def gen(self, graph_json, nodes_info, report_name, verbose):
        profile_info = graph_json["profile"]
        prof_unit = profile_info["prof_unit"]
        if verbose > 1 and prof_unit != "default":
            WARN("[Profiler] only prof_unit is default can set verbose > 1, set verbose to 1")
            verbose = 1

        main_configs, summary_configs = get_configs(graph_json, verbose)
        main_table = self.generate_main_table(main_configs, nodes_info)
        summary_table = self.generate_summary_table(summary_configs, nodes_info)

        # html
        root = ET.Element("html")
        title = ET.SubElement(root, "title")
        title.text = report_name

        # head
        head = ET.SubElement(root, "head")
        style = ET.SubElement(head, "style", attrib={"type": "text/css", "media": "all"})
        style.text = style_css_code

        # body
        body = ET.SubElement(root, "body")

        ## scripts
        script_sort_table = ET.SubElement(body, "script")
        script_chartjs = ET.SubElement(body, "script")
        script_sort_table.text = sort_table_script
        script_chartjs.text = chartjs_script
        if verbose >= 1:
            chart_divs, script = self.generate_chart(main_configs, nodes_info)
            body.append(script)
        d3 = ET.SubElement(body, "script")
        d3.text = d3_script
        dagre_script = ET.SubElement(body, "script")
        dagre_script.text = dagre_d3_script
        if verbose >= 1:
            graph_script = ET.SubElement(body, "script")
            graph_script.text = generate_draw_graph_text(construct_graph(graph_json), nodes_info)
        table_chart_script = ET.SubElement(body, "script")
        table_chart_script.text = r'''
        function show_line_chart() {
            document.getElementById("show_summary_table").style.display="none";
            document.getElementById("show_main_table").style.display="none";
            document.getElementById("svg_graph").style.display="none";
            document.getElementById("show_verbose_chart").style.display="block";
            draw_line_chart();
        };
        function show_summary() {
            document.getElementById("show_summary_table").style.display="block";
            document.getElementById("show_main_table").style.display="none";
            document.getElementById("svg_graph").style.display="none";
            if (document.getElementById("show_verbose_chart").style.display != "none") {
                document.getElementById("show_verbose_chart").style.display="none";
            }
        }
        function show_details() {
            document.getElementById("show_main_table").style.display="block";
            document.getElementById("show_summary_table").style.display="none";
            document.getElementById("svg_graph").style.display="none";
            if (document.getElementById("show_verbose_chart").style.display != "none") {
                document.getElementById("show_verbose_chart").style.display="none";
            }
        }
        function show_graph() {
            document.getElementById("show_main_table").style.display="none";
            document.getElementById("show_summary_table").style.display="none";
            document.getElementById("svg_graph").style.display="block";
            if (document.getElementById("show_verbose_chart").style.display != "none") {
                document.getElementById("show_verbose_chart").style.display="none";
            }
            draw_graph();
        }
        '''

        ## wrapper
        wrapper = ET.SubElement(body, "div", attrib={"id": "wrapper"})
        ## header
        header = ET.SubElement(wrapper, "div", attrib={"id": "header"})
        sitetitle = ET.SubElement(header, "h1", attrib={"class": "sitetitle"})
        sitetitle.text = "Profiler Report"
        ## menu, includes summary, detailed data and chart if verbose is enabled
        menu = ET.SubElement(header, "ul", attrib={"id": "menu"})
        show_details_button = ET.SubElement(menu, "button", attrib={"onclick": "show_details()", "style": "font-weight:bolder"})
        show_details_button.text = "DETAILED DATA"

        show_summary_button = ET.SubElement(menu, "button", attrib={"onclick": "show_summary()"})
        show_summary_button.text = "SUMMARY"

        if verbose >= 1:
            show_graph_button = ET.SubElement(menu, "button", attrib={"onclick": "show_graph()"})
            show_graph_button.text = "GRAPH"
            show_chart_buttom = ET.SubElement(menu, "button", attrib={"onclick" : "show_line_chart()"})
            show_chart_buttom.text = "LINE CHART"

        ## seperate line
        line = ET.SubElement(header, "hr", attrib={"class": "big blue"})

        ## main table
        detail_table = ET.SubElement(body, "div", attrib={"id": "show_main_table"})
        detail_title = ET.SubElement(detail_table, "p", attrib={"class": "titlebar"})
        detail_title.text = "Detailed Data"
        detail_table.append(main_table)

        ## summary table
        base_table = ET.SubElement(wrapper, "div", attrib={"id": "show_summary_table", "class": "hidden"})
        summary_title = ET.SubElement(base_table, "p", attrib={"class": "titlebar"})
        summary_title.text = "Summary"
        base_table.append(summary_table)

        ## chart
        chart = ET.SubElement(body, "div" , attrib={"id": "show_verbose_chart", "class": "hidden"})

        ## svg graph
        svg_graph = ET.SubElement(body, "div" , attrib={"id": "svg_graph", "class": "hidden"})
        sidebar = ET.SubElement(svg_graph, "table" , attrib={"id": "sidebar", "class": "hidden"})
        tooltip = ET.SubElement(svg_graph, "table" , attrib={"id": "tooltip", "class": "hidden"})
        # TODO: Check the actual row numbers of table
        for i in range(3):
            tooltip_table = ET.SubElement(tooltip, "tr" , attrib={"id": f"tooltip-{str(i)}"})
            ET.SubElement(tooltip_table, "td" , attrib={"id": f"tooltip-{str(i)}-td1", "style": "width:30%"})
            ET.SubElement(tooltip_table, "td" , attrib={"id": f"tooltip-{str(i)}-td2", "style": "width:70%"})

        svg_graph = ET.SubElement(svg_graph, "svg" , attrib={"width": "1080", "height": "1920", "id" : "graph"})

        if verbose >= 1:
            for div in chart_divs:
                chart.append(div)

        # Save to html file
        with open(report_name, "w") as f:
            ET.ElementTree(root).write(f, encoding="unicode", method="html")

    def color_convert(self, color):
        if len(color) == 3:
            return f"rgb({color[0]},{color[1]},{color[2]})"
        else:
            return f"rgba({color[0]},{color[1]},{color[2]},{color[3]})"


    def gen_line_chart_dataset(self, title, points, color, show_points = False):
        dataset = dict()
        dataset["label"] = title
        dataset["type"] = "line"
        dataset["data"] = points
        dataset["backgroundColor"] = self.color_convert(color)
        dataset["borderColor"] = self.color_convert(color + [0.3])
        if show_points:
            dataset["pointRadius"] = 3
            dataset["pointBackgroundColor"] = self.color_convert([255 - color[0], 255 - color[1], 255 - color[2]])
        else:
            dataset["borderColor"] = self.color_convert([120,0,0])
            dataset["backgroundColor"] = self.color_convert([120,0,0,0])
            dataset["pointRadius"] = 0
        return dataset


    def gen_line_chart_script(self, chart_name, title, labels, points, x_axis_label, y_axis_label):

        chart = dict()
        data = dict()
        chart["type"] = "scatter"
        data["labels"] = labels
        data["datasets"] = list()
        dataset = self.gen_line_chart_dataset(title, points, [213, 245, 227], True)
        data["datasets"].append(dataset)
        if "utilization" in title:
            line_points = [{"x" : 0, "y" : 100}, {"x" : len(labels) + 1, "y" : 100}]
            line = self.gen_line_chart_dataset("100%", line_points, [0, 0, 255], False)
            data["datasets"].append(line)

        chart["data"] = data
        script = f'''
        var canvas = document.getElementById(\"{title}\");
        var ctx = canvas.getContext(\"2d\");
        ctx.clearRect(0,0,canvas.width, canvas.height);
        var {chart_name} = new Chart(ctx, {json.dumps(chart)});
        '''
        default_call_backs =  r'''
        function(tooltipItem, data){var label = data.labels[tooltipItem.index];return label + '  ' + tooltipItem.xLabel + '  ' + tooltipItem.yLabel;}
        '''
        call_backs = default_call_backs
        if "utilization" in title:
            call_backs = r'''
        function(tooltipItem, data){var label = data.labels[tooltipItem.index];return label + '  ' + tooltipItem.xLabel + '  ' + tooltipItem.yLabel + '%';}
        '''
        script = script + f'''
        {chart_name}.options.tooltips.callbacks.label={call_backs};
        {chart_name}.options.scales.xAxes[0].ticks.beginAtZero=true;
        {chart_name}.options.scales.yAxes[0].ticks.beginAtZero=true;
        {chart_name}.options.scales.xAxes[0].ticks.display=false;
        {chart_name}.options.scales.xAxes[0].ticks.display=false;
        {chart_name}.options.scales.yAxes[0].scaleLabel.display = true;
        {chart_name}.options.scales.yAxes[0].scaleLabel.labelString = \"{y_axis_label}\"
        {chart_name}.options.scales.xAxes[0].scaleLabel.display = true;
        {chart_name}.options.scales.xAxes[0].scaleLabel.labelString = \"{x_axis_label}\"
        '''
        return script

    def generate_chart(self, configs, nodes_info):
        main_titles = [config.header for config in configs]
        def filter(title):
            not_to_draw = [
                "Layer name",
                "Algorithm Ops",
                "Op percentiage",
                "Average bandwidth",
                "MAC_utilization",
                "Input Shapes",
                "Plugin name",
                "Operator IDs",
            ]
            if any([draw in title for draw in not_to_draw]):
                return False
            return True
        main_titles = [title for title in main_titles if filter(title)]
        canvases = []

        datas = dict()
        canvases_string = ""
        #main_titles = [title.replace(" ", "_") for title in main_titles]
        for title in main_titles:
            #refined_title = "\"" + title + "\""
            canvas = ET.Element("canvas", attrib={"id" : title, "height" : "100%"})
            canvases.append(canvas)

            data = list()
            for idx, info in enumerate(nodes_info):
                value = info["content"][title]
                if value[-1] == "%":
                    value = value[:-1]
                try:
                    value = int(value)
                except ValueError:
                    value = float(value)
                data.append({"x" : idx, "y" : value})
            datas[title] = data
        labels = [info["name"] for info in nodes_info]
        script = '''
        function draw_line_chart(){
            var chart = document.getElementById(\"show_verbose_chart\");
            chart.style.display = \"block\";
        '''
        chart_script = ""
        for idx, title in enumerate(main_titles):
            chart_name = f'''chart_{idx}'''
            chart_script = chart_script + self.gen_line_chart_script(chart_name, title, labels, datas[title], "layer id", title) + "\n"
        script = script + chart_script + "\n};"
        s = ET.Element("script")
        s.text = script
        return canvases, s


    def generate_main_table(self, configs, nodes_info):
        div = ET.Element("div" , attrib={"id": "main table"})
        table = ET.Element("table" , attrib={"border": "1", "class" : "sortable"})
        div.append(table)

        header = ET.Element("thead")
        header_tr = ET.Element("tr")
        for content in configs:
            header_tr.append(content.gen_header())
        header.append(header_tr)
        table.append(header)

        for node_info in nodes_info:
            item_tr = ET.Element("tr")
            for content in configs:
                td = content.gen_content(node_info)
                if td.text == "slash_ignore":
                    td = ET.Element("td", attrib = {"class" : "slash_ignore"})
                item_tr.append(td)
            table.append(item_tr)
        return div

    def generate_summary_table(self, configs, nodes_info):
        div = ET.Element("div" , attrib={"id": "summary table"})
        table = ET.Element("table", attrib={"class": "center"})
        div.append(table)

        for content in configs:
            tr = ET.Element("tr")
            td = content.gen_header()
            tr.append(td)
            tr.append(content.gen_content(nodes_info))
            table.append(tr)
        return div



class TickGraphGenerator:
    def gen(self, graph_json, nodes_info, report_name, custom_exec):

        custom_exec_ = dict()
        for eid in custom_exec:
            custom_exec_[eid] = list()
            begin_stack = list()
            for event in custom_exec[eid]:
                if event[0] == "begin":
                    begin_stack.append(event[1])
                else:
                    if len(begin_stack) != 0:
                        begin_tick = begin_stack.pop()
                        end_tick = event[1]
                        duration = end_tick - begin_tick
                        record = dict()
                        record["start"] = begin_tick
                        record["duration"] = duration
                        custom_exec_[eid].append(record)
                    else:
                        continue
        record_nodes = []
        start_tick = 0
        for node in nodes_info:
            node_rec = {
                "name" : node["name"],
                "start" : start_tick,
                "duration" : node[0]
            }
            record = list()
            sum_record = dict()
            for eid in custom_exec_:
                records = custom_exec_[eid]
                for rec in records:
                    if rec["start"] > start_tick and rec["start"] < start_tick + node[0]:
                        record.append({
                            "id" : eid,
                            "start" : rec["start"],
                            "duration" : rec["duration"]
                        })
                        if eid not in sum_record:
                            sum_record[eid] = 0
                        sum_record[eid] = sum_record[eid] + rec["duration"]
            node_rec["record"] = record
            node_rec["sum_record"] = [list([k, v]) for k, v in sum_record.items()]
            record_nodes.append(node_rec)
            start_tick = start_tick + node[0]

        # html
        root = ET.Element("html")
        body = ET.SubElement(root, "body")
        ET.SubElement(body, "div", attrib={"id": "tickgraph_page"})

        script = ET.SubElement(body, "script")
        script.text = d3_script
        script = ET.SubElement(body, "script")
        datasets = f'''
        var datasets = {json.dumps(record_nodes)};
        var custom_color = ["LightCoral", "HotPink", "MediumVioletRed", "Tomato", "Khaki", "Violet", "MediumOrchid", "LawnGreen"];
        '''

        script.text = datasets + r'''
        var nodes_info = {};
        var prev = null;
        function highlight_name(items)
        {
            td = d3.select(items);
            td.style({
                "color" : "yellow"
            });
            name = d3.select("#hightlight_name").text(td.text());
        }

        function moveout_name(items)
        {
            td = d3.select(items);
            td.style({
                "color" : "black"
            });

            if(prev != null)
            {
                text = prev.text();
                d3.select("#hightlight_name").text(text);
            }

        }

        function myclickhandlercustom(rect)
        {
            start = rect.getAttribute("start");
            end = rect.getAttribute("end");
            custom_id = rect.getAttribute("custom_id");
            duration = end - start;
            div = d3.select("#period_div");
            div.selectAll("*").remove();
            tr = div.append("tr");
            tr.append("td").text("start tick:");
            tr.append("td").text(start);

            tr = div.append("tr");
            tr.append("td").text("end tick:");
            tr.append("td").text(end);

            tr = div.append("tr");
            tr.append("td").text("duration:");
            tr.append("td").text(end - start);

            tr = div.append("tr");
            tr.append("td").text("custom id:");
            tr.append("td").text(custom_id);
        }

        function click_draw_tick(items)
        {
            if(prev != null)
            {
                prev.attr("bgcolor", "white");
            }
            name = items.getAttribute("name");
            td = d3.select(items);
            td.attr("bgcolor", "red");
            prev = td;

            div = d3.select("#node_tick_div");
            div.selectAll("*").remove();
            start_tick = nodes_info[td.text()]["start"];
            duration_tick = nodes_info[td.text()]["duration"];
            end_tick = start_tick + duration_tick;

            tr = div.append("tr");
            tr.append("td").text("node start tick:");
            tr.append("td").text(start_tick);
            tr = div.append("tr");
            tr.append("td").text("node end tick:");
            tr.append("td").text(end_tick);
            tr = div.append("tr");
            tr.append("td").text("node cost tick:");
            tr.append("td").text(duration_tick);
            sum_record = nodes_info[td.text()]["sum_record"];
            for(var idx in sum_record)
            {
                content = sum_record[idx];
                tr = div.append("tr");
                tr.append("td").text("eid " + content[0] + " total cycles:");
                tr.append("td").text(content[1]);
            }

            div = d3.select("#period_div");
            div.selectAll("*").remove();

            div = d3.select("#svg_div");
            div.selectAll("*").remove();

            width = 800;
            height = 600;

            svg = div.append("svg").attr("id", "svg_canvas").attr("x", 0).attr("y", 0).attr("width", width).attr("height", height);


            var item_height = 36;
            var unit_name_width = 100;
            padding = {left : 10, right : 10, top : 10, bottom : 10};
            var block_width = width - padding.left - padding.right - unit_name_width;
            period_margin_pad = {top : 4, bottom : 4, left : 10, right : 10};
            period_box_max = block_width - period_margin_pad.left - period_margin_pad.right;
            box_height = item_height - period_margin_pad.top - period_margin_pad.bottom;


            block = svg.append("svg").attr("x", padding.left).attr("y", padding.top + item_height * 0).attr("width", unit_name_width).attr("height", item_height).attr("id", "ticks_header");
            block.append("line").attr("x1", 0).attr("x2", unit_name_width).attr("y1", item_height).attr("y2", item_height)
                .attr("stroke", "grey").attr("stroke-width", "1px").attr("stroke-dasharray", "5px");
            block = svg.append("svg").attr("x", padding.left + unit_name_width).attr("y", padding.top + item_height * 0).attr("width", block_width).attr("height", item_height)
                .attr("id", "ticks_block");
            block.append("line").attr("x1", 0).attr("x2", block_width).attr("y1", item_height).attr("y2", item_height)
                .attr("stroke", "grey").attr("stroke-width", "1px").attr("stroke-dasharray", "5px");

            tec_blocks = [];

            data = nodes_info[td.text()];
            data = data.record

            linear = d3.scale.linear().domain([start_tick, end_tick]).range([0, period_box_max]);
            linear1 = d3.scale.linear().domain([0, period_box_max]).range([start_tick, end_tick]);

            tick_line_num = 40;
            tick_block = d3.select("#ticks_block");
            tick_len0 = 4;
            tick_len1 = 8;
            scale = Math.round(period_box_max / tick_line_num);

            for(var i = 0; i < tick_line_num + 1; i++)
            {
                var len = tick_len0;
                if(i % 10 == 0)
                {
                    len = tick_len1;
                    var value = Math.round(linear1(scale * i));
                    tick_block.append("text").text(value).attr("x", scale * i)
                        .attr("y", 25).attr("font-size", "16");
                }
                tick_block.append("line").attr("x1", scale * i).attr("x2", scale * i).attr("y1", item_height - len).attr("y2", item_height).attr("stroke", "grey").attr("stroke-width", "1px");
            }

            tec_num = 1
            for(var i = 0; i < tec_num; i++)
            {
                block = svg.append("svg").attr("x", padding.left).attr("y", padding.top + item_height * (i + 1)).attr("width", unit_name_width).attr("height", item_height).attr("id", "tec" + i);
                block.append("line").attr("x1", unit_name_width).attr("x2", unit_name_width).attr("y1", 0).attr("y2", item_height).attr("stroke", "grey").attr("stroke-width", "1px").attr("stroke-dasharray", "5px");
                block.append("line").attr("x1", 0).attr("x2", unit_name_width).attr("y1", item_height).attr("y2", item_height).attr("stroke", "grey").attr("stroke-width", "1px").attr("stroke-dasharray", "5px");

                block = svg.append("svg").attr("x", padding.left + unit_name_width).attr("y", padding.top + item_height * (i + 1)).attr("width", block_width).attr("height", item_height).attr("id", "tec_tick" + i);
                tec_blocks.push(block);
                block.append("line").attr("x1", 0).attr("x2", block_width).attr("y1", item_height).attr("y2", item_height).attr("stroke", "grey").attr("stroke-width", "1px").attr("stroke-dasharray", "5px");
            }

            header = d3.select("#ticks_header");
            header.append("text").attr("x", 0).attr("y", 20).attr("font-size", "16").text("");
            for(var i = 0; i < tec_num; i++)
            {
                header = d3.select("#tec" + i);
                header.append("text").attr("x", 24).attr("y", 22).attr("font-size", "16").text("tec" + i);
            }


            for(var tec_id = 0; tec_id < tec_num; tec_id++)
            {
                x_offset = period_margin_pad.left;
                y_offset = period_margin_pad.top;

                periods = data;
                for(var idx in  data)
                {
                    context = data[idx];
                    eid = context.id
                    var color = eid % custom_color.length;
                    color = custom_color[color];


                    start = context.start;
                    end = context.duration + start;
                    x_start = linear(start) + x_offset;
                    x_end = linear(end) + x_offset;

                    rect = tec_blocks[tec_id].append("rect").attr("x", x_start).attr("y", y_offset).attr("width", x_end - x_start).attr("height", box_height).attr("style", "fill:" + color).attr("onclick", "myclickhandlercustom(this)").attr("start", start).attr("end", end).attr("custom_id", eid);
                }
            }
        }

        function draw_tick_graph_page()
        {
            document.getElementById("tickgraph_page").hidden = false;
            page = d3.select("#tickgraph_page");
            page.selectAll("*").remove();
            nodes_info = {}
            for(var id in datasets)
            {
                node = datasets[id];
                name = node["name"];
                nodes_info[name] = node;
            }

            left_div = page.append("div").style(
                {
                    "top": 0,
                    "width": 200,
                    "height": "100%",
                }
            );

            div = left_div.append("div").style(
                {
                    "width": 200,
                    "height": 800,
                    "position": "absolute",
                    "left" : 0,
                    "top" : 40,
                    "overflow-x": "scroll",
                    "overflow-y": "scroll",
                }
            );

            table = div.append("table").attr("border", 1).style(
                {
                    "font-size" : 13
                }
            );

            for(var id in datasets)
            {
                node = datasets[id];
                name = node["name"];
                table.append("tr").append("td").attr("onmouseover", "highlight_name(this)").attr("onmouseout", "moveout_name(this)").attr("onclick", "click_draw_tick(this)").attr("name", name).text(name);
            }

            right_div = page.append("div").style(
                {
                    "position": "absolute",
                    "top" : 0,
                    "left" : 220,
                    "width": 800,
                    "height": "100%",
                }
            );

            div = right_div.append("div").style(
                {
                    "position": "relative",
                    "text-align": "center",
                    "height" : 40,
                    "font-size" : 22
                }
            );

            div.append("text").attr("id", "hightlight_name").text("");


            div = right_div.append("div").attr("id", "svg_div").style(
                {
                    "position": "relative",
                    "height" : 600,
                    "width" : 800,
                    "float" : "left",
                });

            div = right_div.append("div").attr("id", "node_tick_div");
            div = right_div.append("div").attr("id", "period_div");
        }

        draw_tick_graph_page();
            '''

        with open(report_name, "w") as f:
            ET.ElementTree(root).write(f, encoding="unicode", method="html")