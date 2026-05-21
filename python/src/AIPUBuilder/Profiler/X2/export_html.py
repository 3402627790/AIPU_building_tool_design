"""
//-------------------------------------------------------------------------------
// This file is CONFIDENTIAL and any use by you is subject to the terms of the
// agreement between you and Arm China or the terms of the agreement between you
// and the party authorised by Arm China to disclose this file to you.
// The confidential and proprietary information contained in this file may only
// be used by a person authorised under and to the extent permitted by a
// subsisting licensing agreement from Arm China.
//
//        (C) Copyright 2022 Arm Technology (China) Co. Ltd.
//                    All rights reserved.
//
// This entire notice must be reproduced on all copies of this file and copies of
// this file may only be made by a person if such person is permitted to do so
// under the terms of a subsisting license agreement from Arm China.
//
//--------------------------------------------------------------------------------
"""

# cython: language_level=3
# !/usr/bin/python
# -*- coding: UTF-8 -*-

import json
import math
import numpy as np

from xml.etree import ElementTree as ET
from ..js_script import d3_script

from AIPUBuilder.core import Target, Profiler

css_style = r'''

html {
    scrollbar-face-color: #99b;
    scrollbar-arrow-color: #fff;
    scrollbar-highlight-color: #fff;
    scrollbar-shadow-color: #99c;
    scrollbar-3dlight-color: #000;
}

body {
    margin: 0;
    padding: 0;
    background: #f0f0ff;
    color: #000;
    font: 12px "Lucida Grande", "Helvetica Neue", Helvetica, "Arial Unicode MS", "Arial Unicode", Arial, sans-serif;
    color: #333;
    text-align: center;
}

#wrapper {
    position: relative;
    width: 770px;
    margin: 0 auto;
    background: #f8f8ff top center repeat-y;
    text-align: left;
    z-index: 10;
}

#header {
    clear: both;
    background: #EAFAF1;
    position: relative;
}

h1.sitetitle {
    font-size: 32px;
    margin: 8px 0 15px;
    border-top: 1px;
    text-align: center;
}

.titlebar {
    font-size: x-large;
    font-weight: bold;
    border: silver solid 1px;
    border-width: 1px 0;
    margin: 8px 0 12px 0;
    padding: 3px 0;
    line-height: 1.3;
    text-align: center;
}

.summary_table_page {
    text-align: center;
}

#menu {
    margin: 0;
    padding: 0;
    clear: both;
    border-top: 1px white solid;
    list-style: none inside;
    display: flex;
}

#menu button {
    font-size: large;
    flex-grow: 1;
    background: #EAECEE;
    border: solid white;
    text-align: center;
    display: inline-block;
    padding: 6px 2px;
    cursor: pointer;
}

#menu button:hover {
    background: #1C2833;
    border-color: silver;
    color: white;
}

#menu button:focus {
    font-weight: bolder;
    color: #1E8449;
}

.main_table {
    text-align: left;
    margin-left: auto;
    margin-right: auto;
    width: fit-content;
}

table {
    border-collapse: collapse;
    border-spacing: 0;
    border: 1px solid #ddd;
    table-layout: fixed;
    word-wrap: break-word;
}

th, td {
    text-align: right;
    padding: 8px;
}

tr:nth-child(even){
    background-color: #f2f2f2
}

tr:hover {
    background-color: #F7DC6F;
}

.center {
    margin-left: auto;
    margin-right: auto;
}
'''

js_func_script = r'''

var nodes_info = {};
var prev = null;
var start_tick_cycle = null;
var end_tick_cycle = null;

function hidden_all_show_divs()
{
    var show_divs_name = ["table_page", "task_page", "multi_core_page", "tickgraph_page", "summary_table_page", "analysis_page"];
    for (var show_div_id in show_divs_name)
    {
        div_name = show_divs_name[show_div_id];
        div_obj = d3.select("#" + div_name);
        //div_obj.selectAll("*").remove();
        div_obj.style("display", "none");
    }
}


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

function myclickhandler(rect)
{
    start = rect.getAttribute("start");
    end = rect.getAttribute("end");
    read_bytes = rect.getAttribute("read_bytes");
    write_bytes = rect.getAttribute("write_bytes");
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
    tr.append("td").text("read bytes:");
    tr.append("td").text(read_bytes);
    tr = div.append("tr");
    tr.append("td").text("write bytes:");
    tr.append("td").text(write_bytes);
}

function myclickhandler_aiff(rect)
{
    start = rect.getAttribute("start");
    end = rect.getAttribute("end");
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
    tr.append("td").text("mtp0 busy cycles:");
    tr.append("td").text(rect.getAttribute("mtp0"));
    
    tr = div.append("tr");
    tr.append("td").text("mtp1 busy cycles:");
    tr.append("td").text(rect.getAttribute("mtp1"));

    tr = div.append("tr");
    tr.append("td").text("ptp busy cycles:");
    tr.append("td").text(rect.getAttribute("ptp"));

    let map = new Map();
    map.set("itp_busy_cycle", "itp busy cycles:");
    map.set("itp0_busy_cycle", "itp0 busy cycles:");
    map.set("itp1_busy_cycle", "itp1 busy cycles:");
    map.set("wrb_busy_cycle", "wrb busy cycles:");
    map.set("physical_aiff_id", "physical aiff id:");
    for (let [key, text_str] of map)
    {
        if (rect.hasAttribute(key))
        {
            tr = div.append("tr");
            tr.append("td").text(text_str);
            tr.append("td").text(rect.getAttribute(key));
        }
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

var width = 1024;
var height = 600;


function click_draw_tick(items)
{
    if(prev != null)
    {
        prev.attr("bgcolor", "white");
    }
    td = d3.select(items);
    td.attr("bgcolor", "red");
    prev = td;

    div = d3.select("#node_tick_div");
    div.selectAll("*").remove();
    start_tick = nodes_info[td.text()]["start"];
    end_tick = nodes_info[td.text()]["end"];

    tr = div.append("tr");
    tr.append("td").text("node start tick:");
    tr.append("td").text(start_tick);
    tr = div.append("tr");
    tr.append("td").text("node end tick:");
    tr.append("td").text(end_tick);
    tr = div.append("tr");
    tr.append("td").text("node cost tick:");
    tr.append("td").text(end_tick - start_tick);


    div = d3.select("#summary_tick_div");
    div.selectAll("*").remove();

    sum_head = nodes_info[td.text()].stamp[0].custom_sum;
    for (var tec_id = 0; tec_id < tec_num; tec_id++)
    {
        stamp_sum = nodes_info[td.text()].stamp[tec_id].custom_sum;
        if(tec_id == 0)
        {
            tr = div.append("tr");
            tr.append("td").text("");
            for(var eid in sum_head)
            {
                tr.append("td").text("custom " + eid);
            }
        }
        tr = div.append("tr");
        tr.append("td").text("tec " + tec_id)
        for(var eid in sum_head)
        {
            custom_sum = stamp_sum[eid];
            tr.append("td").text(custom_sum);
        }
    }

    div = d3.select("#period_div");
    div.selectAll("*").remove();

    div = d3.select("#svg_div");
    div.selectAll("*").remove();


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
    dma_blocks = [];
    aiff_blocks = [];

    data = nodes_info[td.text()]["stamp"];
    core_id = nodes_info[td.text()]["core_id"];

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

    for(var i = 0; i < tec_num; i++)
    {
        block = svg.append("svg").attr("x", padding.left).attr("y", padding.top + item_height * (i + 1)).attr("width", unit_name_width).attr("height", item_height).attr("id", "tec" + i);
        block.append("line").attr("x1", unit_name_width).attr("x2", unit_name_width).attr("y1", 0).attr("y2", item_height).attr("stroke", "grey").attr("stroke-width", "1px").attr("stroke-dasharray", "5px");
        block.append("line").attr("x1", 0).attr("x2", unit_name_width).attr("y1", item_height).attr("y2", item_height).attr("stroke", "grey").attr("stroke-width", "1px").attr("stroke-dasharray", "5px");

        block = svg.append("svg").attr("x", padding.left + unit_name_width).attr("y", padding.top + item_height * (i + 1)).attr("width", block_width).attr("height", item_height).attr("id", "tec_tick" + i);
        tec_blocks.push(block);
        block.append("line").attr("x1", 0).attr("x2", block_width).attr("y1", item_height).attr("y2", item_height).attr("stroke", "grey").attr("stroke-width", "1px").attr("stroke-dasharray", "5px");

        block = svg.append("svg").attr("x", padding.left).attr("y", padding.top + item_height * (i + 1 + tec_num)).attr("width", unit_name_width).attr("height", item_height).attr("id", "aiff" + i);
        block.append("line").attr("x1", unit_name_width).attr("x2", unit_name_width).attr("y1", 0).attr("y2", item_height).attr("stroke", "grey").attr("stroke-width", "1px").attr("stroke-dasharray", "5px");
        block.append("line").attr("x1", 0).attr("x2", unit_name_width).attr("y1", item_height).attr("y2", item_height).attr("stroke", "grey").attr("stroke-width", "1px").attr("stroke-dasharray", "5px");

        block = svg.append("svg").attr("x", padding.left + unit_name_width).attr("y", padding.top + item_height * (i + 1 + tec_num)).attr("width", block_width).attr("height", item_height).attr("id", "aiff_tick" + i);
        aiff_blocks.push(block);
        block.append("line").attr("x1", 0).attr("x2", block_width).attr("y1", item_height).attr("y2", item_height).attr("stroke", "grey").attr("stroke-width", "1px").attr("stroke-dasharray", "5px");


        block = svg.append("svg").attr("x", padding.left).attr("y", padding.top + item_height * (i + 1 + 2 * tec_num)).attr("width", unit_name_width).attr("height", item_height).attr("id", "dma" + i);
        block.append("line").attr("x1", unit_name_width).attr("x2", unit_name_width).attr("y1", 0).attr("y2", item_height).attr("stroke", "grey").attr("stroke-width", "1px").attr("stroke-dasharray", "5px");
        block.append("line").attr("x1", 0).attr("x2", unit_name_width).attr("y1", item_height).attr("y2", item_height).attr("stroke", "grey").attr("stroke-width", "1px").attr("stroke-dasharray", "5px");

        block = svg.append("svg").attr("x", padding.left + unit_name_width).attr("y", padding.top + item_height * (i + 1 + 2 * tec_num)).attr("width", block_width).attr("height", item_height).attr("id", "dma_tick" + i);
        dma_blocks.push(block);
        block.append("line").attr("x1", 0).attr("x2", block_width).attr("y1", item_height).attr("y2", item_height).attr("stroke", "grey").attr("stroke-width", "1px").attr("stroke-dasharray", "5px");
    }

    header = d3.select("#ticks_header");
    header.append("text").attr("x", 0).attr("y", 20).attr("font-size", "16").text("core" + core_id);
    for(var i = 0; i < tec_num; i++)
    {
        header = d3.select("#tec" + i);
        header.append("text").attr("x", 24).attr("y", 22).attr("font-size", "16").text("tec" + i);

        header = d3.select("#aiff" + i);
        header.append("text").attr("x", 24).attr("y", 22).attr("font-size", "16").text("aiff" + i);

        header = d3.select("#dma" + i);
        header.append("text").attr("x", 24).attr("y", 22).attr("font-size", "16").text("dma" + i);
    }


    for(var tec_id = 0; tec_id < tec_num; tec_id++)
    {
        x_offset = period_margin_pad.left;
        y_offset = period_margin_pad.top;

        periods = data[tec_id]["aiff"];
        for(var period in periods)
        {
            start = periods[period].start;
            end = start + periods[period].duration;
            x_start = linear(start) + x_offset;
            x_end = linear(end) + x_offset;
            mtp0 = periods[period].mtp0_busy_cycle;
            mtp1 = periods[period].mtp1_busy_cycle;
            ptp = periods[period].ptp_busy_cycle;
            element = aiff_blocks[tec_id].append("rect").attr("x", x_start).attr("y", y_offset).attr("width", x_end - x_start).attr("height", box_height).attr("style", "fill:rgb(248, 9, 121)")
            .attr("onclick", "myclickhandler_aiff(this)").attr("start", start).attr("end", end).attr("mtp0", mtp0).attr("mtp1", mtp1).attr("ptp", ptp);
            const properties = ["itp_busy_cycle", "itp0_busy_cycle", "itp1_busy_cycle", "wrb_busy_cycle", "physical_aiff_id"];
            for (let property of properties)
            {
                if (periods[period].hasOwnProperty(property))
                {
                    property_value = periods[period][property];
                    element = element.attr(property, property_value);
                }
            }
        }

        periods = data[tec_id]["dma"];

        for(var period in periods)
        {
            start = periods[period].start;
            end = start + periods[period].duration;
            read_bytes = periods[period].read_bytes;
            write_bytes = periods[period].write_bytes;
            x_start = linear(start) + x_offset;
            x_end = linear(end) + x_offset;

            rect = dma_blocks[tec_id].append("rect").attr("x", x_start).attr("y", y_offset).attr("width", x_end - x_start).attr("height", box_height).attr("style", "fill:rgb(248, 9, 121)")
            .attr("onclick", "myclickhandler(this)").attr("start", start).attr("end", end).attr("read_bytes", read_bytes).attr("write_bytes", write_bytes);
        }

        periods = data[tec_id]["custom"];
        custom_sum = data[tec_id]["custom_sum"];
        for(var eid in  periods)
        {
            var color = eid % custom_color.length;
            color = custom_color[color];
            for(var period in periods[eid])
            {
                start = periods[eid][period].start;
                end = start + periods[eid][period].duration;
                x_start = linear(start) + x_offset;
                x_end = linear(end) + x_offset;

                rect = tec_blocks[tec_id].append("rect").attr("x", x_start).attr("y", y_offset).attr("width", x_end - x_start).attr("height", box_height).attr("style", "fill:" + color).attr("onclick", "myclickhandlercustom(this)").attr("start", start).attr("end", end).attr("custom_id", eid);
            }
        }
    }
}

function draw_tick_graph_page()
{
    hidden_all_show_divs();
    page = d3.select("#tickgraph_page");
    page.style("display", "block");
    page.selectAll("*").remove();

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
            "top" : 0,
            "overflow-x": "scroll",
            "overflow-y": "scroll",
        }
    );

    table = div.append("table").attr("border", 1).style(
        {
            "font-size" : 13
        }
    );

    for(var id in nodes_info)
    {
        node = nodes_info[id];
        name = node["Layer Name"];

        table.append("tr").append("td").attr("onmouseover", "highlight_name(this)").attr("onmouseout", "moveout_name(this)").attr("onclick", "click_draw_tick(this)").text(name);
    }

    right_div = page.append("div").style(
        {
            "position": "absolute",
            "top" : 100,
            "left" : 400,
            "width": width,
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
            "height" : height,
            "width" : width,
            "float" : "left",
        });

    bar_info_div = right_div.append("div").attr("id", "bar_info_div").style(
    {
        "position": "relative",
        "float" : "left",
    });

    div = bar_info_div.append("div").attr("id", "node_tick_div").style(
    {
        "position": "relative",
        "float" : "left",
    });
    div = bar_info_div.append("div").attr("id", "period_div").style(
    {
        "position": "relative",
    });

    div = right_div.append("div").attr("id", "summary_tick_div").style(
    {
        "position": "relative",
        "float" : "left",
    });
}

function draw_header()
{
    header = d3.select("#wrapper");
    header.selectAll("*").remove();
    header = header.append("div").attr("id", "header")
    header.append("h1").attr("class", "sitetitle").text("Profiler Report");
    button_block = header.append("ul").attr("id", "menu");
    detail_button = button_block.append("button").attr("id", "detailed_data_button").text("DETAILED DATA");
    detail_button.on("click", draw_table_page);
    detail_button.style("font-weight", "bolder");
    button_block.append("button")
        .on("click", draw_table_task)
        .style("font-weight", "bolder")
        .text("TASK DATA");
    if(core_num > 1 && verbose > 0)
    {
        detail_button = button_block.append("button").attr("id", "multi_core_button").text("MULTI CORE DATA");
        detail_button.on("click", draw_multi_core_table);
        detail_button.style("font-weight", "bolder");
    }
    detail_button = button_block.append("button").attr("id", "summary_data_button").text("SUMMARY");
    detail_button.on("click", draw_summary_table);
    detail_button.style("font-weight", "bolder");
    if(verbose > 0)
    {
        button = button_block.append("button").attr("id", "tickgraph_button").style("font-weight", "bolder").text("TICK GRAPH DATA");
        button.on("click", draw_tick_graph_page);
        button.style("font-weight", "bolder");
    }

    if (verbose > 1)
    {
        analysis_button = button_block.append("button").attr("id", "analysis_button").style("font-weight", "bolder").text("ANALYSIS DATA");
        analysis_button.on("click", draw_analysis_page);
        analysis_button.style("font-weight", "bolder");
    }
}


function draw_multi_core_table()
{
    hidden_all_show_divs();
    table_page = d3.select("#multi_core_page");
    table_page.style("display", "block");
    table_page.selectAll("*").remove();

    table_page.append("p").attr("class", "titlebar").text("Mult Core Data");
    ul = table_page.append("ul").attr("id", "menu");
    for(i = 0; i < core_num; i++)
    {
        core_button = ul.append("button").attr("core_id", i).text("Core" + i).attr("onclick", "click_draw_core_info(" + i + ")");
    }
    ul.append("button").text("Summary").attr("onclick", "click_draw_core_summary()");
    table_page.append("table").attr("id", "core_table").attr("border", 1).attr("class", "center");
    click_draw_core_summary();
}

function get_core_subgraphs(core_id)
{
    subgraphs = [];
    for(var subgraph_id in datasets.subgraphs)
    {
        subgraph = datasets.subgraphs[subgraph_id];
        if(subgraph["Core ID"] == core_id)
        {
            subgraphs.push(subgraph);
        }
    }
    return subgraphs
}

function get_subgraphs_info(subgraphs)
{
    infos = {};

    busy_cycles = 0;
    ops = 0;
    aiff_busy = 0;
    dma_busy = 0;
    aipu_read = 0;
    aipu_write = 0;
    dma_read = 0;
    dma_write = 0;
    subgraphs_num = subgraphs.length;
    layers_num = 0;

    for(var sub_id in subgraphs)
    {
        subgraph = subgraphs[sub_id];
        node_num = subgraph.nodes.length;

        layers_num = layers_num + node_num;
        busy_cycles = busy_cycles + (subgraph.nodes[subgraph.nodes.length - 1]["end"] - subgraph.nodes[0]["start"]);
        for(var node_id in subgraph.nodes)
        {
            node = subgraph.nodes[node_id];

            ops += node["Algorithm Ops(Million)"];
            aiff_busy += node["AIFF Busy Cycles"];

            dma_busy += node["DMA Busy Cycles"];
            aipu_read += node["AIPU Read Bytes"];
            aipu_write += node["AIPU Write Bytes"];
            dma_read += node["DMA Read Bytes"];
            dma_write += node["DMA Write Bytes"];
        }
    }

    model_cycles = end_tick_cycle - start_tick_cycle;
    idle_cycles = model_cycles - busy_cycles;
    mac_util = ops * 1000000 / (mac_num * model_cycles * 2);
    mac_util_except_idle = ops * 1000000 / (mac_num * busy_cycles * 2);

    console.log(end_tick_cycle, start_tick_cycle);
    infos["Total busy cycles"] = busy_cycles;
    infos["Total idle cycles"] = idle_cycles
    infos["total Ops(Mops)"] = ops.toFixed(3);
    infos["Total AIFF Busy Cycles"] = aiff_busy;
    infos["Total DMA Busy Cycles"] = dma_busy;
    infos["AIPU Read Bytes"] = aipu_read;
    infos["AIPU Write Bytes"] = aipu_write;
    infos["DMA Read Bytes"] = dma_read;
    infos["DMA Write Bytes"] = dma_write;

    infos["Subgraphs Num"] = subgraphs_num;
    infos["Layers Num"] = layers_num;
    infos["MAC utilization"] = (mac_util * 100).toFixed(2) + "%";
    infos["MAC utilization(except idle cycles)"] = (mac_util_except_idle * 100).toFixed(2) + "%";
    return infos
}


function click_draw_core_summary()
{
    table = d3.select("#core_table");
    table.selectAll("*").remove();

    //subgraphs = get_core_subgraphs(core_id);
    tr = table.append("tr");
    tr.append("th").text("");
    for(var i = 0; i < core_num; i++)
    {
        tr.append("th").text("Core " + i);
    }

    multi_core_items = ["Total busy cycles", "Total idle cycles", "total Ops(Mops)", "Total AIFF Busy Cycles", "Total DMA Busy Cycles",
                        "AIPU Read Bytes", "AIPU Write Bytes", "DMA Read Bytes", "DMA Write Bytes", 
                        "Subgraphs Num", "Layers Num", "MAC utilization", "MAC utilization(except idle cycles)"];
    subgraph_infos = []
    for(var i = 0; i < core_num; i++)
    {
        subgraphs = get_core_subgraphs(i);
        subgraph_infos.push(get_subgraphs_info(subgraphs));
    }
    console.log(subgraph_infos)
    for(var header_id in multi_core_items)
    {
        var header = multi_core_items[header_id];
        tr = table.append("tr");
        tr.append("td").text(header);
        for(var i = 0; i < core_num; i++)
        {
            tr.append("td").text(subgraph_infos[i][header]);
        }
    }
}

function click_draw_core_info(core_id)
{
    subgraphs = get_core_subgraphs(core_id);

    table = d3.select("#core_table");
    table.selectAll("*").remove();
    tr = table.append("tr");
    headers = [];
    for(var header_id in datasets.items)
    {
        header = datasets.items[header_id];
        headers.push(header);
        tr.append("th").text(header);
    }

    for(var sub_id in subgraphs)
    {
        subgraph = subgraphs[sub_id];
        node_num = subgraph.nodes.length;
        var running_core = subgraph["Core ID"];

        for(var node_id in subgraph.nodes)
        {
            tr = table.append("tr");
            node = subgraph.nodes[node_id];
            for(var header_id in headers)
            {
                header = headers[header_id];
                if(header == "Core ID")
                {
                    if(node_id == 0)
                    {
                        td = tr.append("td").attr("rowspan", node_num).text(running_core);
                    }
                    continue;
                }
                td = tr.append("td");
                val = node[header];
                if(header.search("Utilization") != -1)
                {
                    val = (val * 100).toFixed(2) + "%";
                }
                td.text(val);
            }
        }
        gap = 0;
        sub_id = Number(sub_id);
        if(sub_id != subgraphs.length - 1)
        {
            gap = subgraphs[sub_id + 1].nodes[0]["start"] - subgraph.nodes[subgraph.nodes.length - 1]["end"];
        }
        else
        {
            gap = end_tick_cycle - subgraph.nodes[subgraph.nodes.length - 1]["end"];
        }
        tr = table.append("tr");
        td = tr.append("td").text("Idle Cycle");
        tr.append("td").attr("colspan", headers.length - 1).text(gap);
    }
}

function draw_table_task()
{
    hidden_all_show_divs()

    table_page = d3.select("#task_page")
    table_page.style("display", "block")
    table_page.selectAll("*").remove()

    table_page.append("p").attr("class", "titlebar").text("Task Data")
    const table = table_page.append("table")
        .attr("id", "main_table")
        .attr("border", 1)
        .attr("class", "center")
    headers = ['Task ID', 'Core ID', 'Node Num', 'Start At (Cycles)', 'Stop At (Cycles)', 'Duration (Cycles)']
    nodes_info = {}

    const tr = table.append("tr")
    for (let header of headers) {
        tr.append("th").text(header)
    }

    for (let subgraph_id in datasets.subgraphs) {
        const subgraph = datasets.subgraphs[subgraph_id]
        const tr = table.append("tr")
        tr.append("td").text(subgraph["Task ID"])
        tr.append("td").text(subgraph["Core ID"])
        tr.append("td").text(subgraph.nodes.length)
        const start = Number(subgraph.nodes.at(0)["start"])
        const end = Number(subgraph.nodes.at(-1)["end"])
        tr.append("td").text(start.toLocaleString())
        tr.append("td").text(end.toLocaleString())
        tr.append("td").text((end - start).toLocaleString())
    }
}

function draw_table_page()
{
    hidden_all_show_divs();

    table_page = d3.select("#table_page");
    table_page.style("display", "block");
    table_page.selectAll("*").remove();

    table_page.append("p").attr("class", "titlebar").text("Detailed Data");
    table = table_page.append("table").attr("id", "main_table").attr("border", 1);
    tr = table.append("tr");
    headers = [];
    nodes_info = {};
    for(var header_id in datasets.items)
    {
        header = datasets.items[header_id];
        headers.push(header);
        tr.append("th").text(header);
    }

    for(var subgraph_id in datasets.subgraphs)
    {
        subgraph = datasets.subgraphs[subgraph_id];
        //color = subgraph.color;
        node_num = subgraph.nodes.length;
        var running_core = subgraph["Core ID"];


        start = subgraph.nodes[0]["start"];
        end = subgraph.nodes[subgraph.nodes.length - 1]["end"];
        if(start_tick_cycle == null)
        {
            start_tick_cycle = start;
        }
        else
        {
            start_tick_cycle = Math.min(start_tick_cycle, start)
        }
        if(end_tick_cycle == null)
        {
            end_tick_cycle = end;
        }
        else
        {
            end_tick_cycle = Math.max(end_tick_cycle, end)
        }

        for(var node_id in subgraph.nodes)
        {
            tr = table.append("tr");
            node = subgraph.nodes[node_id];
            for(var header_id in headers)
            {
                header = headers[header_id];

                if(header == "Core ID") {
                    val = running_core;
                } else {
                    val = node[header];
                }

                if(!isNaN(val)) {
                    if (header == "L2 Cache Reduced Footprint") {
                        val = Math.max(val, 0).toLocaleString();
                    } else if(header.search("Utilization") != -1 || header.search("Rate") != -1) {
                        val = (val * 100).toFixed(2) + "%";
                    } else {
                        val = val.toLocaleString();
                    }
                }

                if(header == "Core ID" || header == "Task ID" ||  header == "Level") {
                    if(node_id == 0) {
                        tr.append("td").attr("rowspan", node_num).text(val);
                    } else {
                        ;
                    }
                } else {
                    tr.append("td").text(val);
                }
            }
            nodes_info[node["Layer Name"]] = node;
        }
    }
}

function draw_summary_table()
{
    hidden_all_show_divs();

    page = d3.select("#summary_table_page");
    page.style("display", "block");
    page.selectAll("*").remove();
    page.append("p").attr("class", "titlebar").text("Summary");

    table = page.append("table").attr("class", "center");
    for(var id in datasets.summaries)
    {
        summary_header = datasets.summaries[id];
        tr = table.append("tr");
        tr.append("td").text(summary_header);
        tr.append("td").text(datasets[summary_header]);
    }
}

function draw_analysis_page()
{
    hidden_all_show_divs();

    page = d3.select("#analysis_page");
    page.style("display", "block");
    page.selectAll("*").remove();
    
    page.append("p").attr("class", "titlebar").text("Openning new tab......");
    window.open(page.attr("path")); 
}

draw_header();
draw_table_page();
'''


def export_html(report_name, perf_data, verbose):
    for subgraph in perf_data["subgraphs"]:
        color = list(np.random.choice(range(256), size=3))
        subgraph["color"] = "#" + "".join([hex(c)[2:] for c in color])
    # html
    root = ET.Element("html")
    title = ET.SubElement(root, "title")
    title.text = report_name
    style = ET.SubElement(root, "style", attrib={
                          "type": "text/css", "media": "all"})
    style.text = css_style
    body = ET.SubElement(root, "body")
    ET.SubElement(body, "div", attrib={"id": "wrapper"})
    ET.SubElement(body, "div", attrib={"id": "table_page", "class": "hidden"})
    ET.SubElement(body, "div", attrib={"id": "task_page", "class": "hidden"})
    ET.SubElement(body, "div", attrib={
                  "id": "summary_table_page", "class": "hidden"})
    ET.SubElement(body, "div", attrib={
                  "id": "tickgraph_page", "class": "hidden"})
    ET.SubElement(body, "div", attrib={
                  "id": "multi_core_page", "class": "hidden"})
    ET.SubElement(body, "div", attrib={"id": "analysis_page", "class": "hidden",
                                       "path": report_name.replace(".html", "_verbose2_done.html")})

    script = ET.SubElement(body, "script")
    script.text = d3_script
    script = ET.SubElement(body, "script")
    perf_data = perf_data.copy()

    for subgraph in perf_data["subgraphs"]:
        nodes = subgraph["nodes"]
        for node in nodes:
            stamp = node["stamp"]
            for tec in stamp:
                customs = tec["custom"]
                summary = dict()

                for eid in customs:
                    total_cycle = 0
                    for record in customs[eid]:
                        total_cycle += record["duration"]
                    summary[eid] = total_cycle
                tec["custom_sum"] = summary

    datasets = f'''
    var tec_num = {perf_data["tec_num"]};
    var core_num = {perf_data["core_num"]};
    var verbose = {verbose};
    var datasets = {json.dumps(perf_data)};
    var mac_num = {Target().ff_mac_num};
    var custom_color = ["LightCoral", "HotPink", "MediumVioletRed", "Tomato", "Khaki", "Violet", "MediumOrchid", "LawnGreen"];
    '''
    script.text = datasets + js_func_script

    with open(report_name, "w") as f:
        ET.ElementTree(root).write(f, encoding="unicode", method="html")
