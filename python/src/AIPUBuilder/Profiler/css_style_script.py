style_css_code=r'''
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

/*
 * Main Structure
 */

#wrapper {
    position: relative;
    width: 770px;
    margin: 0 auto;
    background: #f8f8ff top center repeat-y;
    text-align: left;
    z-index: 10;
}

/* Header */

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

/* Navigator */

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

/* Table */

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

/* summary table */
.show_summary_table {
    text-align: center;
}

/* main table */
.show_main_table {
    text-align: left;
    margin-left: auto;
    margin-right: auto;
    width: fit-content;
}

/* chart when setting verbose to 1 or above */
.show_verbose_chart {
    width:1280px;
    margin:0px auto;
}

.svg_graph {
    width:1280px;
    margin:0px auto;
}

#sidebar {
    position: fixed;
    width: 500px;
    height: auto;
    padding: 5px;
    background-color: #F8F9F9;
    border: 1px solid #ccc;
    -webkit-border-radius: 10px;
    -moz-border-radius: 10px;
    border-radius: 10px;
    font-size: medium;
    -webkit-box-shadow: 4px 4px 10px rbga(0, 0, 0, 0.4);
    -moz-box-shadow: 4px 4px 10px rbga(0, 0, 0, 0.4);
    box-shadow: 4px 4px 10px rbga(0, 0, 0, 0.4);
    pointer-events: none;
}

#tooltip {
    position: absolute;
    width: 300px;
    height: auto;
    padding: 5px;
    background-color: #F4F6F6;
    border: 1px solid #ccc;
    -webkit-border-radius: 10px;
    -moz-border-radius: 10px;
    border-radius: 10px;
    font-size: medium;
    -webkit-box-shadow: 4px 4px 10px rbga(0, 0, 0, 0.4);
    -moz-box-shadow: 4px 4px 10px rbga(0, 0, 0, 0.4);
    box-shadow: 4px 4px 10px rbga(0, 0, 0, 0.4);
    pointer-events: none;
}

#tooltip.hidden {
    display: none;
}


table {
    border-collapse: collapse;
    border-spacing: 0;
    border: 1px solid #ddd;
    table-layout: fixed;
    word-wrap: break-word;
}

th, td {
    text-align: left;
    padding: 8px;
}

tr:nth-child(even){
    background-color: #f2f2f2
}

tr:hover {
    background-color: #F7DC6F;
}

td.slash_ignore {
    background-image: url("data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAlIiBoZWlnaHQ9IjEwMCUiPjxsaW5lIHgxPSIwIiB5MT0iMTAwJSIgeDI9IjEwMCUiIHkyPSIwIiBzdHJva2U9ImJsYWNrIiBzdHJva2Utd2lkdGg9IjEiLz48L3N2Zz4=")
}

/* common */

.center {
    margin-left: auto;
    margin-right: auto;
}

.hidden {
    display: none;
}
'''