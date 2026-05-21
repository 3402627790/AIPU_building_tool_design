import argparse
import os
from databus import Databus
from config.parser import Parser
from backend.backend import Backend
from quantizer.qtlib import QtlibQuantizer

databus = Databus()


def parse_args():
    parser = argparse.ArgumentParser(
        prog="NodeAttributesTuningTools",
        description="This tools aims to automatically search potential optimized values of node attributes and give accuracy feedback to user",
    )
    parser.add_argument("-f", "--field", action="store_true", default=False, help="List config file fields")
    parser.add_argument("-p", "--plugin", action="store_true", default=False,
                        help="List plugins of aipu APIs, datasets, metric and optimization strategy")
    parser.add_argument("-c", "--config", action="store", help="Config file path")
    return parser.parse_args()


def config(args):
    databus["cfg_path"] = os.path.join(databus["base_path"], args.config)
    parser = Parser(databus)
    return parser.load()

def main():
    args = parse_args()
    if args.field:
        parser = Parser(databus)
        print(parser.show_fields())
        quit()
    if args.plugin:
        parser = Parser(databus)
        print(parser.show_plugins())
        quit()
    strategy = config(args)
    strategy()


if __name__ == '__main__':
    main()
