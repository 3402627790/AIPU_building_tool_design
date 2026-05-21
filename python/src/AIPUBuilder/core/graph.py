#!/usr/bin/python
# -*- coding: UTF-8 -*-
# cython: language_level=3
from .._C._core import Graph as _Graph
from .utils import continue_class
from .tensor import *
import copy


@continue_class
class _Graph:
    def __enter__(self):
        global _DefaultGraph
        self.__old = _DefaultGraph
        _DefaultGraph = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global _DefaultGraph
        _DefaultGraph = self.__old

    def remove_betensor(self):
        for n in self.nodes:
            n.remove_betensor()

    @property
    def perf(self):
        """
        This property is a placeholder and should be replaced with the actual performance metric of the graph.

        Returns:
            dict: The performance metric of the graph.
        """

        class DictData(dict):
            def __init__(self, d):
                self.update(d)
                for k in self.keys():
                    if k not in dir(dict):
                        a = k.replace(" ", "_")
                        a = a.replace("@", "_")
                        a = a.replace("(", "_")
                        a = a.replace(")", "_")
                        a = a.replace("/", "_")
                        setattr(self, a, self[k])

        return DictData(self.attrs.get("_perf", dict()).get("profile_data", dict()))


class Graph(_Graph):

    def __init__(self, name="unamed"):
        super().__init__(name)

    def to_png(self, filename="graph.png"):
        """
        Generates a PNG image of the graph.

        Args:
            filename (str): The name of the output file. Defaults to "graph.png".
        """
        _Graph.to_png(self, filename)
        try:
            from IPython.display import Image

            return Image(filename)
        except Exception:
            pass

    def to_svg(self, filename="graph.svg"):
        """
        Generates an SVG image of the graph.

        Args:
            filename (str): The name of the output file. Defaults to "graph.svg".
        """
        _Graph.to_svg(self, filename)
        try:
            from IPython.display import SVG

            return SVG(filename)
        except Exception:
            pass

    def parse_from(self, filename, check_ir=True):
        """
        Parses a graph from a given file.

        Args:
            filename (str): The name of the input file.
            check_ir (bool): Whether to perform an intermediate representation check. Defaults to True.

        """
        return super().parse_from(filename, check_ir)

    @classmethod
    def parse(cls, filename, weight_file=None):
        """
        Parse a graph from a given file.

        Args:
            filename (str): The name of the input file.
            weight_file (str, optional): The name of the weight file. Defaults to None.

        Returns:
            Graph: A new instance of Graph parsed from the given file.
        """
        self = cls()
        self.parse_from(filename, check_ir=False)
        if weight_file:
            self.load_weight(weight_file)
        return self

    def clone(self, new_name=None):
        """
        Clones the graph with a new name.

        Args:
            new_name (str, optional): The name of the cloned graph. Defaults to None.

        Returns:
            Graph: A new instance of Graph that is a clone of this one.
        """
        if new_name is None:
            new_name = self.name
        g = self.__class__(new_name)
        super().clone(g)
        # after super.clone(g), the g has a new topological_sort of self, so the order of node may be changed.
        node_name_id_dict = {}
        for id, nn in enumerate(self.nodes):
            node_name_id_dict.update({nn.name: id})
        for i, n in enumerate(g.nodes):
            id = node_name_id_dict[n.name]
            for j, o in enumerate(n.outputs):
                o.__dict__.update(copy.deepcopy(self.nodes[id].outputs[j].__dict__))
            for k, c in n.constants.items():
                if c.name in self.tensors:
                    c.__dict__.update(
                        copy.deepcopy(self.nodes[id].constants[k].__dict__)
                    )
            for j, p in enumerate(n.placeholders):
                p.__dict__.update(
                    copy.deepcopy(self.nodes[id].placeholders[j].__dict__)
                )
            # copy python side attributes
            n.attrs.update(self.nodes[id].attrs.clone())
        for k, t in g.global_tensors.items():
            t.__dict__.update(copy.deepcopy(self.global_tensors[k].__dict__))
        return g

    def __deepcopy__(self, memo):
        from tempfile import NamedTemporaryFile

        with NamedTemporaryFile("wt") as txt:
            with NamedTemporaryFile("wb") as bin:
                print(txt.name, bin.name)
                s, b = self.serialize()
                txt.write(s)
                bin.write(b)
                print(txt.name, bin.name)
                txt.close()
                bin.close()
                print(s)
                g = Graph.parse(txt.name, bin.name)

    def forward(self, feed_data, disable_pbar=True):
        return self.forward_to(None, feed_data, disable_pbar)

    def forward_to(self, dest_node, feed_data, disable_pbar=True):
        data = feed_data
        if len(self.input_tensors) == 1 and not isinstance(feed_data, list):
            data = [
                feed_data,
            ]
        for inp, d in zip(self.input_tensors, data):
            inp.betensor = d

        import sys
        from AIPUBuilder.logger import tqdm

        with tqdm(
            total=len(self.nodes),
            desc="forward_to",
            file=sys.stdout,
            leave=True,
            disable=disable_pbar,
        ) as pbar:
            for n in self.nodes:
                n.forward()
                if dest_node is not None and n == dest_node:
                    break
                pbar.update(1)
            pbar.refresh()

        ret = []
        if dest_node is None:
            for out in self.output_tensors:
                # ret.append(out.betensor)
                ret.append(out)  # better to carry quantization info like scale
        else:
            for out in dest_node.outputs:
                ret.append(out)

        return ret

    @property
    def serialize_scale_zp(self):
        return self.attrs.get("serialize_scale_zp", False)

    @serialize_scale_zp.setter
    def serialize_scale_zp(self, b: bool):
        self.attrs["serialize_scale_zp"] = b

    @property
    def dump_debug_IR(self):
        return self.attrs.get("dump_debug_IR", False)

    @dump_debug_IR.setter
    def dump_debug_IR(self, b: bool):
        self.attrs["dump_debug_IR"] = b


_DefaultGraph = Graph("_DefaultGraph")  # the default graph must keep alive


def new_default_graph():
    global _DefaultGraph
    _DefaultGraph = Graph("_DefaultGraph")


def get_default_graph():
    return _DefaultGraph
