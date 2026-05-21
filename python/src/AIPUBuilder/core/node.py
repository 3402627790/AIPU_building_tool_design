#!/usr/bin/python
# -*- coding: UTF-8 -*-
# cython: language_level=3
from AIPUBuilder.core.utils import continue_class
from AIPUBuilder.utils import Deprecated
from AIPUBuilder.core.tensor import *
from AIPUBuilder._C._core import Graph, Node, NodeParamDict, AttrDict, NodeList, TensorList, NodeParamValue, TensorMap, Tensor, TensorShape, SubGraphNode
__all__ = [
    "Node",
    "NodeParamDict",
    "AttrDict",
    "NodeParamValue",
    "SubGraphNode",
    "NodeList",
]

try:
    # try to hack pydevd
    from _pydevd_bundle import pydevd_xml, pydevd_resolver
    pydevd_xml._TYPE_RESOLVE_HANDLER._initialize()
    default_type_map = pydevd_xml._TYPE_RESOLVE_HANDLER._default_type_map
    default_type_map.append(
        (NodeParamDict, pydevd_resolver.dictResolver))
    default_type_map.append(
        (TensorMap, pydevd_resolver.dictResolver))
    default_type_map.append(
        (AttrDict, pydevd_resolver.dictResolver))
    default_type_map.append(
        (NodeList, pydevd_resolver.tupleResolver))
    default_type_map.append(
        (TensorList, pydevd_resolver.tupleResolver))
    default_type_map.append(
        (TensorShape, pydevd_resolver.tupleResolver))
except Exception as e:
    pass

__node_list_getitem__ = NodeList.__getitem__


@continue_class
class NodeList:

    def ___getitem__(self, i):
        n = __node_list_getitem__(self, i)
        if SubGraphNode.is_subgraph_node(n):
            sgn = SubGraphNode.from_node(n)
            if sgn is not None:
                return sgn
        return n

    def index(self, one_node):
        """
        Get the index of a node in this NodeList.

        Args:
            i: The value to find.

        Returns:
           int: The index of the finding node.
        """
        return list(self).index(one_node)


@continue_class
class TensorMap:
    def keys(self):
        """
        Returns:
            List: a list of the keys in the TensorMap.
        """
        return list(self)

    def values(self):
        """
        Returns:
            List: a list of the values in the TensorMap.
        """
        return [self[k] for k in self.keys()]

    def pop(self, k):
        """
        Removes the specified item from the TensorMap and returns it.

        Args:
            k (str): the key of the item to remove.

        Returns:
           Any: the value of the removed item.
        """
        ret = self[k]
        del self[k]
        return ret


@continue_class
class NodeParamDict:
    def keys(self):
        """
        Returns:
            List: a list of the keys in the NodeParamDict.
        """
        return list(self)

    def values(self):
        """
        Returns:
            List: a list of the values in the NodeParamDict.
        """
        return [self[k] for k in self.keys()]

    def pop(self, k):
        """
        Pops the specified item from the NodeParamDict and returns it.

        Args:
           k (str): the key of the item to remove.

        Returns:
          Any: the value of the removed item.
        """
        ret = self[k]
        del self[k]
        return ret

    def items(self):
        """
        Returns:
           Iterator: an iterator over the items in the NodeParamDict.
        """
        for k in self:
            v = None
            flag = False
            # filter nonexistent key-values
            try:
                v = self[k]
                flag = True
            except:
                flag = False
            if flag:
                yield k, v
            else:
                pass

    def get(self, key, default=None):
        """
        Gets the value of the specified key.

        Args:
            key (str): the key of the item to retrieve.

        Returns:
            Any: the value of the item with the specified key, or None if the key does not exist.
        """
        if key in self:
            return self[key]
        return default

    def clear(self):
        """
        Clears all items from the NodeParamDict.
        """
        for k in self.keys():
            self.pop(k)

    def copy(self):
        """
        Copies the NodeParamDict to a new instance.
        """
        ret = self.__class__()
        for k in self:
            ret[k] = self[k]
        return ret

    def setdefault(self, k, d=None):
        """
        Sets the value of the specified key to a default value if it does not exist.

        Args:
            k (str): the key of the item to set.
            d (Any): the default value to set.

        Returns:
            Any: the value of the item with the specified key.
        """
        if k in self:
            return self[k]
        self[k] = d
        return d

    def update(self, dict2):
        """
        Updates the NodeParamDict with items from another dictionary.

        Args:
           dict2 (NodeParamDict): the dictionary to update from.
        """
        for k, v in dict2.items():
            self[k] = v

    def popitem(self):
        """
        Pops an item from the NodeParamDict.

        Returns:
           Any: the value of the popped item.
        """
        return self.pop(self.keys()[-1])

    def clone(self):
        """
        Clones the NodeParamDict.

        Returns:
          NodeParamDict: the cloned dictionary.
        """
        import copy
        clone_attrs = self.__class__()

        def _clone_iterator(target):
            if isinstance(target, list):
                new_target = []
                for t in target:
                    new_target.append(_clone_iterator(t))
            elif isinstance(target, tuple):
                new_target = ()
                for i, t in enumerate(target):
                    rt = _clone_iterator(t)
                    new_target = new_target + (rt,)
            elif isinstance(target, dict):
                new_target = {}
                for k, t in target.items():
                    new_target.update({k: _clone_iterator(t)})
            elif isinstance(target, (Node, Tensor)):
                new_target = target.clone(target.name + '_clone_')
            elif isinstance(target, TensorShape):
                new_target = target.clone()
            else:
                new_target = copy.deepcopy(target)
            return new_target

        for key, iterator in self.items():
            try:
                clone_attrs[key] = _clone_iterator(iterator)
            except:
                # Any can't clone in python side
                pass
        return clone_attrs


@continue_class
class AttrDict:
    def keys(self):
        """
        Returns a list of keys in the AttrDict.

        Returns:
           list: A list of keys.
        """
        return list(self)

    def values(self):
        """
        Returns a list of values in the AttrDict.

        Returns:
          list: A list of values.
        """
        return [self[k] for k in self.keys()]

    def pop(self, k):
        """
        Removes and returns the value of the specified key.

        Args:
            k: The key of the value to be removed.

        Returns:
            Any: The value of the specified key.
        """
        ret = self[k]
        del self[k]
        return ret

    def items(self):
        """
        Returns an iterator over the AttrDict's key-value pairs.

        Returns:
          iterator: An iterator over the AttrDict's key-value pairs.
        """
        for k in self:
            v = None
            flag = False
            # filter nonexistent key-values
            try:
                v = self[k]
                flag = True
            except:
                flag = False
            if flag:
                yield k, v
            else:
                pass

    def __str__(self) -> str:
        return "{%s}" % (",".join(["%s:%s" % (k, str(v)) for k, v in self.items()]))

    def get(self, key, default=None):
        """
        Gets the value of the specified key.

        Args:
           key(str): The key of the value to be retrieved.
           default: The default value to be returned if the key does not exist.

        Returns:
          value: The value of the specified key, or the default value if the key does not exist.

        """
        if key in self:
            return self[key]
        return default

    def clear(self):
        """
        Clears all key-value pairs from the dictionary.
        """
        for k in self.keys():
            self.pop(k)

    def copy(self):
        """
        Copies the dictionary to a new dictionary.

        Returns:
          new_dict(AttrDict): A new dictionary that is a copy of the original dictionary.
        """
        ret = self.__class__()
        for k in self:
            ret[k] = self[k]
        return ret

    def setdefault(self, k, d=None):
        """
        Sets the value of a key to a default value if it does not exist.

        Args:
            k(str): The key of the value to be set.
            d: The default value to be set.

        Returns:
            value: The value of the specified key, or the default value if the key does not exist.
        """
        if k in self:
            return self[k]
        self[k] = d
        return d

    def update(self, dict2):
        """
        Updates the dictionary with key-value pairs from another dictionary.

        Args:
           dict2(AttrDict): The dictionary to update the original dictionary with.
        """
        for k, v in dict2.items():
            self[k] = v

    def popitem(self):
        """
        Pops and returns an arbitrary key-value pair from the dictionary.

        Returns:
           key-value pair: The popped key-value pair.
        """
        return self.pop(self.keys()[-1])

    def clone(self):
        """
        Clones the dictionary.

        Returns:
          AttrDict: The cloned dictionary.
        """
        import copy
        clone_attrs = self.__class__()

        def _clone_iterator(target):
            if isinstance(target, list):
                new_target = []
                for t in target:
                    new_target.append(_clone_iterator(t))
            elif isinstance(target, tuple):
                new_target = ()
                for i, t in enumerate(target):
                    rt = _clone_iterator(t)
                    new_target = new_target + (rt,)
            elif isinstance(target, dict):
                new_target = {}
                for k, t in target.items():
                    new_target.update({k: _clone_iterator(t)})
            elif isinstance(target, (Node, Tensor)):
                new_target = target.clone(target.name + '_clone_')
            elif isinstance(target, TensorShape):
                new_target = target.clone()
            else:
                new_target = copy.deepcopy(target)
            return new_target
        for key, iterator in self.items():
            try:
                clone_attrs[key] = _clone_iterator(iterator)
            except:
                # Any can't clone in python side
                pass
        return clone_attrs


__node_clone__ = Node.clone


@continue_class
class Node:
    def clone(self, name=None):
        """
        Creates a clone of the current Node object.

        Args:
            name (str): The name for the cloned node. If not provided, the original name will be used.

        Returns:
           Node: A new Node object that is a copy of the current one.
        """
        n = __node_clone__(self)
        if SubGraphNode.is_subgraph_node(n):
            sgn = SubGraphNode.from_node(n)
            if sgn is not None:
                return sgn
        return n

    def remove_betensor(self):
        """
        Removes the betensor from the current Node object.
        """
        import sys
        for o in self.inputs:
            o.remove_betensor()
        for o in self.outputs:
            o.remove_betensor()
        for p in self.placeholders:
            p.remove_betensor()
        for k, c in self.constants.items():
            c.remove_betensor()

    def forward(self, *args):
        from AIPUBuilder.Optimizer.framework import OP_DICT
        from AIPUBuilder.Optimizer.utils.quant_tool_utils import linear_dequantize, linear_quantize_clip
        from AIPUBuilder.logger import WARN, ERROR, FATAL, DEBUG
        ret = None

        def _node_params_replace(dst_dict, src_dict, replace_keys=None):
            if replace_keys != None:
                for r_key in replace_keys:
                    if r_key in src_dict.keys():
                        dst_dict[r_key] = src_dict[r_key]
                    elif r_key in dst_dict.keys():
                        dst_dict.pop(r_key)
                    else:
                        WARN('the replace_keys:%s is not in src_dict:%s and node.params.' % (
                            r_key, src_dict.__name__))
                return None
            else:
                replace_keys = []
                for k, v in src_dict.items():
                    dst_dict[k] = src_dict[k]
                    replace_keys.append(k)
                return replace_keys

        if self.type not in OP_DICT:
            FATAL(
                'unsupported op "%s", please implement this op firstly' % str(self.type))

        # decide wether to change quantized flag for fake quantization debug usage
        quant_state = self.quantized
        constants_betensor_backup = None
        params_backup = None
        replace_keys = None
        # copy_constants_to(self.constants, constants_backup)
        if True == self.quantized:
            if 'debug_fake_quantize' in self.attrs and self.attrs['debug_fake_quantize']:
                for inp in self.inputs:
                    if 0 == inp.debug_flag:
                        inp.betensor = linear_dequantize(
                            inp.betensor, inp.scale, inp.zerop)
                        inp.debug_flag = 1
                self.quantized = False
                constants_betensor_backup = {}
                params_backup = self.params.clone()
                replace_keys = _node_params_replace(
                    self.params, self.attrs['params_original'])
                for k, t in self.attrs['constants_betensor_original'].items():
                    constants_betensor_backup[k] = self.constants[k].betensor.clone(
                    )
                    self.constants[k].betensor = t.clone()
            else:
                for inp in self.inputs:
                    if 0 != inp.debug_flag:
                        inp.betensor = linear_quantize_clip(
                            inp.betensor, inp.scale, inp.zerop, inp.qmin, inp.qmax)
                        inp.debug_flag = 0
        # call OP's forward(), backup its inputs first in case that inp.betensor be modified in forward function
        maintained_inp_betensors = []
        for ii, iinp in enumerate(self.inputs):
            maintained_inp_betensors.append(iinp.betensor.clone())
        ret = OP_DICT[self.type](self, *args)
        for ii, iinp in enumerate(self.inputs):
            iinp.betensor = maintained_inp_betensors[ii]
        # restore quantized flag and constants
        self.quantized = quant_state
        if None != constants_betensor_backup:
            for k, t in constants_betensor_backup.items():
                self.constants[k].betensor = t.clone()
        if None != params_backup:
            _ = _node_params_replace(self.params, params_backup, replace_keys)
        if True == self.quantized:
            if 'debug_fake_quantize' in self.attrs and self.attrs['debug_fake_quantize']:
                for out in self.outputs:
                    out.debug_flag = 1
            else:
                for out in self.outputs:
                    out.debug_flag = 0

        for t in self.outputs:
            dshape = t.betensor.shape
            sshape = dshape
            flag = False
            if len(list(sshape)) != len(list(dshape)):
                flag = True
            else:
                batch_size_in_IR = self.attrs.get("batch_size_in_IR", 1)
                st = 1 if batch_size_in_IR > 0 else 0
                for i in range(st, len(list(sshape))):
                    if sshape[i] != dshape[i]:
                        flag = True
                        break
            if flag:
                WARN('Get inconformity tensor shape %s with the original IR %s, layer_id=%s, tensor_name=%s' % (
                    str(dshape), str(sshape), self.attrs.get('layer_id', "-1"), self.name), op_name=str(self.type))

        return ret

    def __getitem__(self, key):
        return self.attrs[key]

    def __setitem__(self, key, value):
        self.attrs[key] = value


__subgraph_node_init__ = SubGraphNode.__init__


@continue_class
class SubGraphNode:

    def __init__(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], list):
            __subgraph_node_init__(self, NodeList(args[0]))
        else:
            __subgraph_node_init__(self, *args, **kwargs)

    def get_attr_deep(self, key, default=None):
        """
        Get attribute value deeply from the subgraph.

        Args:
           key (str): The attribute key to retrieve.
           default (any): The default value to return if the attribute is not found. Defaults to None.

        Returns:
            any: The attribute value or the default value.
        """
        if key in self.attrs:
            return self.attrs[key]
        for n in self.subgraph.nodes:
            if SubGraphNode.is_subgraph_node(n):
                return n.get_attr_deep(key, default)
            elif key in n.attrs:
                return n.attrs[key]
        return default
