# cython: language_level=3
#!/usr/bin/python
# -*- coding: UTF-8 -*-

from AIPUBuilder.core import Graph

GLOBAL_PREGSIM_PASS = []
GLOBAL_POSTGSIM_PASS = []


def register_pre_pass(p):
    global GLOBAL_PREGSIM_PASS
    if isinstance(p.order, list):
        for i in p.order:
            class WrappedPass(BasePassPlugin):
                order = i
                name = p.name

                def __call__(self, g: Graph) -> bool:
                    return p(g)
            GLOBAL_PREGSIM_PASS.append(WrappedPass())
    else:
        GLOBAL_PREGSIM_PASS.append(p())


def register_post_pass(p):
    global GLOBAL_POSTGSIM_PASS
    print(p, p.order)
    if isinstance(p.order, list):
        for i in p.order:
            class WrappedPass(BasePassPlugin):
                order = i
                name = p.name

                def __call__(self, g: Graph) -> bool:
                    return p(g)
            GLOBAL_POSTGSIM_PASS.append(WrappedPass())
    else:
        GLOBAL_POSTGSIM_PASS.append(p())


def clear_pre_pass():
    global GLOBAL_PREGSIM_PASS
    GLOBAL_PREGSIM_PASS.clear()


def clear_post_pass():
    global GLOBAL_POSTGSIM_PASS
    GLOBAL_POSTGSIM_PASS.clear()


def get_pre_pass(is_sort=True, is_print=True):
    if is_sort:
        global GLOBAL_PREGSIM_PASS
    GLOBAL_PREGSIM_PASS = sorted(GLOBAL_PREGSIM_PASS, key=lambda x: x.order)
    if is_print and len(GLOBAL_PREGSIM_PASS):
        print("Pre GSIM pass list:\nid\tname")
        for p in GLOBAL_PREGSIM_PASS:
            print(f"{p.order}\t {p.name}")
    return [i for i in GLOBAL_PREGSIM_PASS]


def get_post_pass(is_sort=True, is_print=True):
    if is_sort:
        global GLOBAL_POSTGSIM_PASS
    GLOBAL_POSTGSIM_PASS = sorted(GLOBAL_POSTGSIM_PASS, key=lambda x: x.order)
    if is_print and len(GLOBAL_POSTGSIM_PASS):
        print("Post GSIM pass list:\nid\tname")
        for p in GLOBAL_POSTGSIM_PASS:
            print(f"{p.order}\t {p.name}")
    return [i for i in GLOBAL_POSTGSIM_PASS]


class BasePassPlugin(object):

    order = 100
    name = None

    def __init__(self):
        if self.name is None:
            self.name = self.__class__.__name__

    def __call__(self, g: Graph) -> bool:
        raise NotImplementedError("This method should be overridden by subclasses")


def register_pass_function(reg_func, f, od=100):
    class WrappedPass(BasePassPlugin):
        order = od
        name = f.__name__

        def __call__(self, g: Graph) -> bool:
            return f(g)
    reg_func(WrappedPass())
