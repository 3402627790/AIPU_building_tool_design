import os
import inspect


class Data:

    def __init__(self, key, value, onwer=None):
        self.key = key
        self.value = value
        self.writing = False
        self.onwer = onwer

    def claim(self, instance):
        if not self.writing:
            self.writing = True
            self.owner = instance
            return True
        else:
            assert False, f"Data {self.key} is occupied by {self.owner}, accessing failed"
        return False

    def release(self):
        self.writing = False


class Databus:

    data = {
        "base_path": Data("base_path", os.path.abspath("./"))
    }
    opt_config = ''

    def __len__(self):
        return len(Databus.data)

    def __getitem__(self, key):
        return Databus.data[key].value

    def __setitem__(self, key, value):
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        name = module.__name__
        if key not in Databus.data.keys():
            Databus.data[key] = Data(key, value, name)
        else:
            Databus.data[key].claim(name)
            Databus.data[key].value = value
            Databus.data[key].release()

    def claim(self, key, msg):
        Databus.data[key].claim(msg)

    def release(self, key):
        Databus.data[key].release()

    def __repr__(self, with_owner=False):
        s = ""
        for k, v in Databus.data.items():
            s += k
            s += ":"
            s += str(v.value)
            if with_owner:
                s += f"\t{v.onwer}"
            s += ",\n"
        return s

    def __contains__(self, k):
        return k in Databus.data.keys()

    def __getattr__(self, key):
        return Databus.data[key].value

    def __setattr__(self, key, value):
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        name = module.__name__
        if key not in Databus.data.keys():
            Databus.data[key] = Data(key, value, name)
        else:
            Databus.data[key].claim(name)
            Databus.data[key].value = value
            Databus.data[key].release()

    def get_dict(self):
        return Databus.data
