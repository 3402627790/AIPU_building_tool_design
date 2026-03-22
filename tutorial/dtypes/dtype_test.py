import sys
import os
CUR_DIR=os.path.dirname(os.path.abspath(__file__))
ROOT_DIR=os.path.join(CUR_DIR,"../../")
sys.path.insert(0,os.path.join(ROOT_DIR,"python/src"))
# from AIPUBuilder.plugin_loader import *
from AIPUBuilder.core import *
from AIPUBuilder._C._dtypes import get_dtypes
def main():
    # skip plugin infos for first call
    get_dtypes(OpType.Convolution)

    print("\n==========Tutorial for OP Data Type API==========\n")
    optype = OpType.Pooling
    print("Valid in/out Dtypes for " + str(optype))

    op_dtype_list = get_dtypes(optype)
    # for op_dtype in op_dtype_list:
    for i in range(len(op_dtype_list)):
        oss = "\tvalid_pair[" + str(i) + "]\t in: ["
        op_dtype = op_dtype_list[i]
        in_dtypes = op_dtype.in_dtypes
        for dt in in_dtypes:
            oss += str(dt) + ","
        oss += "]\t, out: ["
        out_dtypes = op_dtype.out_dtypes
        for dt in out_dtypes:
            oss += str(dt) + ","
        oss += "]"
        print(oss)

    print("")

if __name__ == "__main__":
    main()