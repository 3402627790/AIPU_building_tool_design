
import sys
import struct
tmplt="""
const char m_d[]={%s};
"""
with open(sys.argv[1],"rb") as f:
    d=f.read()
with open(sys.argv[2],"w") as f:
    f.write(tmplt%(",".join(["0x"+i.hex() for i in struct.unpack("c"*(len(d)),d) ])))