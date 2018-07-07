import sys

i=0
ret = {}
fname = {}
nargs = {}
args_type = {}
args_name ={}

for line in sys.stdin:
    atama = line.split("(")
    if len(atama) == 1:
        continue
    atamas = atama[0].split()
    fname[i] = atamas[-1]
    if "extern" in atamas[0]:
        ke = 1
    else:
        ke = 0
    ret[i] = " ".join(atamas[ke:-1])

    hara = line.split("(")[1].split(")")
    haras = hara[0].split(",")
    nargs[i] = len(haras)
    harastype = {}
    harasname = {}
    for j in range(nargs[i]):
        one_arg_list = haras[j].split()
        if len(one_arg_list) == 1:
            harastype[j] = "void"
            harasname[j] = ""
        else:
            harastype[j] = " ".join(one_arg_list[:-1])
            harasname[j] = one_arg_list[-1]
    
    args_type[i] = harastype.copy()
    args_name[i] = harasname.copy()
    i+=1

for j in range(i):
    atama = "%s %s " % (ret[j], fname[j] )
    hara_type = ""
    hara_name = ""
    for k in range(nargs[j]):
        hara_type += args_type[j][k] + " "
        hara_name += args_name[j][k] + " "
#    print( argtype )
#    print( argname )
#    print( "#", hara_type )
#    print( "@", hara_name )
    print( fname[j], "(", hara_name, ")" )
    print( " ".join( [ ret[j], fname[j], "(", hara_type, ")" ] ) )

