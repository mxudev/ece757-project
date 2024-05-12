import os

cfg_str_pre = "-size (bytes) "
MAX_SIZE = 4194304
START_SIZE = 2048
CFG_FILE = "ddr_cache.cfg"
os.system("rm -rf " + CFG_FILE + ".out")
for s in range(START_SIZE, MAX_SIZE+1, 4096):
    cfg_str = cfg_str_pre + str(s)
    os.system("sed -i \"1s/.*/" + cfg_str + "/\" " + CFG_FILE)
    os.system("./cacti -infile " + CFG_FILE)


