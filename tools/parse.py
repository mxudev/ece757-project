import csv

INFILE = "ddr_cache.cfg.out"

sram_read = []
sram_write = []
sram_size = []

with open(INFILE, "r") as f:
    csv_file = csv.reader(f)
    for i,c in enumerate(csv_file):
        if i == 0:
            continue
        sram_size.append(int(c[1]))
        sram_read.append(float(c[8]))
        sram_write.append(float(c[9]))

print("sram_read =", sram_read)
print("sram_write =", sram_write)
print("sram_size =", sram_size)
