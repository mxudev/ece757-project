import sram
ss = sram.SramState(16, 0.5)

ans = ss.operate_conv((1,3,3), (1,2,2), (1,1,2,2), 1)

print([x for x in zip(["sram_ld_kernel", "sram_ld_feat", "sram_st_feat", "dram_ld_kernel", "dram_ld_feat", "dram_st_feat"], ans)])
