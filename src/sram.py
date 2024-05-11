import math

SA_H = 256
SA_W = 256
PERCISON = 2 #fp16 or 2 bytes per number
BUSWIDTH = 64 #bytes
class SramState:
    def __init__(self, size, alpha):
        self.size = size
        self.alpha = alpha
        self.beta = 1 - alpha
        self.alpha_size = size * alpha
        self.beta_size = size * self.beta
        self.f_preload_size = 0

    
    def operate_conv(self, in_shape, out_shape, conv_shape, conv_stride):
        #in/out shape: C x H x W
        #conv_shape: C_out x C_in x Kernel_x x Kernel_y
        #conv_stride: just the stride number
        #-------Kernel Accesss-------
        #DRAM ld
        K = conv_shape[0] #number of kernel vectors
        F = out_shape[1] * out_shape[2] #number of feature vectors
        k_size = conv_shape[1] * conv_shape[2] * conv_shape[3] * PERCISON #kernel vector size in bytes
        dram_ld_kernel = math.ceil(F/SA_H) * (K * k_size - self.alpha_size) / BUSWIDTH
        #SRAM ld / st
        sram_ld_kernel = math.ceil(F/SA_H) * K
        sram_st_from_dram_kernel = dram_ld_kernel

        #-------Festure Accesss-------
        #DRAM ld
        #TODO


        

