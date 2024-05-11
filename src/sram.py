import math
import numpy as np

SA_H = 256
SA_W = 256
PRECISION = 2 #fp16 or 2 bytes per number
BUSWIDTH = 64 #bytes

def conv_sim(featuresize2d_x: int, featuresize2d_y: int, kernelsize2d_x: int, kernelsize2d_y:int, stride: int):
    res = np.zeros((featuresize2d_x, featuresize2d_y))
    for i in range(kernelsize2d_x, featuresize2d_x+1, stride):
        for j in range (kernelsize2d_y, featuresize2d_y+1, stride):
            res[i-kernelsize2d_x:i, j-kernelsize2d_y:j] += 1
    return np.sort(res.flatten()) #ascending

class SramState:
    def __init__(self, size, alpha):
        self.size = size #in bytes
        self.alpha = alpha
        self.beta = 1 - alpha
        self.alpha_size = size * alpha
        self.beta_size = size * self.beta
        self.f_preload_size = 0 #in bytes, loaded total features

    
    def operate_conv(self, in_shape, out_shape, conv_shape, conv_stride):
        #in/out shape: C x H x W
        #conv_shape: C_out x C_in x Kernel_x x Kernel_y
        #conv_stride: just the stride number
        #-------Kernel Accesss-------
        #DRAM ld
        K = conv_shape[0] #number of kernel vectors TODO make sure this is right for multiple channels
        F = out_shape[1] * out_shape[2] #number of feature vectors
        k_size = conv_shape[1] * conv_shape[2] * conv_shape[3] * PRECISION #kernel vector size in bytes
        dram_ld_kernel = K * math.ceil(k_size / BUSWIDTH) + (math.ceil(F/SA_H)-1) * math.ceil((K * k_size - self.alpha_size) / BUSWIDTH) #TODO added ceiling

        #SRAM ld / st
        sram_ld_kernel = math.ceil(F/SA_H) * K
        sram_st_from_dram_kernel = dram_ld_kernel #TODO: propose: just keep one term and count all terms when calc the actual energy

        #-------feature Accesss-------
        ld_seq = conv_sim(in_shape[1], in_shape[2], conv_shape[2], conv_shape[3], conv_stride)
        #SRAM ld
        sram_ld_feat = np.sum(ld_seq)*math.ceil(in_shape[0]*PRECISION/BUSWIDTH)*math.ceil(K/SA_W) #SRAM loads a all channels: in_shape[0]*PRECISION/BUSWIDTH

        #SRAM st
        sram_st_feat = out_shape[1]*out_shape[2] * math.ceil(out_shape[0]*PRECISION/BUSWIDTH)

        #DRAM ld
        #filling sram part
        num_chn_slot_filled = math.floor((self.beta_size - self.f_preload_size)/(in_shape[0]*PRECISION))
        dram_ld_feat = math.ceil(in_shape[0]*PRECISION/BUSWIDTH) * num_chn_slot_filled #fill as much unfilled sram as possible
        self.f_preload_size += num_chn_slot_filled * (in_shape[0]*PRECISION)

        assert(self.f_preload_size <= self.beta_size)

        
        num_chn_slot_rem = in_shape[1]*in_shape[2] - self.f_preload_size//(in_shape[0]*PRECISION) #should be whole number. no ceil needed
        print(self.f_preload_size//(in_shape[0]*PRECISION))

        dram_ld_feat += np.sum(ld_seq[:num_chn_slot_rem]) * math.ceil(in_shape[0]*PRECISION/BUSWIDTH) * math.ceil(K/SA_W)


        #DRAM ST
        #update content for next iter
        self.f_preload_size = math.floor((self.beta_size - self.f_preload_size)/(out_shape[0]*PRECISION))*(out_shape[0]*PRECISION)
        dram_st_feat = (out_shape[1]*out_shape[2] - self.f_preload_size/(out_shape[0]*PRECISION)) * math.ceil(out_shape[0]*PRECISION/BUSWIDTH)
        
        return sram_ld_kernel, sram_ld_feat, sram_st_feat, dram_ld_kernel, dram_ld_feat, dram_st_feat
        

