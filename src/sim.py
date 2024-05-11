import numpy as np

def conv_sim(featuresize2d: int, kernelsize2d: int, stride: int):
    res = np.zeros((featuresize2d, featuresize2d))
    cnt = 0
    for i in range(kernelsize2d, featuresize2d+1, stride):
        for j in range (kernelsize2d, featuresize2d+1, stride):
            res[i-kernelsize2d:i, j-kernelsize2d:j] += 1
            cnt += 1
    return res,cnt

def kernel_transaction_count():
    pass
