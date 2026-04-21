from typing import Tuple, Callable

import torch

### Constants
FP4_E2M1_MAX = 6
FP8_E4M3_MAX = 448
NVFP_GROUPSIZE = 16
MXFP_GROUPSIZE = 32
FP32_EXPONENT_BIAS = 127
FP32_MIN_NORMAL = 2 ** (-FP32_EXPONENT_BIAS + 1)

FP4_GRID =  [-6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
FP4_BITPACKING_PERM = [15, 14, 13, 12, 11, 10,  9,  8,  0,  1,  2,  3,  4,  5,  6,  7]
FP4_SCALE = 3 / 4

def cast_to_fp4(x):
    sign = torch.sign(x)
    x = torch.abs(x)
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign

def quantize_fp4(x: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, q_min: int, q_max: int):
    return cast_to_fp4(x / scales)

def dequantize_fp4(q: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor):
    return q.mul(scales)

def quantize_dequantize_fp4(x: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, q_min: int, q_max: int):
    xq = dequantize_fp4(quantize_fp4(x, scales, zeros, q_min, q_max), scales, zeros)
    return x + (xq - x).detach()


#### MXFP quantization
def cast_to_eBm0(x: torch.Tensor, ebits: int, emax: int):
    """
    Args:
        x: input tensor
        ebits: number of exponent bits
        emax: maximum exponent value for element data format
    """
    assert ebits % 2 == 0, "EBm0 expects even number of bits"
    assert x.ge(0).all(), "EBm0 expects positive inputs"
    qmin = -(2 ** (ebits - 1) - 1)
    qmax = +(2 ** (ebits - 1) - 1)
    # We clamp values instead of overflow (see https://github.com/microsoft/microxcaling/blob/7bc41952de394f5cc5e782baf132e7c7542eb4e4/mx/mx_ops.py#L83)
    return 2 ** (x.clamp(min=FP32_MIN_NORMAL).log2().floor().clamp(qmin, qmax) - emax)


def cast_to_eBm0_improved(x: torch.Tensor, ebits: int, emax: int):
    """
    Args:
        x: input tensor
        ebits: number of exponent bits
        emax: maximum exponent value for element data format
    """
    assert ebits % 2 == 0, "EBm0 expects even number of bits"
    assert x.ge(0).all(), "EBm0 expects positive inputs"
    smin = x.min().clamp(min=FP32_MIN_NORMAL)
    smax = x.max().clamp(min=FP32_MIN_NORMAL)
    # We clamp values instead of overflow (see https://github.com/microsoft/microxcaling/blob/7bc41952de394f5cc5e782baf132e7c7542eb4e4/mx/mx_ops.py#L83)
    return 2 ** ((smax.log2() - smin.log2()) * ((255 * (x.clamp(min=FP32_MIN_NORMAL).log2() - smin.log2()) / (smax.log2() - smin.log2())).floor().clamp(0, 255)) / 255 + smin.log2())


#### NVFP quantization
def cast_to_e4m3(x: torch.tensor):
    x = torch.clamp(x, min=2e-3, max=448.0)
    
    exponent = torch.floor(torch.log2(x + 1e-9))
    mantissa_val = x / (2**exponent) - 1.0 
    
    quantized_mantissa_val = torch.round(mantissa_val * 8) / 8
    
    reconstructed_val = (1 + quantized_mantissa_val) * (2**exponent)
    return reconstructed_val

