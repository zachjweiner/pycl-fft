__copyright__ = "Copyright (C) 2021 Zachary J Weiner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import numpy as np


r2c_dtype_map = {
    np.dtype("float32"): np.dtype("complex64"),
    np.dtype("float64"): np.dtype("complex128")
}

c2r_dtype_map = {
    np.dtype("complex64"): np.dtype("float32"),
    np.dtype("complex128"): np.dtype("float64")
}


def get_r2c_output_shape(shape, in_place=False):
    shape = list(shape)
    if in_place:
        # r array is padded by 2 in the last dimension
        shape[-1] -= 2
    # FIXME: odd shapes
    shape[-1] = shape[-1] // 2 + 1
    return tuple(shape)


def get_c2r_output_shape(shape, in_place=False):
    shape = list(shape)
    # FIXME: odd shapes
    shape[-1] = 2 * (shape[-1] - 1)
    if in_place:
        shape[-1] += 2
    return tuple(shape)


def get_c_strides(shape):
    strides = [1]
    for s in shape[:0:-1]:
        strides.append(strides[-1] * max(1, s))
    return tuple(strides[::-1])


def is_in_place(x, y):
    if x is None or y is None:
        return False
    return (x.base_data == y.base_data) and (x.offset == y.offset)


__all__ = [
    "r2c_dtype_map",
    "c2r_dtype_map",
    "get_r2c_output_shape",
    "get_c2r_output_shape",
    "get_c_strides",
    "is_in_place",
]
