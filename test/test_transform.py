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
import scipy.fft as sp  # pylint: disable=E0611
import pyopencl as cl
import pyopencl.array as cla
import pycl_fft as clf
import pytest

from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests)


def get_rerr(a, b):
    err = np.abs(a / b - 1)
    return np.max(err), np.average(err)


precisions = ["single", "double"]
types = ["c2c", "r2c", 1, 2, 3, 4]
_vkfft_types = [("vkfft", typ) for typ in types]
backends_types = [("clfft", typ) for typ in ("c2c", "r2c")] + _vkfft_types
shapes = [
    (128,),
    # (2176,),
    # (4096,),
    (64, 64),
    (384, 512),
    (64, 64, 64),
    (32, 48, 26),
]


@pytest.mark.parametrize("shape", shapes)
@pytest.mark.parametrize("precision", precisions)
@pytest.mark.parametrize("backend, type", backends_types)
def test_transforms(ctx_factory, shape, precision, type, backend):
    # the Transform cache hangs on to contexts, which pile up when running pytest
    # and exceed OpenCL limits on simultaneous contexts as well as RAM
    clf.clear_cache()  # pylint: disable=E1101
    clf.set_backend(backend)

    # pylint is unhappy with scipy.fft
    # pylint: disable=E1101

    ctx = ctx_factory()
    queue = cl.CommandQueue(ctx)

    nbits = {"single": 32, "double": 64}[precision]
    if type == "c2c":
        dtype = np.dtype(f"complex{2*nbits}")
        forward = clf.fftn
        backward = clf.ifftn
        scipy_forward = sp.fftn
        scipy_backward = sp.ifftn
        call_kwargs = {}
    else:
        dtype = np.dtype(f"float{nbits}")
        if type == "r2c":
            forward = clf.rfftn
            backward = clf.irfftn
            scipy_forward = sp.rfftn
            scipy_backward = sp.irfftn
            call_kwargs = {}
        else:
            forward = clf.dctn
            backward = clf.idctn
            scipy_forward = sp.dctn
            scipy_backward = sp.idctn
            call_kwargs = dict(type=type)

    print(f"{type=}, dtype={dtype.name}, {shape=}, {backend=}")

    from numpy.random import default_rng
    rng = default_rng()

    x_h = rng.random(shape).astype(dtype)
    if dtype.kind == "c":
        x_h += 1j * (rng.random(shape).astype(dtype))
    y_h = scipy_forward(x_h, **call_kwargs, norm="backward")

    x = cla.empty(queue, (3,)+shape, dtype)
    y = cla.empty(queue, (3,)+y_h.shape, y_h.dtype)

    # forward transforms

    max_rtol = 1e-10 if precision == "double" else 1e-2
    avg_rtol = 1e-12 if precision == "double" else 1e-4
    if isinstance(type, int):
        # DCT forward transforms have large peak error for some reason
        max_rtol *= 1e3

    # test automatic construction of required arrays
    x[0] = x_h
    out = forward(x[0], **call_kwargs)
    max_err, avg_err = get_rerr(out.get(), y_h)
    print(f"forward:\t{max_err=:.3e}\t{avg_err=:.3e}")
    assert avg_err < avg_rtol, avg_err
    assert max_err < max_rtol, max_err

    # test offsets: set y_h to CL result and test with lower tols
    y_h = out.get()
    max_rtol = 1e-14 if precision == "double" else 1e-6
    avg_rtol = 1e-15 if precision == "double" else 1e-7

    def offset_aligned(ary):
        align = ctx.devices[0].mem_base_addr_align // 8
        return ary.offset % align == 0

    def invalid_offset_check(x, y):
        return backend == "clfft" and not (offset_aligned(x) and offset_aligned(y))

    for ix in range(x.shape[0]):
        for iy in range(y.shape[0]):
            x[:] = 0
            y[:] = 0
            x[ix] = x_h

            if invalid_offset_check(x[ix], y[iy]):
                continue

            _ = forward(x[ix], y[iy], **call_kwargs)
            max_err, avg_err = get_rerr(y_h, y[iy].get())
            assert avg_err < avg_rtol, avg_err
            assert max_err < max_rtol, max_err

    # test in-place
    slc = [slice(None)]*len(shape)
    if type == "r2c":
        in_shape = list(shape)
        in_shape[-1] += 2
        in_shape = tuple(in_shape)
        slc[-1] = slice(-2)
    else:
        in_shape = shape[:]
    slc = tuple(slc)

    x_h_pad = np.zeros(in_shape, dtype)
    x_h_pad[slc] = x_h
    x_pad = cla.to_device(queue, x_h_pad)

    # transform returns proper view of x_pad
    x_pad = forward(x_pad, x_pad, **call_kwargs)
    max_err, avg_err = get_rerr(y_h, x_pad.get())
    assert avg_err < avg_rtol, avg_err
    assert max_err < max_rtol, max_err

    # backward transforms

    x_h = scipy_backward(y_h, **call_kwargs, norm="forward")

    max_rtol = 1e-10 if precision == "double" else 1e-2
    avg_rtol = 1e-12 if precision == "double" else 1e-4
    if type != "c2c":
        # single-precision inverse transforms have large peak error for some reason
        max_rtol *= 1e3
    if isinstance(type, int):
        avg_rtol *= 10

    # test automatic construction of required arrays
    y[0] = y_h
    out = backward(y[0], **call_kwargs)
    max_err, avg_err = get_rerr(out.get(), x_h)
    print(f"backward\t{max_err=:.3e}\t{avg_err=:.3e}\n")
    assert avg_err < avg_rtol, avg_err
    assert max_err < max_rtol, max_err

    # test offsets: set y_h to CL result and test with lower tols
    x_h = out.get()
    max_rtol = 1e-14 if precision == "double" else 1e-5
    avg_rtol = 1e-15 if precision == "double" else 1e-6

    for ix in range(x.shape[0]):
        for iy in range(y.shape[0]):
            x[:] = 0
            y[:] = 0
            y[iy] = y_h

            if invalid_offset_check(x[ix], y[iy]):
                continue

            _ = backward(y[iy], x[ix], **call_kwargs)
            max_err, avg_err = get_rerr(x_h, x[ix].get())
            assert max_err < max_rtol, max_err
            assert avg_err < avg_rtol, avg_err

    # test in-place
    x_pad[...] = y_h
    # transform returns proper view of x_pad
    x_pad = backward(x_pad, x_pad, **call_kwargs)
    max_err, avg_err = get_rerr(x_h, x_pad.get()[slc])
    assert avg_err < avg_rtol, avg_err
    assert max_err < max_rtol, max_err

    # import gc
    # print(gc.get_referrers(ctx))
    # import sys
    # print(sys.getrefcount(ctx))
    # clf.clear_cache()  # pylint: disable=E1101
    # gc.collect()
    # print(sys.getrefcount(ctx))


@pytest.mark.parametrize("backend", ["vkfft", "clfft"])
def test_caching(ctx_factory, backend):
    # pylint: disable=E1101
    ctx = ctx_factory()

    clf.clear_cache()  # from previous tests

    if backend == "vkfft":
        Transform = clf.vkfft.Transform
    elif backend == "clfft":
        Transform = clf.clfft.Transform

    def get_hits():
        return Transform.cache_info().hits

    def get_misses():
        return Transform.cache_info().misses

    _ = Transform(ctx, (4,), np.dtype("float64"))
    assert get_hits() == 0
    assert get_misses() == 1
    _ = Transform(ctx, (4,), np.dtype("float64"))
    assert get_misses() == 1
    assert get_hits() == 1

    _ = Transform(ctx, (8,), np.dtype("float64"))
    assert get_misses() == 2

    _ = Transform(ctx, (8,), np.dtype("float64"))
    assert get_misses() == 2
    assert get_hits() == 2

    _ = Transform(ctx, (8,), np.dtype("complex128"))
    assert get_misses() == 3
    assert get_hits() == 2

    _ = Transform(ctx, (8,), np.dtype("complex128"))
    assert get_misses() == 3
    assert get_hits() == 3

    _ = Transform(ctx, (8,), np.dtype("complex128"), norm=1)
    assert get_misses() == 4
    assert get_hits() == 3
    _ = Transform(ctx, (8,), np.dtype("complex128"), norm=1)
    assert get_misses() == 4
    assert get_hits() == 4

    clf.clear_cache()  # from previous tests
    assert get_misses() == 0
    assert get_hits() == 0


if __name__ == "__main__":
    context = cl.create_some_context()

    for backend, type in backends_types:
        for shape in shapes:
            for precision in precisions:
                test_transforms(lambda: context, shape, precision, type, backend)

    for backend in ["clfft", "vkfft"]:
        test_caching(lambda: context, backend)
