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
import pyopencl as cl
import pyopencl.array as cla

from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests)


def get_rerr(a, b):
    return np.max(np.abs(a/b - 1))


def test_vkfft_bindings(ctx_factory):
    ctx = ctx_factory()
    device = ctx.devices[0]
    platform = device.platform
    queue = cl.CommandQueue(ctx)

    from pycl_fft.vkfft import Configuration, Application, LaunchParams
    c = Configuration()

    # check that uninitialized CL objects return None
    assert c.platform is None
    assert c.device is None
    assert c.context is None

    # check that CL objects are wrapped correctly
    c.platform = platform
    assert c.platform == platform

    c.device = device
    assert c.device == device

    c.context = ctx
    assert c.context == ctx

    # check that a simple transform works
    shape = (64, 48, 27)
    assert c.size == []
    c.FFTdim = len(shape)
    c.size = shape[::-1]
    c.doublePrecision = True

    app = Application(c)
    pars = LaunchParams()
    pars.commandQueue = queue

    rng = np.random.default_rng()
    x = rng.random(shape) + 1j * rng.random(shape)
    ary = cla.to_device(queue, x)
    pars.buffer = ary.data

    app.append(-1, pars)
    queue.finish()

    y = np.fft.fftn(x)

    err = get_rerr(y, ary.get())
    assert err < 1e-10, err


def test_clfft_bindings(ctx_factory):
    import pycl_fft.clfft as clf

    ctx = ctx_factory()
    shape = (64, 48, 27)
    plan = clf.Plan(ctx, len(shape), shape)

    assert ctx == plan.context
    assert plan.dimension == len(shape)

    assert tuple(plan.lengths) == shape
    plan.lengths = (4, 4, 4)
    assert tuple(plan.lengths) == (4, 4, 4)
    plan.lengths = shape

    precision = clf.Precision.DOUBLE
    plan.precision = precision
    assert plan.precision == precision

    assert plan.batch_size == 1
    assert plan.forward_scale == 1

    # plan scales are only in float32 for some reason
    assert np.abs(np.product(shape) * plan.backward_scale - 1) < 1e-5
    plan.backward_scale = 1
    assert plan.backward_scale == 1

    strides = [1]
    for s in shape[:0:-1]:
        strides.append(strides[-1] * max(1, s))
    strides = tuple(strides[::-1])

    # clfft seems to work with column-major strides
    plan.input_strides = strides
    plan.output_strides = strides
    assert tuple(plan.input_strides) == strides
    assert tuple(plan.output_strides) == strides

    assert plan.input_distance == np.product(shape)
    assert plan.output_distance == np.product(shape)

    assert plan.input_layout == clf.Layout.COMPLEX_INTERLEAVED
    assert plan.output_layout == clf.Layout.COMPLEX_INTERLEAVED

    queue = cl.CommandQueue(ctx)
    plan.bake(queue)

    rng = np.random.default_rng()
    x = rng.random(shape) + 1j * rng.random(shape)
    ary = cla.to_device(queue, x)
    evts = plan.enqueue_transform(
        clf.Direction.FORWARD, [queue], None, [ary.data], None, None)
    evts[0].wait()

    y = np.fft.fftn(x)
    err = get_rerr(y, ary.get())
    assert err < 1e-10, err

    ary[:] = x
    plan(True, ary)

    err = get_rerr(y, ary.get())
    assert err < 1e-10, err

    plan(False, ary)
    err = get_rerr(x, ary.get() / x.size)
    assert err < 1e-10, err


if __name__ == "__main__":
    test_vkfft_bindings(cl.create_some_context)
    test_clfft_bindings(cl.create_some_context)
