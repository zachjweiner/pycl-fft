
Low-level wrapper to clFFT
==========================

.. currentmodule:: pycl_fft

The :mod:`pycl_fft.clfft` module provides a (nearly) one-to-one wrapper of the |clfft|_ API.
The names of classes and methods omit "clFFT" and conform to PEP-8; the correspondence between :class:`~pycl_fft.clfft.Plan` parameters and :func:`clfftSetPlan*` and :func:`clfftGetPlan*`
methods should be apparent (but is documented below).
Simply set (and get) parameters as normal Python class attributes.
(|clfft|_ methods that set two parameters at once, e.g., input and output layouts, have been wrapped accordingly.)
Consult the |clfft|_ documentation for further details.

Currently, plan copying (:func:`clfftCopyPlan`) and callbacks (:func:`clfftSetPlanCallback`)
are unimplemented.

Here's a complete example of a 3-D, double-precision, in-place, complex-to-complex transform::

    import numpy as np
    import pyopencl as cl
    import pyopencl.array as cla
    import pycl_fft.clfft as clf

    context = cl.create_some_context()
    shape = (64, 48, 27)
    plan = clf.Plan(context, len(shape), shape[::-1])
    plan.precision = clf.Precision.DOUBLE

    rng = np.random.default_rng()
    buf_h = rng.random(shape) + 1j * rng.random(shape)

    strides = np.array(buf_h.strides) // buf_h.itemsize
    plan.input_strides = strides[::-1]
    plan.output_strides = strides[::-1]

    queue = cl.CommandQueue(context)
    plan.bake(queue)

    buf = cla.to_device(queue, buf_h)
    plan(True, buf)

    assert np.max(np.abs(buf.get() / np.fft.fftn(buf_h) - 1)) < 1e-12

As apparent in the above example, |clfft|_ specifies plan sizes and array strides in the reverse order of (C-style, row-major) array shapes.
(This is an artifact of using pointers to C arrays for inputs that can vary in length and maintaining that the first entry denote the contiguous array axis for transforms of any dimension.)
Outside of the low-level wrapper, :mod:`pycl_fft` adheres to :mod:`numpy`-like semantics, and :class:`~pycl_fft.clfft.Transform` handles this translation automatically.

.. automodule:: pycl_fft.clfft
