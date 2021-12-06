
Low-level wrapper to clFFT
==========================

.. currentmodule:: pycl_fft

The :mod:`pycl_fft.clfft` module provides a (nearly) one-to-one wrapper of the :mod:`clFFT` API.
The names of classes and methods omit "clFFT" and conform to PEP-8; the correspondence between :class:`~pycl_fft.clfft.Plan` parameters and :func:`clfftSetPlan*` and :func:`clfftGetPlan*`
methods should be apparent (but is documented below).
Simply set (and get) parameters as normal Python class attributes.
(:mod:`clFFT` methods that set two parameters at once, e.g., input and output layouts, have been wrapped accordingly.)
Consult the :mod:`clFFT` documentation for further details.

Currently, plan copying (:func:`clfftCopyPlan`) and callbacks (:func:`clfftSetPlanCallback`)
are unimplemented.

Here's a complete example of a 3-D, double-precision, in-place, complex-to-complex transform::

    import numpy as np
    import pyopencl as cl
    import pyopencl.array as cla
    import pycl_fft.clfft as clf

    context = cl.create_some_context()
    shape = (64, 48, 27)
    plan = clf.Plan(context, len(shape), shape)

    plan.precision = clf.Precision.DOUBLE

    x = np.random.rand(*shape) + 1j * np.random.rand(*shape)

    # clfft seems to work with column-major strides, so set them manually
    strides = np.array(x.strides) // x.itemsize
    plan.input_strides = strides
    plan.output_strides = strides

    queue = cl.CommandQueue(context)
    plan.bake([queue])

    ary = cla.to_device(queue, x)
    plan(True, ary)


.. automodule:: pycl_fft.clfft
