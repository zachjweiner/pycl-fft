
Low-level wrapper to VkFFT
==========================

.. currentmodule:: pycl_fft

The :mod:`pycl_fft.vkfft` module provides a (nearly) one-to-one wrapper of the |vkfft|_ API.
Consult the |vkfft|_ documentation for details; the below simply documents the relationship between the Python wrapper and the |vkfft|_ API.

The names of classes and methods omit "VkFFT" and conform to PEP-8.
For ease of wrapping (and cross-referencing with the VkFFT documentation), all attribute names are unchanged (and so often do not conform to PEP-8).
Conversions from Python types are trivial (i.e., :class:`int`\ s to integers, or
iterables thereof mapping to C arrays), except for :mod:`pyopencl` objects, for which
the conversion is performed behind the scenes via the underlying pointers.
So, for example, the following just works::

    config = pycl_fft.vkfft.Configuration()
    context = pyopencl.Context(...)
    config.context = context

Retrieving ``config.context`` then returns the same :class:`pyopencl.Context`.
If a particular OpenCL object has not been previously set for ``config``,
accessing it will return *None*.

Here's a complete example of a 3-D, double-precision, in-place, complex-to-complex transform::

    import numpy as np
    import pyopencl as cl
    import pyopencl.array as cla
    from pycl_fft.vkfft import Configuration, Application, LaunchParams

    config = Configuration()

    context = cl.create_some_context()
    config.context = context
    config.device = config.context.devices[0]
    config.platform = config.device.platform

    shape = (64, 64, 64)
    config.FFTdim = len(shape)
    config.size = shape[::-1]
    config.doublePrecision = True

    app = Application(config)
    pars = LaunchParams()

    queue = cl.CommandQueue(context)
    pars.commandQueue = queue

    buf_h = np.random.rand(*shape) + 1j * np.random.rand(*shape)
    buf = cla.to_device(queue, buf_h)
    pars.buffer = buf.data

    app.append(-1, pars)
    queue.finish()


.. automodule:: pycl_fft.vkfft
