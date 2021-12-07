Welcome to pycl_fft's documentation!
======================================

:mod:`pycl_fft` provides thin, PyOpenCL-based bindings to `VkFFT <https://github.com/DTolm/VkFFT>`_ and `clFFT <https://github.com/clMathLibraries/clFFT>`_, libraries implementing GPU-accelerated
Fast Fourier Transforms with OpenCL.
It also provides convenient wrappers that expose :mod:`numpy.fft`- and :mod:`scipy.fft`-like interfaces.

:mod:`pycl_fft` was written with the following goals in mind:

* to provide transparent bindings to the entire |vkfft|_ and |clfft|_ APIs
* to provide simple, Pythonic interfaces to the "standard" set of transforms that both are flexible and incur minimal overhead for both backends
* and to enable straightforward extension of those interfaces as needed.

To this end, |vkfft|_'s and |clfft|_'s public APIs are exposed using :mod:`pybind11`.
Relatively slim wrappers classes handle the somewhat cumbersome configuration process, which then enable a set of :mod:`numpy.fft`/:mod:`scipy.fft`-like functions to compute transforms.

All basic transform types are supported for arbitrary dimensions (up to three, as supported by the underlying libraries) and sizes, for both in- and out-of-place mode.
The |vkfft|_ backend supports arbitrary buffer offsets (|clfft|_ itself does not support offsets).
Currently, no wrappers exist to perform convolutions with |vkfft|_, though one may use the low-level bindings to do so exactly as with |vkfft|_ directly.

To install, simply clone the source and run ``pip install .`` or ::

    pip install git+https://github.com/zachjweiner/pycl-fft.git

which will install :mod:`numpy` and :mod:`pyopencl` if needed.
The |vkfft|_ header is bundled and its wrappers will be built automatically.
The |clfft|_ shared library is first searched for inside a directory specified by the environment variable ``CLFFT_DIR``; if unset, the path to :mod:`conda`-installed libraries (``CONDA_PREFIX`` for the active environment) is checked.
If ``libclFFT.so`` is not found, the |clfft|_ bindings are not built and only the |vkfft|_ backend wil be available.
Therefore, the simplest way to enable |clfft|_ support is to simply install in a conda
environment::

    conda install -c conda-forge clfft
    pip install git+https://github.com/zachjweiner/pycl-fft.git

The tests require :mod:`pytest` and |scipy|_.


Table of Contents
-----------------

.. toctree::
   :maxdepth: 2

   ref_transform
   ref_vk_wrapper
   ref_cl_wrapper


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
