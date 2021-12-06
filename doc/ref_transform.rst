
Fourier transforms
==================

.. automodule:: pycl_fft


Transform classes
-----------------

The above functions dispatch to Transform classes for each backend, which provide slightly
more control over the underlying plan/application.
The interfaces for both backends differ only where the underlying libraries' own
functionality differ.

.. note::

    :class:`Transform`\ s are cached using :func:`functools.lru_cache`:
    for each call pattern (i.e., set of arguments passed and their values)
    only one :class:`~pycl_fft.vkfft.Application` or
    :class:`~pycl_fft.clfft.Plan` and one :class:`Transform` is created.

.. autoclass:: pycl_fft.vkfft.Transform

.. autoclass:: pycl_fft.clfft.Transform
