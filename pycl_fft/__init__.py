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


import pyopencl.array as cla
from pycl_fft.util import (
    r2c_dtype_map, c2r_dtype_map, get_r2c_output_shape, get_c2r_output_shape,
    is_in_place)
import pycl_fft.vkfft  # noqa
try:
    import pycl_fft.clfft  # noqa
except ModuleNotFoundError:
    pass


__doc__ = """
High-level interface
--------------------

The following functions provide :mod:`numpy.fft`/:mod:`scipy.fft`-like
functionality for forward and inverse transforms of
:class:`pyopencl.array.Array`\\ s.
Because instances of :class:`pycl_fft.vkfft.Transform` and
:class:`pycl_fft.clfft.Transform` are cached, using the high-level interface
comes with minimal performance penalty.
However, by default, they are minimally destructive at the expense of extra memory
allocations, and therefore performance.
For example, if you only pass an ``input`` array, an ``output`` array is allocated
and returned so that ``input`` is not implicitly overwritten.
(For :func:`irfftn`, a further allocation of a temporary buffer is also required
for out-of-place transforms.)
To avoid this expense, allocate and pass an ``output`` array yourself, or pass a
:class:`pyopencl.tools.MemoryPool` ``allocator``.

Furthermore, to opt in to in-place transforms, pass the same array as ``input`` and
``output``.
For :func:`rfftn` and :func:`irfftn`, the real array must be padded.
Namely, :func:`rfftn` assumes that the length of the last axis of the input (real)
array is two longer than that of the transform to be performed, and :func:`irfftn`
returns a real array with last axis two longer than strictly needed to hold the
actual output.
Consult the |vkfft|_ or |clfft|_ documentation for further details.

Batching is supported in a similar manner to :func:`scipy.fft.fftn` via the
``axes`` keyword argument.
The |vkfft|_ backend supports array dimensions up to four and batching along any
axis (excluding the first axis for four-dimensional arrays and also the last axis
for real-to-complex/complex-to-real transforms).
The |clfft|_ backend only supports batching along the first axis.

Both the |vkfft|_ and |clfft|_ backends are supported, which one may choose
between by passing ``backend="vkfft"`` (the default) or ``backend="clfft"``.

.. autofunction:: fftn
.. autofunction:: ifftn
.. autofunction:: rfftn
.. autofunction:: irfftn
.. autofunction:: dctn
.. autofunction:: idctn

.. autofunction:: set_backend
"""

default_backend = "vkfft"


def set_backend(backend):
    """
    Set the default backend for transforms to one of ``"vkfft"`` (the default)
    or ``"clfft"``.
    """

    global default_backend
    default_backend = backend


def clear_cache():
    import pycl_fft.vkfft as vkf
    vkf.Transform.cache_clear()  # pylint: disable=E1101
    try:
        import pycl_fft.clfft as clf
        clf.Transform.cache_clear()  # pylint: disable=E1101
    except ModuleNotFoundError:
        pass


def get_transform_class(backend):
    global default_backend
    backend = backend or default_backend
    if backend == "vkfft":
        return pycl_fft.vkfft.Transform
    elif backend == "clfft":
        return pycl_fft.clfft.Transform
    else:
        raise NotImplementedError(f"Transforms for backend {backend}.")


def _process_shape_and_axes(shape, axes):
    """
    :returns: shape, axes, nbatch
    """

    if axes is None:
        # if invalid,
        return shape, axes, 1
    elif len(axes) > 3:
        raise ValueError(
            "FFTs with dimension greater than three are unsupported.")
    elif 0 in axes:
        if len(shape) == 4:
            raise ValueError(
                "Can't transform along first dimension of 4D arrays.")
        return shape, axes, 1
    else:
        axes = tuple(ax - 1 for ax in axes)
        return shape[1:], axes, shape[0]


def fftn(input: cla.Array, output: cla.Array = None, temp: cla.Array = None,
         allocator=None, backend=None, axes: tuple = None):
    if output is None:
        output = cla.empty_like(input, allocator=allocator)

    shape, axes, nbatch = _process_shape_and_axes(input.shape, axes)
    Transform = get_transform_class(backend)
    transform = Transform(
        input.context, shape, input.dtype, in_place=is_in_place(input, output),
        axes=axes, nbatch=nbatch)

    return transform.forward(input=input, output=output, temp=temp)


def ifftn(input: cla.Array, output: cla.Array = None, temp: cla.Array = None,
          allocator=None, backend=None, axes: tuple = None):
    if output is None:
        output = cla.empty_like(input, allocator=allocator)

    shape, axes, nbatch = _process_shape_and_axes(input.shape, axes)
    Transform = get_transform_class(backend)
    transform = Transform(
        input.context, shape, input.dtype, in_place=is_in_place(input, output),
        axes=axes, nbatch=nbatch)

    return transform.backward(input=input, output=output, temp=temp)


def rfftn(input: cla.Array, output: cla.Array = None, temp: cla.Array = None,
          allocator=None, backend=None, axes: tuple = None):
    in_place = is_in_place(input, output)
    cdtype = r2c_dtype_map[input.dtype]

    shape, axes, nbatch = _process_shape_and_axes(input.shape, axes)
    cshape = get_r2c_output_shape(shape, in_place=in_place)
    if in_place:
        shape = shape[:-1] + (shape[-1] - 2,)

    if output is None:
        output = cla.empty(
            input.queue, (nbatch,)+cshape, cdtype, allocator=allocator)

    Transform = get_transform_class(backend)
    transform = Transform(
        input.context, shape, input.dtype, in_place=in_place, type="r2c",
        axes=axes, nbatch=nbatch)

    result = transform.forward(input=input, output=output, temp=temp)
    if in_place:
        return result.view(dtype=cdtype)
    else:
        return result


def irfftn(input: cla.Array, output: cla.Array = None, temp: cla.Array = None,
           allocator=None, backend=None, axes: tuple = None):
    in_place = is_in_place(input, output)
    rdtype = c2r_dtype_map[input.dtype]

    shape, axes, nbatch = _process_shape_and_axes(input.shape, axes)
    shape = get_c2r_output_shape(shape)

    if output is None:
        output = cla.empty(
            input.queue, (nbatch,)+shape, rdtype, allocator=allocator)
    if not in_place and temp is None:
        temp = cla.empty_like(input, allocator=allocator)

    Transform = get_transform_class(backend)
    transform = Transform(
        input.context, shape, rdtype, in_place=in_place, type="c2r",
        axes=axes, nbatch=nbatch)

    result = transform.backward(input=input, output=output, temp=temp)
    if in_place:
        return result.view(dtype=rdtype)
    else:
        return result


def dctn(input: cla.Array, output: cla.Array = None, type: int = 2,
         temp: cla.Array = None, allocator=None, backend=None, axes: tuple = None):
    if output is None:
        output = cla.empty_like(input, allocator=allocator)

    if backend is not None and backend != "vkfft":
        raise NotImplementedError("Only the vkfft backend supports dctn.")

    shape, axes, nbatch = _process_shape_and_axes(input.shape, axes)
    transform = pycl_fft.vkfft.Transform(
        input.context, shape, input.dtype, in_place=is_in_place(input, output),
        type=type, axes=axes, nbatch=nbatch)

    return transform.forward(input=input, output=output, temp=temp)


def idctn(input: cla.Array, output: cla.Array = None, type: int = 2,
          temp: cla.Array = None, allocator=None, backend=None, axes: tuple = None):
    if output is None:
        output = cla.empty_like(input, allocator=allocator)

    if backend is not None and backend != "vkfft":
        raise NotImplementedError("Only the vkfft backend supports dctn.")

    shape, axes, nbatch = _process_shape_and_axes(input.shape, axes)
    transform = pycl_fft.vkfft.Transform(
        input.context, shape, input.dtype, in_place=is_in_place(input, output),
        type=type, axes=axes, nbatch=nbatch)

    return transform.backward(input=input, output=output, temp=temp)


__all__ = [
    "fftn",
    "ifftn",
    "rfftn",
    "irfftn",
    "dctn",
    "idctn",
]
