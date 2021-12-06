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


import atexit
from functools import lru_cache
import numpy as np
import pyopencl as cl
import pyopencl.array as cla
from pycl_fft._clfft import (
    SetupData,
    Plan as _Plan,
    Status,
    Dimension,
    Layout,
    Precision,
    Direction,
    ResultLocation,
    ResultTransposition,
    CallbackType,
    teardown as _teardown,
    __version__,
)
from pycl_fft.util import get_c2r_output_shape, get_r2c_output_shape, get_c_strides

import logging
logger = logging.getLogger(__name__)

__doc__ = """
Plans
-----

.. class:: Plan(context: pyopencl.Context, dimension: int, lengths: tuple)

    Except where noted, all attrbutes can be set after creation.
    The plan may be :meth:`bake`\\ ed before calls to
    :meth:`enqueue_transform`; however, changing any plan parameters
    will "unbake" the transform.
    :class:`Plan`\\ s are destroyed (i.e., via :func:`clfftDestroyPlan`)
    automatically when garbage collected.

    .. attribute:: context

        Read-only (can only be set at initialization).

    .. attribute:: precision

        A :class:`Precision`.

    .. attribute:: forward_scale

        A :class:`float` by which to scale (multiply) the forward transform.

    .. attribute:: backward_scale

        A :class:`float` by which to scale (multiply) the backward transform.

    .. attribute:: batch_size

        An :class:`int`.

    .. attribute:: dimension

        A :class:`Dimension` (or :class:`int`) specifying the dimensionality of the
        transform.

    .. attribute:: lengths

        An iterable specifying the length of the transform along each axis.

    .. attribute:: input_strides

        An iterable specifying the strides of the input array(s) in units of the
        datatype.

    .. attribute:: output_strides

        An iterable specifying the strides of the output array(s) in units of the
        datatype.

    .. attribute:: input_distance

        An :class:`int` specifying the distance between inputs for batched
        transforms in units of the datatype.

    .. attribute:: output_distance

        An :class:`int` specifying the distance between outputs for batched
        transforms in units of the datatype.

    .. attribute:: input_layout

        A :class:`Layout`.

    .. attribute:: output_layout

        A :class:`Layout`.

    .. attribute:: placeness

        A :class:`ResultLocation`.

    .. attribute:: transposed

        A :class:`ResultTransposition`.

    .. attribute:: temp_buffer_size

        Read only.

    .. method:: bake(queues: list[pyopencl.CommandQueue])

    .. method:: enqueue_transform(dir: Direction, \
            queues: list[pyopencl.CommandQueue], \
            wait_for: list[pyopencl.Event], \
            inputs: list[pyopencl.Buffer], \
            outputs: list[pyopencl.Buffer], \
            temp_buffer: pyopencl.Buffer)

        :returns: A :class:`list` of :class:`pyopencl.Events`, one per
            passed :class:`pyopencl.CommandQueue`.

    .. automethod:: __call__

.. class:: SetupData

    Wrapper to :class:`clfftSetupData`; one is automatically created at module
    import (meaning the required initialization of the :mod:`clFFT` API
    is performed automatically).

.. function:: teardown

    Wrapper to :func:`clfftTeardown`.
    Registered as an exit hook via :mod:`atexit` upon module import; shouldn't have
    to be called by the user.

Constants
---------

.. class:: Status

    :class:`enum` of possible status codes returned by CLFFT.
    Extends the standard set of OpenCL status codes (refer to
    :class:`pyopencl.status_code`) to include the following (see the :mod:`clFFT`
    documentation for further explanation).

    .. attribute:: BUGCHECK
    .. attribute:: NOTIMPLEMENTED
    .. attribute:: TRANSPOSED_NOTIMPLEMENTED
    .. attribute:: FILE_NOT_FOUND
    .. attribute:: FILE_CREATE_FAILURE
    .. attribute:: VERSION_MISMATCH
    .. attribute:: INVALID_PLAN
    .. attribute:: DEVICE_NO_DOUBLE
    .. attribute:: DEVICE_MISMATCH

.. class:: Dimension

    .. attribute:: 1D
    .. attribute:: 2D
    .. attribute:: 3D

.. class:: Layout

    .. attribute:: COMPLEX_INTERLEAVED
    .. attribute:: COMPLEX_PLANAR
    .. attribute:: HERMITIAN_INTERLEAVED
    .. attribute:: HERMITIAN_PLANAR
    .. attribute:: REAL

.. class:: Precision

    .. attribute:: SINGLE
    .. attribute:: DOUBLE
    .. attribute:: SINGLE_FAST
    .. attribute:: DOUBLE_FAST

.. class:: Direction

    .. attribute:: FORWARD
    .. attribute:: BACKWARD
    .. attribute:: MINUS
    .. attribute:: PLUS

.. class:: ResultLocation

    .. attribute:: INPLACE
    .. attribute:: OUTOFPLACE

.. class:: ResultTransposition

    .. attribute:: NOTRANSPOSE
    .. attribute:: TRANSPOSED

.. class:: CallbackType

    .. attribute:: PRECALLBACK
    .. attribute:: POSTCALLBACK

"""


setup_data = SetupData()
torn_down = False


@atexit.register
def teardown():
    _teardown()
    global torn_down
    torn_down = True


class Plan(_Plan):
    def __init__(self, context: cl.Context, dimension: int, shape: tuple):
        try:
            super().__init__(context, dimension, shape)
        except RuntimeError as e:
            self.initialized = False
            raise RuntimeError(Status(int(e.args[0])).name)

        self.initialized = True

    def __call__(self, forward, input, output=None, temp=None):
        """
        Convenience wrapper to :meth:`enqueue_transform` for use with
        :class:`pyopencl.array.Array`\\ s.

        :arg input: A :class:`pyopencl.array.Array` or iterable thereof.

        :arg output: A :class:`pyopencl.array.Array` or iterable thereof.
            Defaults to *None* (for in-place transforms).

        :arg temp: A :class:`pyopencl.array.Array` for temporary/scratch usage
            by :mod:`clFFT`.
            Defaults to *None* (in which case :mod:`clFFT` allocates when needed).

        The ``queues`` are determined by those attached to each ``input``, and
        all of their :attr:`~pyopencl.array.Array.events` are passed
        to ``wait_for``.

        :returns: A :class:`list` of :class:`pyopencl.Events`, one per
            passed :class:`pyopencl.CommandQueue`.
        """

        if isinstance(input, cla.Array):
            input = [input]

        if isinstance(output, cla.Array):
            output = [output]

        inputs = [ary.base_data for ary in input]
        outputs = None if output is None else [ary.base_data for ary in output]

        wait_for = []
        for ary in input:
            wait_for.extend(ary.events)

        queues = [ary.queue for ary in input]
        temp_buffer = None if temp is None else temp.data
        direction = Direction.FORWARD if forward else Direction.BACKWARD

        events = self.enqueue_transform(
            direction, queues, wait_for, inputs, outputs, temp_buffer)

        for ary, evt in zip(input, events):
            ary.add_event(evt)

        if outputs is not None:
            for ary, evt in zip(output, events):
                ary.add_event(evt)

    def __del__(self):
        # apparently atexit-registered functions can be called before all plans
        # are garbage collected, so don't destroy the plan if clfftTearDown has
        # been called
        global torn_down
        if self.initialized and not torn_down:
            self.destroy()


@lru_cache
class Transform:
    """
    :arg ctx: A :class:`pyopencl.Context`.

    :arg shape: The shape of the input to the forward transform.
        The dimension of the transform is inferred from the length of this argument.

    :arg dtype: The datatype of the input to the forward transform.

    :arg type: Which type of transform to implement.
        Valid options are ``"c2c"``, ``"r2c"``, or ``"c2r"`` (:mod:`clFFT` does not
        support discrete cosine transforms).
        Note that out-of-place real-to-complex and complex-to-real transforms
        require separate :class:`~pycl_fft.clfft.Plan`\\ s because the
        number of buffers required and their strides differ.

    :arg in_place: Whether to overwrite output in the supplied input array.
        Defaults to *False*.
        Note that out-of-place transforms require an additional array to hold the
        output (and that out-of-place complex-to-real transforms require another
        temporary array on top of that).

    :arg norm: Whether to normalize the inverse transform.
        Defaults to *False*.

    Any remaining keyword arguments are set as attributes to the
    :class:`~pycl_fft.clfft.Plan`.
    Note that this allows one to overwrite any values previously set by
    :meth:`Transform`\\ 's initialization, which could lead to invalid
    configurations or unexpected results.

    .. automethod:: __call__

    :meth:`forward` and :meth:`backward` are convenience wrappers to
    :meth:`__call__` to perform the specified transform.

    .. automethod:: forward
    .. automethod:: backward
    """

    def __init__(self, ctx: cl.Context, shape: tuple, dtype, type="c2c",
                 in_place: bool = False, norm: str = None, **kwargs):
        self.plan = Plan(ctx, len(shape), shape[::-1])

        self.plan.batch_size = 1

        self.in_place = in_place
        if in_place:
            self.plan.placeness = ResultLocation.INPLACE
        else:
            self.plan.placeness = ResultLocation.OUTOFPLACE

        self.separate_buffer_required = (type == "c2r") and (not in_place)

        cshape = get_r2c_output_shape(shape)  # shape arg excludes padding
        rshape = get_c2r_output_shape(cshape, in_place)  # accounts for padding
        if type == "c2c":
            self.plan.input_strides = get_c_strides(shape)[::-1]
            self.plan.output_strides = get_c_strides(shape)[::-1]
        elif type == "r2c":
            self.plan.input_strides = get_c_strides(rshape)[::-1]
            self.plan.output_strides = get_c_strides(cshape)[::-1]
            self.plan.input_strides = self.plan.input_strides
        elif type == "c2r":
            self.plan.input_strides = get_c_strides(cshape)[::-1]
            self.plan.output_strides = get_c_strides(rshape)[::-1]

        dtype = np.dtype(dtype)
        if dtype in (np.float64, np.complex128):
            self.plan.precision = Precision.DOUBLE
        elif dtype in (np.float32, np.complex64):
            self.plan.precision = Precision.SINGLE
        else:
            raise NotImplementedError(f"Transforms for {dtype} are unsupported.")

        if type == "c2c":
            self.plan.input_layout = Layout.COMPLEX_INTERLEAVED
            self.plan.output_layout = Layout.COMPLEX_INTERLEAVED
        elif type == "r2c":
            self.plan.input_layout = Layout.REAL
            self.plan.output_layout = Layout.HERMITIAN_INTERLEAVED
        elif type == "c2r":
            self.plan.input_layout = Layout.HERMITIAN_INTERLEAVED
            self.plan.output_layout = Layout.REAL
        else:
            raise ValueError(f"Transforms of type {type} are unsupported.")

        if norm == "forward":
            self.plan.forward_scale = 1 / np.product(shape)
            self.plan.backward_scale = 1
        elif norm == "backward":
            self.plan.forward_scale = 1
            self.plan.backward_scale = 1 / np.product(shape)
        elif norm is None:
            self.plan.forward_scale = 1
            self.plan.backward_scale = 1

        for key, value in kwargs.items():
            setattr(self.plan, key, value)

    def __call__(self, forward: bool, input: cla.Array, output: cla.Array = None,
                 temp: cla.Array = None):
        """
        :arg forward: Whether to do a forward (*True*) or backward (*False*)
            transform.

        :arg input: The input array for the transform.

        :arg output: The output array for the transform.
            Required if ``in_place == False``.

        :arg temp: A scratch/temporary array.
            Required (and only used) for out-of-place c2r transforms.

        :returns: The :class:`~pyopencl.array.Array` holding the output for
            the given transform.

        .. note::

            If a temporary buffer is required by :mod:`clFFT`, it will allocate
            and manage one itself if it is not supplied via ``temp``.
        """

        if not self.in_place:
            if output is None:
                raise TypeError(
                    "Transform.__call__() missing argument output that is "
                    "required for out-of-place transforms.")

            # if self.separate_buffer_required and not forward:
                # if temp is None:
                #         raise TypeError(
                #             "Transform.__call__() missing argument buffer that is"
                #             "required for out-of-place c2r transforms.")

        if input.offset != 0 or output.offset != 0:
            raise ValueError("clFFT does not support offsets.")

        self.plan(forward, input, output, temp)

        return input if self.in_place else output  # FIXME: no return?

    def forward(self, input: cla.Array, output: cla.Array = None,
                temp: cla.Array = None):
        return self(True, input, output, temp=temp)

    def backward(self, input: cla.Array, output: cla.Array = None,
                 temp: cla.Array = None):
        return self(False, input, output, temp=temp)


__all__ = [
    "SetupData",
    "Plan",
    "Status",
    "Dimension",
    "Layout",
    "Precision",
    "Direction",
    "ResultLocation",
    "ResultTransposition",
    "CallbackType",
    "teardown",
    "setup_data",
    "__version__",
    "Transform",
]
