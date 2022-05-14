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


from functools import lru_cache
import numpy as np
import pyopencl as cl
import pyopencl.array as cla
from pycl_fft.util import get_r2c_output_shape
from pycl_fft._vkfft import (
    Configuration as _Configuration,
    LaunchParams as _LaunchParams,
    Result,
    Application as _Application)

import logging
logger = logging.getLogger(__name__)

__doc__ = """
.. class:: Configuration()

    Create one with an empty initialization::

        config = Configuration()

    and then set all attributes manually.

.. autoclass:: Application(configuration: Configuration)

    :meth:`__init__` wraps :func:`initializeVkFFT`.
    Propagates and raises any exceptions from unsuccessful
    initializations.
    Its :meth:`__del__` method calls :func:`deleteVkFFT` (invoked automatically by
    Python's garbage collector).

    .. automethod:: append

.. autoclass:: LaunchParams

.. class:: Result()
"""


class ApplicationInitializationError(RuntimeError):
    pass


class PyOpenCLReferenceHandlingMixIn:
    """
    Currently, Python can/will garbage collect any :mod:`pyopencl` even if any of
    the VkFFT objects retain references to them.
    This means, for example, that the following code will crash::

        config.context = cl.create_some_context()
        config.context

    because Python will have garbage collected the context, meaning the pointer
    (via which the Python communicates with the wrapper) no longer holds a valid
    context.
    This mixin class ensures that the VkFFT wrapper objects retain references
    to PyOpenCL objects over the duration of their lifetime.
    """

    def __setattr__(self, key, val):
        # FIXME: this is a little slow
        if getattr(val, "__module__", None) in ("pyopencl._cl", "pyopencl.array"):
            super().__setattr__(f"_py_{key}", val)
        super().__setattr__(key, val)


class Configuration(_Configuration, PyOpenCLReferenceHandlingMixIn):
    pass


class LaunchParams(_LaunchParams, PyOpenCLReferenceHandlingMixIn):
    """
    Like :class:`~pycl_fft.vkfft.Configuration`, create one with an empty
    initialization::

        pars = LaunchParams()

    and then set all attributes manually.
    """
    pass


class Application(_Application, PyOpenCLReferenceHandlingMixIn):
    def __init__(self, configuration: Configuration):
        try:
            super().__init__(configuration)
        except RuntimeError as e:
            self.initialized = False
            raise ApplicationInitializationError(Result(int(e.args[0])).name)

        self.initialized = True

    def __del__(self):
        if self.initialized:
            # only call deleteVkFFT if application creation was successful;
            # otherwise, VkFFT has already done so and calling deleteVkFFT again
            # leads to bus errors
            self.delete()

    def append(self, inverse: int, launchParams: LaunchParams):
        """
        Wraps :func:`VkFFTAppend`, handling error codes and raising
        :class:`RuntimeError`\\ s as need.
        """

        res = Result(super().append(inverse, launchParams))
        if res != Result.SUCCESS:
            raise RuntimeError(res.name)


@lru_cache
class Transform:
    """
    :arg ctx: A :class:`pyopencl.Context`.

    :arg shape: The shape of the input to the forward transform.
        The dimension of the transform is inferred from the length of this argument.

    :arg dtype: The datatype of the input to the forward transform.

    :arg type: Which type of transform to implement.
        Valid options are ``"c2c"``, ``"r2c"``, ``"c2r"``, and integers 1-4 (for
        discrete cosine transforms).
        Note that out-of-place real-to-complex and complex-to-real transforms
        require separate :class:`~pycl_fft.vkfft.Application`\\ s because the
        number of buffers required and their strides differ.

    :arg in_place: Whether to overwrite output in the supplied input array.
        Defaults to *False*.
        Note that out-of-place transforms require an additional array to hold the
        output (and that out-of-place complex-to-real transforms require another
        temporary array on top of that).

    :arg axes: Axes over which to perform the transform.
        Defaults to *None*, in which case transforms are performed over all axes of
        the specified ``shape``.

    :arg nbatch: The number of batches for batched transforms.
        Defaults to ``1``.

    :arg norm: Whether to normalize the inverse transform.
        Defaults to *False*.

    Any remaining keyword arguments are set as attributes to the
    :class:`~pycl_fft.vkfft.Configuration`.
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
                 in_place: bool = False, axes: tuple = None, nbatch: int = 1,
                 norm: bool = False, **kwargs):

        # scenarios
        # 1. destroy input, subsequent transforms can modify output
        #   -> buffer = input, return buffer (1 array)
        # 2. preserve input, subsequent transforms can modify output
        #   -> buffer = output, input = input, return output (2 arrays)
        # 3. destroy input, subsequent transforms cannot modify output
        #   -> buffer = input, output = output, return output (2)
        #   OR set inverseReturnToInputBuffer and pass
        #      buffer = buffer, input = input, return input (2)
        #   useful if you have a spare buffer array anyway
        # 4. preserve input, subsequent transforms cannot modify output
        #   -> buffer = buffer, input = input, output = output, return output (3)

        logger.info(f"Initializing Transform with {type=}, {shape=}, {in_place=}.")

        self.config = Configuration()
        self.config.FFTdim = len(shape)
        self.config.size = shape[::-1]
        self.config.numberBatches = nbatch
        if axes is not None:
            if type in ("r2c", "c2r") and len(shape) - 1 not in axes:
                raise ValueError(
                    "VkFFT does not support omitting last axis of "
                    f"{type} transforms.")
            omit_dims = [int(i not in axes) for i in range(len(shape))][::-1]
            self.config.omitDimension = omit_dims

        self.config.specifyOffsetsAtLaunch = True

        self.in_place = in_place
        self.separate_buffer_required = (type == "c2r") and (not in_place)

        if not in_place:
            self.config.isInputFormatted = True
            cshape = get_r2c_output_shape(shape)[::-1]
            if type == "c2c":
                self.config.inputBufferStride = np.cumprod(self.config.size)
                self.config.bufferStride = np.cumprod(self.config.size)
            elif type == "r2c":
                self.makeForwardPlanOnly = True
                self.config.inputBufferStride = np.cumprod(self.config.size)
                self.config.bufferStride = np.cumprod(cshape)
            elif type == "c2r":
                self.makeInversePlanOnly = True
                self.config.inputBufferStride = np.cumprod(cshape)
                self.config.bufferStride = np.cumprod(cshape)
                self.config.isOutputFormatted = True
                self.config.outputBufferStride = np.cumprod(self.config.size)

        dtype = np.dtype(dtype)
        if dtype in (np.float64, np.complex128):
            self.config.doublePrecision = True
        elif dtype not in (np.float32, np.complex64):
            raise NotImplementedError(f"Transforms for {dtype} are unsupported.")

        if type in ("r2c", "c2r"):
            self.config.performR2C = True
        elif type in range(1, 5):
            self.config.performDCT = type
        elif type != "c2c":
            raise ValueError(f"Transforms of type {type} are unsupported.")

        self.config.normalize = norm

        self.config.context = ctx
        self.config.device = ctx.devices[0]

        for key, value in kwargs.items():
            setattr(self.config, key, value)

        self.app = Application(self.config)

    def __call__(self, forward, input: cla.Array, output: cla.Array = None,
                 temp: cla.Array = None, _temp: cla.Array = None,
                 kernel: cla.Array = None):
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

            Calls to :class:`pycl_fft.vkfft.Transform`\\ s perform no memory
            allocations, even when needed (for or out-of-place mode), instead raising
            :class:`ValueError`\\ s.
            The user must supply these arrays as needed.

        .. note::

            Until |vkfft|_ implements event handling, this method enforces
            synchronization by calling
            :meth:`pyopencl.CommandQueue.finish`\\ for the ``input`` array's
            queue before and after invoking the transform.
        """

        # FIXME: optimize pars creation
        pars = LaunchParams()
        pars.commandQueue = input.queue

        if self.in_place:
            pars.buffer = input.base_data
            pars.bufferOffset = input.offset
        else:
            pars.inputBuffer = input.base_data
            pars.inputBufferOffset = input.offset

            if output is None:
                raise TypeError(
                    "Transform.__call__() missing argument output that is "
                    "required for out-of-place transforms.")

            if self.separate_buffer_required and not forward:
                pars.outputBuffer = output.base_data
                pars.outputBufferOffset = output.offset
                if temp is not None:
                    pars.buffer = temp.base_data
                    pars.bufferOffset = temp.offset
                else:
                    raise TypeError(
                        "Transform.__call__() missing argument temp that is"
                        "required for out-of-place c2r transforms.")
            else:
                pars.buffer = output.base_data
                pars.bufferOffset = output.offset

        if _temp is not None:
            pars.tempBuffer = _temp.data
            pars.tempBufferOffset = _temp.offset
        if kernel is not None:
            pars.kernel = kernel.data
            pars.kernelOffset = kernel.offset

        direction = -1 if forward else 1
        input.finish()  # FIXME: pass wait_for to VkFFT
        self.app.append(direction, pars)
        input.queue.finish()  # FIXME: events in VkFFT

        return input if self.in_place else output  # FIXME: no return?

    def forward(self, input: cla.Array, output: cla.Array = None,
                temp: cla.Array = None, _temp: cla.Array = None,
                kernel: cla.Array = None):
        return self(True, input, output, temp=temp, _temp=_temp, kernel=kernel)

    def backward(self, input: cla.Array, output: cla.Array = None,
                 temp: cla.Array = None, _temp: cla.Array = None,
                 kernel: cla.Array = None):
        return self(False, input, output, temp=temp, _temp=_temp, kernel=kernel)


__all__ = [
    "Configuration",
    "Application",
    "LaunchParams",
    "Result",
    "Transform",
]
