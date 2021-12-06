// Copyright (C) 2021 Zachary J Weiner
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#define VKFFT_BACKEND 3
#include <vkFFT.h>

#define DEF_SIMPLE_RW(name) \
  def_readwrite(#name, &cls::name)

#define DEF_ERROR_VALUE(name) \
  value(#name, VKFFT_ERROR_##name)

#define py_obj_to_cl_ptr(name, ClType) \
  [](cls *self, py::object obj) \
  { \
    ClType *clobj = new ClType; \
    *clobj = (ClType)(obj.attr("int_ptr").cast<intptr_t>()); \
    self->name = clobj; \
  }

#define cl_ptr_to_py_obj(name, pyType) \
  [](cls *self) -> py::object \
  { \
      if(self->name == 0) return py::object(py::cast(nullptr)); \
      intptr_t int_ptr = (intptr_t) *self->name; \
      py::object cl = py::module_::import("pyopencl"); \
      return cl.attr(#pyType).attr("from_int_ptr")(int_ptr); \
  }

#define DEF_CL_PROPERTY(name, ClType, pyType) \
  def_property(#name, cl_ptr_to_py_obj(name, pyType), py_obj_to_cl_ptr(name, ClType))

#define set_from_list(name, T) \
  [](cls *self, const std::vector<T> &input) \
  { \
    for(std::vector<T>::size_type i = 0; i != input.size(); i++) \
      self->name[i] = input[i]; \
  }

#define get_to_list(name, T) \
  [](cls *self) \
  { \
    std::vector<T> result; \
    for(int i = 0; i < (int)self->FFTdim; i++) \
      result.push_back(self->name[i]); \
    return result; \
  }

#define DEF_CARRAY_PROPERTY(name, type) \
  def_property(#name, get_to_list(name, type), set_from_list(name, type))

VkFFTApplication* init_application(const VkFFTConfiguration* config)
{
    VkFFTApplication* app = new VkFFTApplication({});
    const int err = initializeVkFFT(app, *config);

    // C/C++ can't convert enums to strings, so return the int and let Python do it
    if (err != VKFFT_SUCCESS) throw std::runtime_error(std::to_string((int)err));

    return app;
}

PYBIND11_MODULE(_vkfft, m)
{
    {
        typedef VkFFTConfiguration cls;
        py::class_<cls>(m, "Configuration")
            .def(py::init<>())
            .DEF_SIMPLE_RW(FFTdim)
            .DEF_CARRAY_PROPERTY(size, uint64_t)
            .DEF_CL_PROPERTY(platform, cl_platform_id, Platform)
            .DEF_CL_PROPERTY(device, cl_device_id, Device)
            .DEF_CL_PROPERTY(context, cl_context, Context)
            .DEF_SIMPLE_RW(userTempBuffer)
            .DEF_SIMPLE_RW(bufferSize)
            .DEF_SIMPLE_RW(tempBufferSize)
            .DEF_SIMPLE_RW(inputBufferSize)
            .DEF_SIMPLE_RW(outputBufferSize)
            .DEF_SIMPLE_RW(kernelSize)
            .DEF_CL_PROPERTY(buffer, cl_mem, Buffer)
            .DEF_CL_PROPERTY(tempBuffer, cl_mem, Buffer)
            .DEF_CL_PROPERTY(inputBuffer, cl_mem, Buffer)
            .DEF_CL_PROPERTY(outputBuffer, cl_mem, Buffer)
            .DEF_CL_PROPERTY(kernel, cl_mem, Buffer)
            .DEF_SIMPLE_RW(bufferOffset)
            .DEF_SIMPLE_RW(tempBufferOffset)
            .DEF_SIMPLE_RW(inputBufferOffset)
            .DEF_SIMPLE_RW(outputBufferOffset)
            .DEF_SIMPLE_RW(kernelOffset)
            .DEF_SIMPLE_RW(specifyOffsetsAtLaunch)
            .DEF_SIMPLE_RW(coalescedMemory)
            .DEF_SIMPLE_RW(aimThreads)
            .DEF_SIMPLE_RW(numSharedBanks)
            .DEF_SIMPLE_RW(inverseReturnToInputBuffer)
            .DEF_SIMPLE_RW(numberBatches)
            .DEF_SIMPLE_RW(useUint64)
            .DEF_CARRAY_PROPERTY(omitDimension, uint64_t)
            .DEF_SIMPLE_RW(fixMaxRadixBluestein)
            .DEF_SIMPLE_RW(performBandwidthBoost)
            .DEF_SIMPLE_RW(doublePrecision)
            .DEF_SIMPLE_RW(halfPrecision)
            .DEF_SIMPLE_RW(halfPrecisionMemoryOnly)
            .DEF_SIMPLE_RW(doublePrecisionFloatMemory)
            .DEF_SIMPLE_RW(performR2C)
            .DEF_SIMPLE_RW(performDCT)
            .DEF_SIMPLE_RW(disableMergeSequencesR2C)
            .DEF_SIMPLE_RW(normalize)
            .DEF_SIMPLE_RW(disableReorderFourStep)
            .DEF_SIMPLE_RW(useLUT)
            .DEF_SIMPLE_RW(makeForwardPlanOnly)
            .DEF_SIMPLE_RW(makeInversePlanOnly)
            .DEF_CARRAY_PROPERTY(bufferStride, uint64_t)
            .DEF_SIMPLE_RW(isInputFormatted)
            .DEF_SIMPLE_RW(isOutputFormatted)
            .DEF_CARRAY_PROPERTY(inputBufferStride, uint64_t)
            .DEF_CARRAY_PROPERTY(outputBufferStride, uint64_t)
            .DEF_SIMPLE_RW(considerAllAxesStrided)
            .DEF_SIMPLE_RW(keepShaderCode)
            .DEF_SIMPLE_RW(printMemoryLayout)
            .DEF_SIMPLE_RW(saveApplicationToString)
            .DEF_SIMPLE_RW(loadApplicationFromString)
            .DEF_SIMPLE_RW(loadApplicationString)
            .DEF_CARRAY_PROPERTY(performZeropadding, uint64_t)
            .DEF_CARRAY_PROPERTY(fft_zeropad_left, uint64_t)
            .DEF_CARRAY_PROPERTY(fft_zeropad_right, uint64_t)
            .DEF_SIMPLE_RW(frequencyZeroPadding)
            .DEF_SIMPLE_RW(performConvolution)
            .DEF_SIMPLE_RW(conjugateConvolution)
            .DEF_SIMPLE_RW(crossPowerSpectrumNormalization)
            .DEF_SIMPLE_RW(coordinateFeatures)
            .DEF_SIMPLE_RW(matrixConvolution)
            .DEF_SIMPLE_RW(symmetricKernel)
            .DEF_SIMPLE_RW(numberKernels)
            .DEF_SIMPLE_RW(kernelConvolution)
            .DEF_SIMPLE_RW(registerBoost)
            .DEF_SIMPLE_RW(registerBoostNonPow2)
            .DEF_SIMPLE_RW(registerBoost4Step)
            .DEF_SIMPLE_RW(swapTo3Stage4Step)
            .DEF_SIMPLE_RW(devicePageSize)
            .DEF_SIMPLE_RW(localPageSize)
            .DEF_CARRAY_PROPERTY(maxComputeWorkGroupCount, uint64_t)
            .DEF_CARRAY_PROPERTY(maxComputeWorkGroupSize, uint64_t)
            .DEF_SIMPLE_RW(maxThreadsNum)
            .DEF_SIMPLE_RW(sharedMemorySizeStatic)
            .DEF_SIMPLE_RW(sharedMemorySize)
            .DEF_SIMPLE_RW(sharedMemorySizePow2)
            .DEF_SIMPLE_RW(warpSize)
            .DEF_SIMPLE_RW(halfThreads)
            .DEF_SIMPLE_RW(allocateTempBuffer)
            .DEF_SIMPLE_RW(reorderFourStep)
            .DEF_SIMPLE_RW(maxCodeLength)
            .DEF_SIMPLE_RW(maxTempLength)
            .DEF_CL_PROPERTY(commandQueue, cl_command_queue, CommandQueue)
        ;
    }

    {
        typedef VkFFTApplication cls;
        py::class_<cls>(m, "Application")
            .def(py::init(&init_application))
            .def("append", VkFFTAppend)
            .def("delete", deleteVkFFT)
        ;
    }

    {
        typedef VkFFTLaunchParams cls;
        py::class_<cls>(m, "LaunchParams")
            .def(py::init<>())
            .DEF_CL_PROPERTY(commandQueue, cl_command_queue, CommandQueue)
            .DEF_CL_PROPERTY(buffer, cl_mem, Buffer)
            .DEF_CL_PROPERTY(tempBuffer, cl_mem, Buffer)
            .DEF_CL_PROPERTY(inputBuffer, cl_mem, Buffer)
            .DEF_CL_PROPERTY(outputBuffer, cl_mem, Buffer)
            .DEF_CL_PROPERTY(kernel, cl_mem, Buffer)
            .DEF_SIMPLE_RW(bufferOffset)
            .DEF_SIMPLE_RW(tempBufferOffset)
            .DEF_SIMPLE_RW(inputBufferOffset)
            .DEF_SIMPLE_RW(outputBufferOffset)
            .DEF_SIMPLE_RW(kernelOffset)
        ;
    }

    {
        typedef VkFFTResult cls;
        py::enum_<cls>(m, "Result")
            .value("SUCCESS", VKFFT_SUCCESS)
            .DEF_ERROR_VALUE(MALLOC_FAILED)
            .DEF_ERROR_VALUE(INSUFFICIENT_CODE_BUFFER)
            .DEF_ERROR_VALUE(INSUFFICIENT_TEMP_BUFFER)
            .DEF_ERROR_VALUE(PLAN_NOT_INITIALIZED)
            .DEF_ERROR_VALUE(NULL_TEMP_PASSED)
            .DEF_ERROR_VALUE(INVALID_PHYSICAL_DEVICE)
            .DEF_ERROR_VALUE(INVALID_DEVICE)
            .DEF_ERROR_VALUE(INVALID_QUEUE)
            .DEF_ERROR_VALUE(INVALID_COMMAND_POOL)
            .DEF_ERROR_VALUE(INVALID_FENCE)
            .DEF_ERROR_VALUE(ONLY_FORWARD_FFT_INITIALIZED)
            .DEF_ERROR_VALUE(ONLY_INVERSE_FFT_INITIALIZED)
            .DEF_ERROR_VALUE(INVALID_CONTEXT)
            .DEF_ERROR_VALUE(INVALID_PLATFORM)
            .DEF_ERROR_VALUE(ENABLED_saveApplicationToString)
            .DEF_ERROR_VALUE(EMPTY_FFTdim)
            .DEF_ERROR_VALUE(EMPTY_size)
            .DEF_ERROR_VALUE(EMPTY_bufferSize)
            .DEF_ERROR_VALUE(EMPTY_buffer)
            .DEF_ERROR_VALUE(EMPTY_tempBufferSize)
            .DEF_ERROR_VALUE(EMPTY_tempBuffer)
            .DEF_ERROR_VALUE(EMPTY_inputBufferSize)
            .DEF_ERROR_VALUE(EMPTY_inputBuffer)
            .DEF_ERROR_VALUE(EMPTY_outputBufferSize)
            .DEF_ERROR_VALUE(EMPTY_outputBuffer)
            .DEF_ERROR_VALUE(EMPTY_kernelSize)
            .DEF_ERROR_VALUE(EMPTY_kernel)
            .DEF_ERROR_VALUE(EMPTY_applicationString)
            .DEF_ERROR_VALUE(UNSUPPORTED_RADIX)
            .DEF_ERROR_VALUE(UNSUPPORTED_FFT_LENGTH)
            .DEF_ERROR_VALUE(UNSUPPORTED_FFT_LENGTH_R2C)
            .DEF_ERROR_VALUE(UNSUPPORTED_FFT_LENGTH_DCT)
            .DEF_ERROR_VALUE(UNSUPPORTED_FFT_OMIT)
            .DEF_ERROR_VALUE(FAILED_TO_ALLOCATE)
            .DEF_ERROR_VALUE(FAILED_TO_MAP_MEMORY)
            .DEF_ERROR_VALUE(FAILED_TO_ALLOCATE_COMMAND_BUFFERS)
            .DEF_ERROR_VALUE(FAILED_TO_BEGIN_COMMAND_BUFFER)
            .DEF_ERROR_VALUE(FAILED_TO_END_COMMAND_BUFFER)
            .DEF_ERROR_VALUE(FAILED_TO_SUBMIT_QUEUE)
            .DEF_ERROR_VALUE(FAILED_TO_WAIT_FOR_FENCES)
            .DEF_ERROR_VALUE(FAILED_TO_RESET_FENCES)
            .DEF_ERROR_VALUE(FAILED_TO_CREATE_DESCRIPTOR_POOL)
            .DEF_ERROR_VALUE(FAILED_TO_CREATE_DESCRIPTOR_SET_LAYOUT)
            .DEF_ERROR_VALUE(FAILED_TO_ALLOCATE_DESCRIPTOR_SETS)
            .DEF_ERROR_VALUE(FAILED_TO_CREATE_PIPELINE_LAYOUT)
            .DEF_ERROR_VALUE(FAILED_SHADER_PREPROCESS)
            .DEF_ERROR_VALUE(FAILED_SHADER_PARSE)
            .DEF_ERROR_VALUE(FAILED_SHADER_LINK)
            .DEF_ERROR_VALUE(FAILED_SPIRV_GENERATE)
            .DEF_ERROR_VALUE(FAILED_TO_CREATE_SHADER_MODULE)
            .DEF_ERROR_VALUE(FAILED_TO_CREATE_INSTANCE)
            .DEF_ERROR_VALUE(FAILED_TO_SETUP_DEBUG_MESSENGER)
            .DEF_ERROR_VALUE(FAILED_TO_FIND_PHYSICAL_DEVICE)
            .DEF_ERROR_VALUE(FAILED_TO_CREATE_DEVICE)
            .DEF_ERROR_VALUE(FAILED_TO_CREATE_FENCE)
            .DEF_ERROR_VALUE(FAILED_TO_CREATE_COMMAND_POOL)
            .DEF_ERROR_VALUE(FAILED_TO_CREATE_BUFFER)
            .DEF_ERROR_VALUE(FAILED_TO_ALLOCATE_MEMORY)
            .DEF_ERROR_VALUE(FAILED_TO_BIND_BUFFER_MEMORY)
            .DEF_ERROR_VALUE(FAILED_TO_FIND_MEMORY)
            .DEF_ERROR_VALUE(FAILED_TO_SYNCHRONIZE)
            .DEF_ERROR_VALUE(FAILED_TO_COPY)
            .DEF_ERROR_VALUE(FAILED_TO_CREATE_PROGRAM)
            .DEF_ERROR_VALUE(FAILED_TO_COMPILE_PROGRAM)
            .DEF_ERROR_VALUE(FAILED_TO_GET_CODE_SIZE)
            .DEF_ERROR_VALUE(FAILED_TO_GET_CODE)
            .DEF_ERROR_VALUE(FAILED_TO_DESTROY_PROGRAM)
            .DEF_ERROR_VALUE(FAILED_TO_LOAD_MODULE)
            .DEF_ERROR_VALUE(FAILED_TO_GET_FUNCTION)
            .DEF_ERROR_VALUE(FAILED_TO_SET_DYNAMIC_SHARED_MEMORY)
            .DEF_ERROR_VALUE(FAILED_TO_MODULE_GET_GLOBAL)
            .DEF_ERROR_VALUE(FAILED_TO_LAUNCH_KERNEL)
            .DEF_ERROR_VALUE(FAILED_TO_EVENT_RECORD)
            .DEF_ERROR_VALUE(FAILED_TO_ADD_NAME_EXPRESSION)
            .DEF_ERROR_VALUE(FAILED_TO_INITIALIZE)
            .DEF_ERROR_VALUE(FAILED_TO_SET_DEVICE_ID)
            .DEF_ERROR_VALUE(FAILED_TO_GET_DEVICE)
            .DEF_ERROR_VALUE(FAILED_TO_CREATE_CONTEXT)
            .DEF_ERROR_VALUE(FAILED_TO_CREATE_PIPELINE)
            .DEF_ERROR_VALUE(FAILED_TO_SET_KERNEL_ARG)
            .DEF_ERROR_VALUE(FAILED_TO_CREATE_COMMAND_QUEUE)
            .DEF_ERROR_VALUE(FAILED_TO_RELEASE_COMMAND_QUEUE)
            .DEF_ERROR_VALUE(FAILED_TO_ENUMERATE_DEVICES)
            .DEF_ERROR_VALUE(FAILED_TO_GET_ATTRIBUTE)
            .DEF_ERROR_VALUE(FAILED_TO_CREATE_EVENT)
        ;
    }

    m.attr("__version__") = VkFFTGetVersion();
}
