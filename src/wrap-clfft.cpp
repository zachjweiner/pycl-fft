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

#include <clFFT.h>
#include <clFFT.version.h>

#define CALL_GUARDED(err, CALL) \
	clfftStatus err = CALL; \
    if (err != 0) throw std::runtime_error(std::to_string((int)err));

#define DEF_SIMPLE_RW(name) \
  def_readwrite(#name, &cls::name)

#define set_plan_attr(Name, T) \
    [](cls *self, T val) \
    { \
        CALL_GUARDED(err, clfftSet##Name(self->plan_handle, (T)val)) \
    }

#define get_plan_attr(Name, T) \
    [](cls *self) \
    { \
        T out; \
        CALL_GUARDED(err, clfftGet##Name(self->plan_handle, &out)) \
        return out; \
    }

#define DEF_PLAN_RW(name, Name, T) \
  def_property(#name, get_plan_attr(Name, T), set_plan_attr(Name, T))

#define set_plan_attr_dir(Name, T, dir) \
    [](cls *self, T val) \
    { \
        CALL_GUARDED(err, clfftSet##Name(self->plan_handle, dir, (T)val)) \
    }

#define get_plan_attr_dir(Name, T, dir) \
    [](cls *self) \
    { \
        T out; \
        CALL_GUARDED(err, clfftGet##Name(self->plan_handle, dir, &out)) \
        return out; \
    }

#define DEF_PLAN_RW_DIR(name, Name, T, dir) \
  def_property(#name, get_plan_attr_dir(Name, T, dir), set_plan_attr_dir(Name, T, dir))

#define set_plan_attr_vec(Name, T) \
    [](cls *self, const std::vector<T> &val) \
    { \
        CALL_GUARDED(err, clfftSet##Name(self->plan_handle, (clfftDim)val.size(), const_cast<T*>(val.data()))) \
    }

#define get_plan_attr_vec(Name, T) \
    [](cls *self) \
    { \
        clfftDim dim = self->get_dim(); \
        std::vector<T> result(dim); \
        CALL_GUARDED(err, clfftGet##Name(self->plan_handle, dim, const_cast<T*>(result.data()))) \
        return result; \
    }

#define DEF_PLAN_RW_VEC(name, Name, T) \
  def_property(#name, get_plan_attr_vec(Name, T), set_plan_attr_vec(Name, T))

#define DEF_PREFIXED_VALUE(name) \
  value(#name, CLFFT_##name)

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

clfftSetupData init_setup_data() {
    clfftSetupData setup_data;
	CALL_GUARDED(err, clfftInitSetupData(&setup_data))
	CALL_GUARDED(err2, clfftSetup(&setup_data))
    return setup_data;
}

std::tuple<int, int, int> get_clfft_version() {
    cl_uint libMajor, libMinor, libPatch;
    CALL_GUARDED(err, clfftGetVersion(&libMajor, &libMinor, &libPatch))
    return std::make_tuple((int)libMajor, (int)libMinor, (int)libPatch);
}

#define py_objs_to_ptr_vector(py_list, name, type) \
    std::vector<type> name; \
    if (! py_list.is(py::none())) \
    { \
        for (py::handle obj: py_list) \
            name.push_back((type)(obj.attr("int_ptr").cast<intptr_t>())); \
    }


class plan {
    private:
        bool destroyed = false;

    public:
        clfftPlanHandle plan_handle;
        cl_context* context;

        plan(py::object ctx, const int dimension, const std::vector<size_t> lens)
        {
            context = new cl_context;
            *context = (cl_context)(ctx.attr("int_ptr").cast<intptr_t>());

        	CALL_GUARDED(err, clfftCreateDefaultPlan(
                &plan_handle, *context, (clfftDim)dimension, lens.data())
            )
        }

        void destroy() {
            if (!destroyed)
            {
                CALL_GUARDED(err, clfftDestroyPlan(&plan_handle))
                destroyed = true;
            }
        }

        void bake(py::object py_queues) {
            py_objs_to_ptr_vector(py_queues, queues, cl_command_queue)
            cl_uint num_queues = queues.size();

            CALL_GUARDED(err, clfftBakePlan(
                plan_handle, num_queues, queues.data(), nullptr, nullptr)
            )
        }

        std::vector<py::object> enqueue_transform(
                clfftDirection dir, py::object queues, py::object wait_for,
                py::object inputs, py::object outputs, py::object temp_buffer) {
            py_objs_to_ptr_vector(queues, cl_queues, cl_command_queue)
            cl_uint num_queues = cl_queues.size();

            py_objs_to_ptr_vector(wait_for, cl_wait_for, cl_event)
            cl_uint num_wait_for = cl_wait_for.size();

            std::vector<cl_event> cl_events(num_queues);
            py_objs_to_ptr_vector(inputs, cl_inputs, cl_mem)
            py_objs_to_ptr_vector(outputs, cl_outputs, cl_mem)

            cl_mem temp_buffer_ptr = nullptr;
            if (! temp_buffer.is(py::none()))
            {
                temp_buffer_ptr = (cl_mem)(temp_buffer.attr("int_ptr").cast<intptr_t>());
            }

            CALL_GUARDED(err, clfftEnqueueTransform(
                plan_handle, dir, num_queues, cl_queues.data(), num_wait_for,
                cl_wait_for.data(), cl_events.data(), cl_inputs.data(), cl_outputs.data(),
                temp_buffer_ptr)
            )

            std::vector<py::object> events;
            py::object cl = py::module_::import("pyopencl");
            for (cl_event evt: cl_events)
            {
                intptr_t int_ptr = (intptr_t)evt;
                events.push_back(cl.attr("Event").attr("from_int_ptr")(int_ptr));
            }

            return events;
        }

        void set_callback(
            const char* funcName, const char* funcString, int localMemSize,
            clfftCallbackType callbackType, cl_mem *userdata, int numUserdataBuffers) {
                CALL_GUARDED(err, clfftSetPlanCallback(
                    plan_handle, funcName, funcString, localMemSize, callbackType,
                    userdata, numUserdataBuffers)
                )
            }

        clfftDim get_dim() {
            clfftDim dimension;
            cl_uint	size;
            CALL_GUARDED(err, clfftGetPlanDim(plan_handle, &dimension, &size))
            return dimension;
        }

        size_t get_in_distance() {
            size_t idist, odist;
            CALL_GUARDED(err, clfftGetPlanDistance(plan_handle, &idist, &odist))
            return idist;
        }

        size_t get_out_distance() {
            size_t idist, odist;
            CALL_GUARDED(err, clfftGetPlanDistance(plan_handle, &idist, &odist))
            return odist;
        }

        void set_in_distance(size_t idist) {
            size_t tmp;
            size_t odist;
            CALL_GUARDED(err, clfftGetPlanDistance(plan_handle, &tmp, &odist))
            CALL_GUARDED(err2, clfftSetPlanDistance(plan_handle, idist, odist))
        }

        void set_out_distance(size_t odist) {
            size_t idist;
            size_t tmp;
            CALL_GUARDED(err, clfftGetPlanDistance(plan_handle, &idist, &tmp))
            CALL_GUARDED(err2, clfftSetPlanDistance(plan_handle, idist, odist))
        }

        clfftLayout get_in_layout() {
            clfftLayout ilayout, olayout;
            CALL_GUARDED(err, clfftGetLayout(plan_handle, &ilayout, &olayout))
            return ilayout;
        }

        clfftLayout get_out_layout() {
            clfftLayout ilayout, olayout;
            CALL_GUARDED(err, clfftGetLayout(plan_handle, &ilayout, &olayout))
            return olayout;
        }

        void set_in_layout(clfftLayout ilayout) {
            clfftLayout tmp;
            clfftLayout olayout;
            CALL_GUARDED(err, clfftGetLayout(plan_handle, &tmp, &olayout))
            if (ilayout == CLFFT_REAL)
            {
                // if (olayout != CLFFT_HERMITIAN_PLANAR)
                    olayout = CLFFT_HERMITIAN_INTERLEAVED;
            }
            else if (ilayout == CLFFT_HERMITIAN_INTERLEAVED
                     || ilayout == CLFFT_HERMITIAN_PLANAR)
                olayout = CLFFT_REAL;
            else
                olayout = ilayout;
            CALL_GUARDED(err2, clfftSetLayout(plan_handle, ilayout, olayout))
        }

        void set_out_layout(clfftLayout olayout) {
            clfftLayout ilayout;
            clfftLayout tmp;
            CALL_GUARDED(err, clfftGetLayout(plan_handle, &ilayout, &tmp))
            if (olayout == CLFFT_HERMITIAN_INTERLEAVED || olayout == CLFFT_HERMITIAN_PLANAR)
                ilayout = CLFFT_REAL;
            else if (olayout == CLFFT_REAL)
            {
                // if (ilayout != CLFFT_HERMITIAN_PLANAR)
                    ilayout = CLFFT_HERMITIAN_INTERLEAVED;
            }
            else
                ilayout = olayout;
            CALL_GUARDED(err2, clfftSetLayout(plan_handle, ilayout, olayout))
        }
};

PYBIND11_MODULE(_clfft, m)
{
    {
        typedef clfftSetupData cls;
        py::class_<cls>(m, "SetupData")
            .def(py::init(&init_setup_data))
            .DEF_SIMPLE_RW(major)
            .DEF_SIMPLE_RW(minor)
            .DEF_SIMPLE_RW(patch)
            .DEF_SIMPLE_RW(debugFlags)
        ;
    }

    {
        typedef plan cls;
        py::class_<cls>(m, "Plan")
            .def(py::init<py::object, const int, const std::vector<size_t>>())
            .def("destroy", &plan::destroy)
            .def_property_readonly("context", cl_ptr_to_py_obj(context, Context))
            .DEF_PLAN_RW(precision, PlanPrecision, clfftPrecision)
            .DEF_PLAN_RW_DIR(forward_scale, PlanScale, cl_float, CLFFT_FORWARD)
            .DEF_PLAN_RW_DIR(backward_scale, PlanScale, cl_float, CLFFT_BACKWARD)
            .DEF_PLAN_RW(batch_size, PlanBatchSize, size_t)
            .def_property("dimension", &plan::get_dim, set_plan_attr(PlanDim, clfftDim))
            .DEF_PLAN_RW_VEC(lengths, PlanLength, size_t)
            .DEF_PLAN_RW_VEC(input_strides, PlanInStride, size_t)
            .DEF_PLAN_RW_VEC(output_strides, PlanOutStride, size_t)
            .def_property("input_distance", &plan::get_in_distance, &plan::set_in_distance)
            .def_property("output_distance", &plan::get_out_distance, &plan::set_out_distance)
            .def_property("input_layout", &plan::get_in_layout, &plan::set_in_layout)
            .def_property("output_layout", &plan::get_out_layout, &plan::set_out_layout)
            .DEF_PLAN_RW(placeness, ResultLocation, clfftResultLocation)
            .DEF_PLAN_RW(transposed, PlanTransposeResult, clfftResultTransposed)
            .def_property_readonly("temp_buffer_size", get_plan_attr(TmpBufSize, size_t))
            .def("bake", &plan::bake)
            .def("enqueue_transform", &plan::enqueue_transform)
            // unimplemented: clfftCopyPlan, clfftSetPlanCallback
        ;
    }

    py::enum_<clfftStatus>(m, "Status")
        .DEF_PREFIXED_VALUE(INVALID_GLOBAL_WORK_SIZE)
        .DEF_PREFIXED_VALUE(INVALID_MIP_LEVEL)
        .DEF_PREFIXED_VALUE(INVALID_BUFFER_SIZE)
        .DEF_PREFIXED_VALUE(INVALID_GL_OBJECT)
        .DEF_PREFIXED_VALUE(INVALID_OPERATION)
        .DEF_PREFIXED_VALUE(INVALID_EVENT)
        .DEF_PREFIXED_VALUE(INVALID_EVENT_WAIT_LIST)
        .DEF_PREFIXED_VALUE(INVALID_GLOBAL_OFFSET)
        .DEF_PREFIXED_VALUE(INVALID_WORK_ITEM_SIZE)
        .DEF_PREFIXED_VALUE(INVALID_WORK_GROUP_SIZE)
        .DEF_PREFIXED_VALUE(INVALID_WORK_DIMENSION)
        .DEF_PREFIXED_VALUE(INVALID_KERNEL_ARGS)
        .DEF_PREFIXED_VALUE(INVALID_ARG_SIZE)
        .DEF_PREFIXED_VALUE(INVALID_ARG_VALUE)
        .DEF_PREFIXED_VALUE(INVALID_ARG_INDEX)
        .DEF_PREFIXED_VALUE(INVALID_KERNEL)
        .DEF_PREFIXED_VALUE(INVALID_KERNEL_DEFINITION)
        .DEF_PREFIXED_VALUE(INVALID_KERNEL_NAME)
        .DEF_PREFIXED_VALUE(INVALID_PROGRAM_EXECUTABLE)
        .DEF_PREFIXED_VALUE(INVALID_PROGRAM)
        .DEF_PREFIXED_VALUE(INVALID_BUILD_OPTIONS)
        .DEF_PREFIXED_VALUE(INVALID_BINARY)
        .DEF_PREFIXED_VALUE(INVALID_SAMPLER)
        .DEF_PREFIXED_VALUE(INVALID_IMAGE_SIZE)
        .DEF_PREFIXED_VALUE(INVALID_IMAGE_FORMAT_DESCRIPTOR)
        .DEF_PREFIXED_VALUE(INVALID_MEM_OBJECT)
        .DEF_PREFIXED_VALUE(INVALID_HOST_PTR)
        .DEF_PREFIXED_VALUE(INVALID_COMMAND_QUEUE)
        .DEF_PREFIXED_VALUE(INVALID_QUEUE_PROPERTIES)
        .DEF_PREFIXED_VALUE(INVALID_CONTEXT)
        .DEF_PREFIXED_VALUE(INVALID_DEVICE)
        .DEF_PREFIXED_VALUE(INVALID_PLATFORM)
        .DEF_PREFIXED_VALUE(INVALID_DEVICE_TYPE)
        .DEF_PREFIXED_VALUE(INVALID_VALUE)
        .DEF_PREFIXED_VALUE(MAP_FAILURE)
        .DEF_PREFIXED_VALUE(BUILD_PROGRAM_FAILURE)
        .DEF_PREFIXED_VALUE(IMAGE_FORMAT_NOT_SUPPORTED)
        .DEF_PREFIXED_VALUE(IMAGE_FORMAT_MISMATCH)
        .DEF_PREFIXED_VALUE(MEM_COPY_OVERLAP)
        .DEF_PREFIXED_VALUE(PROFILING_INFO_NOT_AVAILABLE)
        .DEF_PREFIXED_VALUE(OUT_OF_HOST_MEMORY)
        .DEF_PREFIXED_VALUE(OUT_OF_RESOURCES)
        .DEF_PREFIXED_VALUE(MEM_OBJECT_ALLOCATION_FAILURE)
        .DEF_PREFIXED_VALUE(COMPILER_NOT_AVAILABLE)
        .DEF_PREFIXED_VALUE(DEVICE_NOT_AVAILABLE)
        .DEF_PREFIXED_VALUE(DEVICE_NOT_FOUND)
        .DEF_PREFIXED_VALUE(SUCCESS)
        .DEF_PREFIXED_VALUE(BUGCHECK)
        .DEF_PREFIXED_VALUE(NOTIMPLEMENTED)
        .DEF_PREFIXED_VALUE(TRANSPOSED_NOTIMPLEMENTED)
        .DEF_PREFIXED_VALUE(FILE_NOT_FOUND)
        .DEF_PREFIXED_VALUE(FILE_CREATE_FAILURE)
        .DEF_PREFIXED_VALUE(VERSION_MISMATCH)
        .DEF_PREFIXED_VALUE(INVALID_PLAN)
        .DEF_PREFIXED_VALUE(DEVICE_NO_DOUBLE)
        .DEF_PREFIXED_VALUE(DEVICE_MISMATCH)
    ;

    py::enum_<clfftDim>(m, "Dimension")
        .DEF_PREFIXED_VALUE(1D)
        .DEF_PREFIXED_VALUE(2D)
        .DEF_PREFIXED_VALUE(3D)
    ;

    py::enum_<clfftLayout>(m, "Layout")
        .DEF_PREFIXED_VALUE(COMPLEX_INTERLEAVED)
        .DEF_PREFIXED_VALUE(COMPLEX_PLANAR)
        .DEF_PREFIXED_VALUE(HERMITIAN_INTERLEAVED)
        .DEF_PREFIXED_VALUE(HERMITIAN_PLANAR)
        .DEF_PREFIXED_VALUE(REAL)
    ;

    py::enum_<clfftPrecision>(m, "Precision")
        .DEF_PREFIXED_VALUE(SINGLE)
        .DEF_PREFIXED_VALUE(DOUBLE)
        .DEF_PREFIXED_VALUE(SINGLE_FAST)
        .DEF_PREFIXED_VALUE(DOUBLE_FAST)
    ;

    py::enum_<clfftDirection>(m, "Direction")
        .DEF_PREFIXED_VALUE(FORWARD)
        .DEF_PREFIXED_VALUE(BACKWARD)
        .DEF_PREFIXED_VALUE(MINUS)
        .DEF_PREFIXED_VALUE(PLUS)
    ;

    py::enum_<clfftResultLocation>(m, "ResultLocation")
        .DEF_PREFIXED_VALUE(INPLACE)
        .DEF_PREFIXED_VALUE(OUTOFPLACE)
    ;

    py::enum_<clfftResultTransposed>(m, "ResultTransposition")
        .DEF_PREFIXED_VALUE(NOTRANSPOSE)
        .DEF_PREFIXED_VALUE(TRANSPOSED)
    ;

    py::enum_<clfftCallbackType>(m, "CallbackType")
        .value("PRECALLBACK", PRECALLBACK)
        .value("POSTCALLBACK", POSTCALLBACK)
    ;

    m.def("teardown", &clfftTeardown);

    m.attr("__version__") = get_clfft_version();
}
