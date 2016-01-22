#ifndef TYPED_BUFFER_HPP
#define TYPED_BUFFER_HPP

#ifdef MAC
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

namespace cle {

template <typename T>
class TypedBuffer : public cl::Buffer {
public:

    typedef T value_type;

    TypedBuffer(
            const cl::Context& context,
            cl_mem_flags flags,
            ::size_t size,
            T* host_ptr = NULL,
            cl_int* err = NULL
            )
        : cl::Buffer(context, flags, size * sizeof(T), host_ptr, err)
        , size_(size)
    {}

    TypedBuffer(
            cl_mem_flags flags,
            ::size_t size,
            T* host_ptr = NULL,
            cl_int* err = NULL
            )
        : cl::Buffer(flags, size * sizeof(T), host_ptr, err)
        , size_(size)
    {}

    // template<typename IteratorType>
    // TypedBuffer(
    //         IteratorType startIterator,
    //         IteratorType endIterator,
    //         bool readOnly,
    //         bool useHostPtr = false,
    //         cl_int* err = NULL)
    //     : cl::Buffer(startIterator, endIterator, readOnly, useHostPtr, err)
    // , size_((endIterator - startIterator) / sizeof(T))
    // {}

    // template<typename IteratorType>
    // TypedBuffer(
    //         const cl::Context& context,
    //         IteratorType startIterator,
    //         IteratorType endIterator,
    //         bool readOnly,
    //         bool useHostPtr = false,
    //         cl_int* err = NULL
    //         )
    //     : cl::Buffer(context, startIterator, endIterator, readOnly, useHostPtr, err)
    //     , size_((endIterator - startIterator) / sizeof(T))
    // {}

    // template<typename IteratorType>
    // TypedBuffer(
    //         cl::CommandQueue const& queue,
    //         IteratorType startIterator,
    //         IteratorType endIterator,
    //         bool readOnly,
    //         bool useHostPtr = false,
    //         cl_int* err = NULL
    //         )
    //     : cl::Buffer(queue, startIterator, endIterator, readOnly, useHostPtr, err)
    //     , size_((endIterator - startIterator) / sizeof(T))
    // {}

    TypedBuffer() : cl::Buffer(), size_(0) {}

    // __CL_EXPLICIT_CONSTRUCTORS TypedBuffer(cl_mem const& buffer)
    //     : cl::Buffer(buffer)
    //     , size_(0)
    // {}


    // Shallow copy
    // TypedBuffer& operator = (cl_mem const& rhs) {
    //     size_ = 0; // TODO: fetch size from cl::Buffer::getInfo()
    //     cl::Buffer::operator=(rhs);
    //     return *this;
    // }

    // TypedBuffer(cl::Buffer const& buf)
    //     : cl::Buffer(buf)
    //     , size_(0)
    // {}

    // TypedBuffer<T>& operator = (cl::Buffer const& rhs) {
    //     size_ = 0; // TODO: fetch size from cl::Buffer::getInfo()
    //     cl::Buffer::operator=(rhs);
    //     return *this;
    // }


    TypedBuffer<T>& operator = (TypedBuffer<T> const& rhs) {
        size_ = rhs.size();
        cl::Buffer::operator=(rhs);
        return *this;
    }

    TypedBuffer(TypedBuffer<T> const& buf)
        : cl::Buffer(buf)
        , size_(buf.size())
    {}

#if defined(CL_HPP_RVALUE_REFERENCE_SUPPORTED)
    // TypedBuffer(cl::Buffer&& buf) : cl::Buffer(std::move(buf)) {}
    //
    // TypedBuffer(TypedBuffer<T>&& buf) : cl::Buffer(std::move(buf)) {}
    //
    // TypedBuffer& operator = (cl::Buffer &&rhs) {
    //     size_ = 0; // TODO
    //     cl::Buffer::operator=(std::move(rhs));
    //     return *this;
    // }
    //
    // TypedBuffer& operator = (TypedBuffer &&rhs) {
    //     size_ = std::move(rhs.size_);
    //     cl::Buffer::operator=(std::move(rhs));
    //     return *this;
    // }
#endif

    ::size_t size() {
        return size_;
    }

    ::size_t bytes() {
        return this->size() * sizeof(T);
    }

private:
    // Size in number of "T" elements
    ::size_t size_;
};

}

#endif /* TYPED_BUFFER_HPP */
