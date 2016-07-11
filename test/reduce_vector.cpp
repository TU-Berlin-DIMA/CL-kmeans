#include <cl_kernels/reduce_vector_parcol_api.hpp>
#include <matrix.hpp>

#include <cstdint>
#include <memory>
#include <vector>
#include <algorithm>

#define BOOST_TEST_MODULE TestReduceVector
#include <boost/test/unit_test.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/test/data/monomorphic.hpp>

#include "data_generator.hpp"
#include "opencl_setup.hpp"

namespace bdata = boost::unit_test::data;

void reduce_vector_verify(
        cle::Matrix<uint32_t, std::allocator<uint32_t>, uint32_t> const& data,
        std::vector<uint32_t>& reduced
        ) {

    reduced.resize(data.rows());
    std::fill(reduced.begin(), reduced.end(), 0);

    for (uint32_t col = 0; col < data.cols(); ++col) {
        for (uint32_t row = 0; row < data.rows(); ++row) {
            reduced[row] += data(row, col);
        }
    }
}

void reduce_vector_run(
        cl::Context context,
        cl::CommandQueue queue,
        cle::Matrix<uint32_t, std::allocator<uint32_t>, uint32_t> const& data,
        std::vector<uint32_t>& reduced
        ) {

    reduced.resize(data.rows());
    cle::TypedBuffer<cl_uint> d_buffer(context, CL_MEM_READ_WRITE, data.size());

    cle::ReduceVectorParcolAPI<cl_uint, cl_uint> reducevector;
    reducevector.initialize(context);
    cl::Event event;

    queue.enqueueWriteBuffer(
        d_buffer,
        CL_FALSE,
        0,
        d_buffer.bytes(),
        data.data(),
        NULL,
        NULL);

    reducevector(
        cl::EnqueueArgs(
            queue,
            cl::NDRange(0),
            cl::NDRange(0)
            ),
        data.cols(),
        data.rows(),
        d_buffer,
        event);

    queue.enqueueReadBuffer(
        d_buffer,
        CL_TRUE,
        0,
        reduced.size() * sizeof(uint32_t),
        reduced.data(),
        NULL,
        NULL);
}

BOOST_AUTO_TEST_CASE(ReduceVectorParcolAscending) {

    auto const& data = dgen.ascending();
    std::vector<uint32_t> test_output;
    std::vector<uint32_t> verify_output;

    reduce_vector_run(context, queue, data, test_output);
    reduce_vector_verify(data, verify_output);

    BOOST_TEST(test_output == verify_output, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(ReduceVectorParcolLarge) {

    auto const& data = dgen.large();
    std::vector<uint32_t> test_output;
    std::vector<uint32_t> verify_output;

    reduce_vector_run(context, queue, data, test_output);
    reduce_vector_verify(data, verify_output);

    BOOST_TEST(test_output == verify_output, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_CASE(ReduceVectorParcolRandom) {

    auto const& data = dgen.random();
    std::vector<uint32_t> test_output;
    std::vector<uint32_t> verify_output;

    reduce_vector_run(context, queue, data, test_output);
    reduce_vector_verify(data, verify_output);

    BOOST_TEST(test_output == verify_output, boost::test_tools::per_element());
}

BOOST_AUTO_TEST_SUITE_END()
