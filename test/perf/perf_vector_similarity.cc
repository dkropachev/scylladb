/*
 * Copyright (C) 2026-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

#include "cql3/functions/vector_similarity_fcts.hh"
#include "db/config.hh"
#include "types/vector.hh"

#include <array>
#include <bit>
#include <cstdint>
#include <string_view>
#include <vector>

#include <seastar/core/byteorder.hh>
#include <seastar/testing/perf_tests.hh>

namespace {

enum class similarity_function {
    cosine,
    euclidean,
    dot_product,
};

enum class similarity_backend {
    scalar,
    ssse3,
    automatic,
};

bytes serialize_float_vector(std::span<const float> values) {
    bytes serialized(bytes::initialized_later{}, values.size() * sizeof(float));
    char* out = reinterpret_cast<char*>(serialized.data());
    for (float value : values) {
        produce_be<uint32_t>(out, std::bit_cast<uint32_t>(value));
    }
    return serialized;
}

template <size_t Dimension>
struct vector_similarity {
    std::vector<float> left;
    std::vector<float> right;
    bytes_opt serialized_left;
    bytes_opt serialized_right;

    vector_similarity()
        : left(Dimension)
        , right(Dimension) {
        for (size_t i = 0; i < Dimension; ++i) {
            left[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.25f;
            right[i] = static_cast<float>(static_cast<int>((i * 7) % 19) - 9) * 0.125f;
        }
        serialized_left = serialize_float_vector(left);
        serialized_right = serialize_float_vector(right);
    }

    static constexpr vector_dimension_t dimension = static_cast<vector_dimension_t>(Dimension);
};

template <similarity_function Function>
const cql3::functions::function_name& function_name_for();

template <>
const cql3::functions::function_name& function_name_for<similarity_function::cosine>() {
    return cql3::functions::SIMILARITY_COSINE_FUNCTION_NAME;
}

template <>
const cql3::functions::function_name& function_name_for<similarity_function::euclidean>() {
    return cql3::functions::SIMILARITY_EUCLIDEAN_FUNCTION_NAME;
}

template <>
const cql3::functions::function_name& function_name_for<similarity_function::dot_product>() {
    return cql3::functions::SIMILARITY_DOT_PRODUCT_FUNCTION_NAME;
}

template <similarity_backend Backend>
db::simd_optimization_mode backend_mode_for();

template <>
db::simd_optimization_mode backend_mode_for<similarity_backend::scalar>() {
    return db::simd_optimization_mode::off;
}

template <>
db::simd_optimization_mode backend_mode_for<similarity_backend::ssse3>() {
    return db::simd_optimization_mode::sse;
}

template <>
db::simd_optimization_mode backend_mode_for<similarity_backend::automatic>() {
    return db::simd_optimization_mode::automatic;
}

template <size_t Dimension>
std::vector<data_type> vector_arg_types() {
    auto type = vector_type_impl::get_instance(float_type, static_cast<vector_dimension_t>(Dimension));
    return {type, type};
}

template <size_t Dimension, similarity_function Function, similarity_backend Backend = similarity_backend::automatic>
struct vector_similarity_execute : vector_similarity<Dimension> {
    cql3::functions::vector_similarity_fct function;
    std::array<bytes_opt, 2> parameters;

    vector_similarity_execute()
        : vector_similarity<Dimension>()
        , function(function_name_for<Function>().name, vector_arg_types<Dimension>(), backend_mode_for<Backend>())
        , parameters{this->serialized_left, this->serialized_right} {
    }
};

template <size_t Dimension, similarity_function Function, similarity_backend Backend>
struct vector_similarity_serialized_backend : vector_similarity<Dimension> {
    cql3::functions::serialized_similarity_function_t function;

    vector_similarity_serialized_backend()
        : vector_similarity<Dimension>()
        , function(cql3::functions::detail::select_serialized_similarity_function(
                function_name_for<Function>(),
                backend_mode_for<Backend>())) {
    }
};

template <typename Fixture, typename Func>
void score_legacy(Fixture& fixture, Func func) {
    const auto left = cql3::functions::detail::extract_float_vector(fixture.serialized_left, fixture.dimension);
    const auto right = cql3::functions::detail::extract_float_vector(fixture.serialized_right, fixture.dimension);
    perf_tests::do_not_optimize(func(left, right));
}

template <typename Fixture, typename Func>
void score_serialized(Fixture& fixture, Func func) {
    perf_tests::do_not_optimize(func(fixture.serialized_left, fixture.serialized_right, fixture.dimension));
}

template <typename Fixture>
void score_selected_serialized(Fixture& fixture) {
    perf_tests::do_not_optimize(fixture.function(fixture.serialized_left, fixture.serialized_right, fixture.dimension));
}

template <typename Fixture>
void score_execute(Fixture& fixture) {
    perf_tests::do_not_optimize(fixture.function.execute(fixture.parameters));
}

using vector_similarity_128 = vector_similarity<128>;
using vector_similarity_768 = vector_similarity<768>;
using vector_similarity_1536 = vector_similarity<1536>;
using vector_similarity_tail_130 = vector_similarity<130>;
using vector_similarity_tail_777 = vector_similarity<777>;
using vector_similarity_tail_1543 = vector_similarity<1543>;

using vector_similarity_serialized_scalar_cosine_128 = vector_similarity_serialized_backend<128, similarity_function::cosine, similarity_backend::scalar>;
using vector_similarity_serialized_scalar_euclidean_128 = vector_similarity_serialized_backend<128, similarity_function::euclidean, similarity_backend::scalar>;
using vector_similarity_serialized_scalar_dot_product_128 =
        vector_similarity_serialized_backend<128, similarity_function::dot_product, similarity_backend::scalar>;
using vector_similarity_serialized_scalar_cosine_768 = vector_similarity_serialized_backend<768, similarity_function::cosine, similarity_backend::scalar>;
using vector_similarity_serialized_scalar_euclidean_768 = vector_similarity_serialized_backend<768, similarity_function::euclidean, similarity_backend::scalar>;
using vector_similarity_serialized_scalar_dot_product_768 =
        vector_similarity_serialized_backend<768, similarity_function::dot_product, similarity_backend::scalar>;
using vector_similarity_serialized_scalar_cosine_1536 = vector_similarity_serialized_backend<1536, similarity_function::cosine, similarity_backend::scalar>;
using vector_similarity_serialized_scalar_euclidean_1536 =
        vector_similarity_serialized_backend<1536, similarity_function::euclidean, similarity_backend::scalar>;
using vector_similarity_serialized_scalar_dot_product_1536 =
        vector_similarity_serialized_backend<1536, similarity_function::dot_product, similarity_backend::scalar>;

using vector_similarity_serialized_ssse3_cosine_128 = vector_similarity_serialized_backend<128, similarity_function::cosine, similarity_backend::ssse3>;
using vector_similarity_serialized_ssse3_euclidean_128 = vector_similarity_serialized_backend<128, similarity_function::euclidean, similarity_backend::ssse3>;
using vector_similarity_serialized_ssse3_dot_product_128 =
        vector_similarity_serialized_backend<128, similarity_function::dot_product, similarity_backend::ssse3>;
using vector_similarity_serialized_ssse3_cosine_768 = vector_similarity_serialized_backend<768, similarity_function::cosine, similarity_backend::ssse3>;
using vector_similarity_serialized_ssse3_euclidean_768 = vector_similarity_serialized_backend<768, similarity_function::euclidean, similarity_backend::ssse3>;
using vector_similarity_serialized_ssse3_dot_product_768 =
        vector_similarity_serialized_backend<768, similarity_function::dot_product, similarity_backend::ssse3>;
using vector_similarity_serialized_ssse3_cosine_1536 = vector_similarity_serialized_backend<1536, similarity_function::cosine, similarity_backend::ssse3>;
using vector_similarity_serialized_ssse3_euclidean_1536 =
        vector_similarity_serialized_backend<1536, similarity_function::euclidean, similarity_backend::ssse3>;
using vector_similarity_serialized_ssse3_dot_product_1536 =
        vector_similarity_serialized_backend<1536, similarity_function::dot_product, similarity_backend::ssse3>;

using vector_similarity_execute_cosine_128 = vector_similarity_execute<128, similarity_function::cosine>;
using vector_similarity_execute_euclidean_128 = vector_similarity_execute<128, similarity_function::euclidean>;
using vector_similarity_execute_dot_product_128 = vector_similarity_execute<128, similarity_function::dot_product>;
using vector_similarity_execute_cosine_768 = vector_similarity_execute<768, similarity_function::cosine>;
using vector_similarity_execute_euclidean_768 = vector_similarity_execute<768, similarity_function::euclidean>;
using vector_similarity_execute_dot_product_768 = vector_similarity_execute<768, similarity_function::dot_product>;
using vector_similarity_execute_cosine_1536 = vector_similarity_execute<1536, similarity_function::cosine>;
using vector_similarity_execute_euclidean_1536 = vector_similarity_execute<1536, similarity_function::euclidean>;
using vector_similarity_execute_dot_product_1536 = vector_similarity_execute<1536, similarity_function::dot_product>;

using vector_similarity_execute_scalar_cosine_128 = vector_similarity_execute<128, similarity_function::cosine, similarity_backend::scalar>;
using vector_similarity_execute_scalar_euclidean_128 = vector_similarity_execute<128, similarity_function::euclidean, similarity_backend::scalar>;
using vector_similarity_execute_scalar_dot_product_128 = vector_similarity_execute<128, similarity_function::dot_product, similarity_backend::scalar>;
using vector_similarity_execute_scalar_cosine_768 = vector_similarity_execute<768, similarity_function::cosine, similarity_backend::scalar>;
using vector_similarity_execute_scalar_euclidean_768 = vector_similarity_execute<768, similarity_function::euclidean, similarity_backend::scalar>;
using vector_similarity_execute_scalar_dot_product_768 = vector_similarity_execute<768, similarity_function::dot_product, similarity_backend::scalar>;
using vector_similarity_execute_scalar_cosine_1536 = vector_similarity_execute<1536, similarity_function::cosine, similarity_backend::scalar>;
using vector_similarity_execute_scalar_euclidean_1536 = vector_similarity_execute<1536, similarity_function::euclidean, similarity_backend::scalar>;
using vector_similarity_execute_scalar_dot_product_1536 = vector_similarity_execute<1536, similarity_function::dot_product, similarity_backend::scalar>;

} // namespace

PERF_TEST_F(vector_similarity_128, legacy_cosine) {
    return score_legacy(*this, cql3::functions::detail::compute_cosine_similarity);
}

PERF_TEST_F(vector_similarity_128, serialized_cosine) {
    return score_serialized(*this, cql3::functions::detail::compute_serialized_cosine_similarity);
}

PERF_TEST_F(vector_similarity_serialized_scalar_cosine_128, serialized_scalar) {
    return score_selected_serialized(*this);
}

PERF_TEST_F(vector_similarity_serialized_ssse3_cosine_128, serialized_ssse3) {
    return score_selected_serialized(*this);
}

PERF_TEST_F(vector_similarity_128, legacy_euclidean) {
    return score_legacy(*this, cql3::functions::detail::compute_euclidean_similarity);
}

PERF_TEST_F(vector_similarity_128, serialized_euclidean) {
    return score_serialized(*this, cql3::functions::detail::compute_serialized_euclidean_similarity);
}

PERF_TEST_F(vector_similarity_serialized_scalar_euclidean_128, serialized_scalar) {
    return score_selected_serialized(*this);
}

PERF_TEST_F(vector_similarity_serialized_ssse3_euclidean_128, serialized_ssse3) {
    return score_selected_serialized(*this);
}

PERF_TEST_F(vector_similarity_128, legacy_dot_product) {
    return score_legacy(*this, cql3::functions::detail::compute_dot_product_similarity);
}

PERF_TEST_F(vector_similarity_128, serialized_dot_product) {
    return score_serialized(*this, cql3::functions::detail::compute_serialized_dot_product_similarity);
}

PERF_TEST_F(vector_similarity_serialized_scalar_dot_product_128, serialized_scalar) {
    return score_selected_serialized(*this);
}

PERF_TEST_F(vector_similarity_serialized_ssse3_dot_product_128, serialized_ssse3) {
    return score_selected_serialized(*this);
}

PERF_TEST_F(vector_similarity_768, legacy_cosine) {
    return score_legacy(*this, cql3::functions::detail::compute_cosine_similarity);
}

PERF_TEST_F(vector_similarity_768, serialized_cosine) {
    return score_serialized(*this, cql3::functions::detail::compute_serialized_cosine_similarity);
}

PERF_TEST_F(vector_similarity_serialized_scalar_cosine_768, serialized_scalar) {
    return score_selected_serialized(*this);
}

PERF_TEST_F(vector_similarity_serialized_ssse3_cosine_768, serialized_ssse3) {
    return score_selected_serialized(*this);
}

PERF_TEST_F(vector_similarity_768, legacy_euclidean) {
    return score_legacy(*this, cql3::functions::detail::compute_euclidean_similarity);
}

PERF_TEST_F(vector_similarity_768, serialized_euclidean) {
    return score_serialized(*this, cql3::functions::detail::compute_serialized_euclidean_similarity);
}

PERF_TEST_F(vector_similarity_serialized_scalar_euclidean_768, serialized_scalar) {
    return score_selected_serialized(*this);
}

PERF_TEST_F(vector_similarity_serialized_ssse3_euclidean_768, serialized_ssse3) {
    return score_selected_serialized(*this);
}

PERF_TEST_F(vector_similarity_768, legacy_dot_product) {
    return score_legacy(*this, cql3::functions::detail::compute_dot_product_similarity);
}

PERF_TEST_F(vector_similarity_768, serialized_dot_product) {
    return score_serialized(*this, cql3::functions::detail::compute_serialized_dot_product_similarity);
}

PERF_TEST_F(vector_similarity_serialized_scalar_dot_product_768, serialized_scalar) {
    return score_selected_serialized(*this);
}

PERF_TEST_F(vector_similarity_serialized_ssse3_dot_product_768, serialized_ssse3) {
    return score_selected_serialized(*this);
}

PERF_TEST_F(vector_similarity_1536, legacy_cosine) {
    return score_legacy(*this, cql3::functions::detail::compute_cosine_similarity);
}

PERF_TEST_F(vector_similarity_1536, serialized_cosine) {
    return score_serialized(*this, cql3::functions::detail::compute_serialized_cosine_similarity);
}

PERF_TEST_F(vector_similarity_serialized_scalar_cosine_1536, serialized_scalar) {
    return score_selected_serialized(*this);
}

PERF_TEST_F(vector_similarity_serialized_ssse3_cosine_1536, serialized_ssse3) {
    return score_selected_serialized(*this);
}

PERF_TEST_F(vector_similarity_1536, legacy_euclidean) {
    return score_legacy(*this, cql3::functions::detail::compute_euclidean_similarity);
}

PERF_TEST_F(vector_similarity_1536, serialized_euclidean) {
    return score_serialized(*this, cql3::functions::detail::compute_serialized_euclidean_similarity);
}

PERF_TEST_F(vector_similarity_serialized_scalar_euclidean_1536, serialized_scalar) {
    return score_selected_serialized(*this);
}

PERF_TEST_F(vector_similarity_serialized_ssse3_euclidean_1536, serialized_ssse3) {
    return score_selected_serialized(*this);
}

PERF_TEST_F(vector_similarity_1536, legacy_dot_product) {
    return score_legacy(*this, cql3::functions::detail::compute_dot_product_similarity);
}

PERF_TEST_F(vector_similarity_1536, serialized_dot_product) {
    return score_serialized(*this, cql3::functions::detail::compute_serialized_dot_product_similarity);
}

PERF_TEST_F(vector_similarity_serialized_scalar_dot_product_1536, serialized_scalar) {
    return score_selected_serialized(*this);
}

PERF_TEST_F(vector_similarity_serialized_ssse3_dot_product_1536, serialized_ssse3) {
    return score_selected_serialized(*this);
}

PERF_TEST_F(vector_similarity_tail_130, serialized_cosine) {
    return score_serialized(*this, cql3::functions::detail::compute_serialized_cosine_similarity);
}

PERF_TEST_F(vector_similarity_tail_130, serialized_euclidean) {
    return score_serialized(*this, cql3::functions::detail::compute_serialized_euclidean_similarity);
}

PERF_TEST_F(vector_similarity_tail_130, serialized_dot_product) {
    return score_serialized(*this, cql3::functions::detail::compute_serialized_dot_product_similarity);
}

PERF_TEST_F(vector_similarity_tail_777, serialized_cosine) {
    return score_serialized(*this, cql3::functions::detail::compute_serialized_cosine_similarity);
}

PERF_TEST_F(vector_similarity_tail_777, serialized_euclidean) {
    return score_serialized(*this, cql3::functions::detail::compute_serialized_euclidean_similarity);
}

PERF_TEST_F(vector_similarity_tail_777, serialized_dot_product) {
    return score_serialized(*this, cql3::functions::detail::compute_serialized_dot_product_similarity);
}

PERF_TEST_F(vector_similarity_tail_1543, serialized_cosine) {
    return score_serialized(*this, cql3::functions::detail::compute_serialized_cosine_similarity);
}

PERF_TEST_F(vector_similarity_tail_1543, serialized_euclidean) {
    return score_serialized(*this, cql3::functions::detail::compute_serialized_euclidean_similarity);
}

PERF_TEST_F(vector_similarity_tail_1543, serialized_dot_product) {
    return score_serialized(*this, cql3::functions::detail::compute_serialized_dot_product_similarity);
}

PERF_TEST_F(vector_similarity_execute_cosine_128, execute) {
    return score_execute(*this);
}

PERF_TEST_F(vector_similarity_execute_scalar_cosine_128, execute_scalar) {
    return score_execute(*this);
}

PERF_TEST_F(vector_similarity_execute_euclidean_128, execute) {
    return score_execute(*this);
}

PERF_TEST_F(vector_similarity_execute_scalar_euclidean_128, execute_scalar) {
    return score_execute(*this);
}

PERF_TEST_F(vector_similarity_execute_dot_product_128, execute) {
    return score_execute(*this);
}

PERF_TEST_F(vector_similarity_execute_scalar_dot_product_128, execute_scalar) {
    return score_execute(*this);
}

PERF_TEST_F(vector_similarity_execute_cosine_768, execute) {
    return score_execute(*this);
}

PERF_TEST_F(vector_similarity_execute_scalar_cosine_768, execute_scalar) {
    return score_execute(*this);
}

PERF_TEST_F(vector_similarity_execute_euclidean_768, execute) {
    return score_execute(*this);
}

PERF_TEST_F(vector_similarity_execute_scalar_euclidean_768, execute_scalar) {
    return score_execute(*this);
}

PERF_TEST_F(vector_similarity_execute_dot_product_768, execute) {
    return score_execute(*this);
}

PERF_TEST_F(vector_similarity_execute_scalar_dot_product_768, execute_scalar) {
    return score_execute(*this);
}

PERF_TEST_F(vector_similarity_execute_cosine_1536, execute) {
    return score_execute(*this);
}

PERF_TEST_F(vector_similarity_execute_scalar_cosine_1536, execute_scalar) {
    return score_execute(*this);
}

PERF_TEST_F(vector_similarity_execute_euclidean_1536, execute) {
    return score_execute(*this);
}

PERF_TEST_F(vector_similarity_execute_scalar_euclidean_1536, execute_scalar) {
    return score_execute(*this);
}

PERF_TEST_F(vector_similarity_execute_dot_product_1536, execute) {
    return score_execute(*this);
}

PERF_TEST_F(vector_similarity_execute_scalar_dot_product_1536, execute_scalar) {
    return score_execute(*this);
}
