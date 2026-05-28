/*
 * Copyright (C) 2025-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

#pragma once

#include "native_scalar_function.hh"
#include "cql3/assignment_testable.hh"
#include "cql3/functions/function_name.hh"
#include <cstdint>
#include <span>
#include <string_view>

namespace db {
enum class simd_optimization_mode : uint8_t;
}

namespace cql3 {
namespace functions {

static const function_name SIMILARITY_COSINE_FUNCTION_NAME = function_name::native_function("similarity_cosine");
static const function_name SIMILARITY_EUCLIDEAN_FUNCTION_NAME = function_name::native_function("similarity_euclidean");
static const function_name SIMILARITY_DOT_PRODUCT_FUNCTION_NAME = function_name::native_function("similarity_dot_product");

using similarity_function_t = float (*)(std::span<const float>, std::span<const float>);
using serialized_similarity_function_t = float (*)(const bytes_opt&, const bytes_opt&, vector_dimension_t);
extern thread_local const std::unordered_map<function_name, similarity_function_t> SIMILARITY_FUNCTIONS;

std::vector<data_type> retrieve_vector_arg_types(const function_name& name, const std::vector<shared_ptr<assignment_testable>>& provided_args);

class vector_similarity_fct : public native_scalar_function {
public:
    vector_similarity_fct(const sstring& name, const std::vector<data_type>& arg_types, db::simd_optimization_mode backend);

    virtual bytes_opt execute(std::span<const bytes_opt> parameters) override;

private:
    serialized_similarity_function_t _similarity_func;
    vector_dimension_t _dimension;
};

namespace detail {

// Extract float vector directly from serialized bytes, bypassing data_value overhead.
// This is an internal API exposed for testing purposes.
// Vector<float, N> wire format: N floats as big-endian uint32_t values, 4 bytes each.
std::vector<float> extract_float_vector(const bytes_opt& param, vector_dimension_t dimension);

float compute_cosine_similarity(std::span<const float> v1, std::span<const float> v2);
float compute_euclidean_similarity(std::span<const float> v1, std::span<const float> v2);
float compute_dot_product_similarity(std::span<const float> v1, std::span<const float> v2);

float compute_serialized_cosine_similarity(const bytes_opt& v1, const bytes_opt& v2, vector_dimension_t dimension);
float compute_serialized_euclidean_similarity(const bytes_opt& v1, const bytes_opt& v2, vector_dimension_t dimension);
float compute_serialized_dot_product_similarity(const bytes_opt& v1, const bytes_opt& v2, vector_dimension_t dimension);
sstring select_serialized_similarity_backend(db::simd_optimization_mode backend);
serialized_similarity_function_t select_serialized_similarity_function(const function_name& name, db::simd_optimization_mode backend);

} // namespace detail

} // namespace functions
} // namespace cql3
