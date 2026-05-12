/*
 * Copyright (C) 2025-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

#include "vector_similarity_fcts.hh"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <string_view>

#include <seastar/core/byteorder.hh>

#ifdef __linux__
#include <sched.h>
#endif

#ifdef __x86_64__
#include <x86intrin.h>
#define arch_target(name) [[gnu::target(name)]]
#else
#define arch_target(name)
#endif

#ifdef __aarch64__
#include <arm_neon.h>
#ifdef __linux__
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif
#endif

#include "db/config.hh"
#include "exceptions/exceptions.hh"
#include "types/types.hh"
#include "types/vector.hh"

namespace cql3 {
namespace functions {

namespace detail {

namespace {

enum class similarity_operation {
    cosine,
    euclidean,
    dot_product,
};

bytes_view require_float_vector(const bytes_opt& param, vector_dimension_t dimension) {
    if (!param) {
        throw exceptions::invalid_request_exception("Cannot extract float vector from null parameter");
    }

    const size_t expected_size = dimension * sizeof(float);
    if (param->size() != expected_size) {
        throw exceptions::invalid_request_exception(
            fmt::format("Invalid vector size: expected {} bytes for {} floats, got {} bytes",
                       expected_size, dimension, param->size()));
    }

    return bytes_view(param->data(), param->size());
}

float read_be_float(const int8_t* p) noexcept {
    const uint32_t bits =
            (uint32_t(uint8_t(p[0])) << 24) |
            (uint32_t(uint8_t(p[1])) << 16) |
            (uint32_t(uint8_t(p[2])) << 8) |
            uint32_t(uint8_t(p[3]));
    return std::bit_cast<float>(bits);
}

float finish_cosine_similarity(float dot_product, float squared_norm_a, float squared_norm_b) {
    if (squared_norm_a == 0 || squared_norm_b == 0) {
        return std::numeric_limits<float>::quiet_NaN();
    }

    // The cosine similarity is in the range [-1, 1].
    // It is mapped to a similarity score in the range [0, 1] (-1 -> 0, 1 -> 1)
    // for consistency with other similarity functions.
    return (1.0f + (dot_product / std::sqrt(squared_norm_a * squared_norm_b))) / 2.0f;
}

float finish_euclidean_similarity(float squared_distance) {
    // The squared Euclidean (L2) distance is of range [0, inf).
    // It is mapped to a similarity score in the range (0, 1] (0 -> 1, inf -> 0)
    // for consistency with other similarity functions.
    return 1.0f / (1.0f + squared_distance);
}

float finish_dot_product_similarity(float dot_product) {
    // The dot product is in the range [-1, 1] for L2-normalized vectors.
    // It is mapped to a similarity score in the range [0, 1] (-1 -> 0, 1 -> 1)
    // for consistency with other similarity functions.
    return (1.0f + dot_product) / 2.0f;
}

template <similarity_operation Operation>
float compute_serialized_scalar(bytes_view v1, bytes_view v2, vector_dimension_t dimension) {
    #pragma clang fp contract(fast) reassociate(on) // Allow the compiler to optimize the loop.
    float dot_product = 0.0f;
    float squared_norm_a = 0.0f;
    float squared_norm_b = 0.0f;
    float squared_distance = 0.0f;

    for (size_t i = 0; i < dimension; ++i) {
        const float a = read_be_float(v1.data() + i * sizeof(float));
        const float b = read_be_float(v2.data() + i * sizeof(float));

        if constexpr (Operation == similarity_operation::cosine) {
            dot_product += a * b;
            squared_norm_a += a * a;
            squared_norm_b += b * b;
        } else if constexpr (Operation == similarity_operation::euclidean) {
            const float diff = a - b;
            squared_distance += diff * diff;
        } else {
            dot_product += a * b;
        }
    }

    if constexpr (Operation == similarity_operation::cosine) {
        return finish_cosine_similarity(dot_product, squared_norm_a, squared_norm_b);
    } else if constexpr (Operation == similarity_operation::euclidean) {
        return finish_euclidean_similarity(squared_distance);
    } else {
        return finish_dot_product_similarity(dot_product);
    }
}

template <similarity_operation Operation, typename Compute>
float compute_serialized_with(const bytes_opt& v1, const bytes_opt& v2, vector_dimension_t dimension, Compute compute) {
    return compute(
            require_float_vector(v1, dimension),
            require_float_vector(v2, dimension),
            dimension);
}

float compute_serialized_scalar_cosine(const bytes_opt& v1, const bytes_opt& v2, vector_dimension_t dimension) {
    return compute_serialized_with<similarity_operation::cosine>(v1, v2, dimension, compute_serialized_scalar<similarity_operation::cosine>);
}

float compute_serialized_scalar_euclidean(const bytes_opt& v1, const bytes_opt& v2, vector_dimension_t dimension) {
    return compute_serialized_with<similarity_operation::euclidean>(v1, v2, dimension, compute_serialized_scalar<similarity_operation::euclidean>);
}

float compute_serialized_scalar_dot_product(const bytes_opt& v1, const bytes_opt& v2, vector_dimension_t dimension) {
    return compute_serialized_with<similarity_operation::dot_product>(v1, v2, dimension, compute_serialized_scalar<similarity_operation::dot_product>);
}

struct serialized_similarity_backend {
    std::string_view name;
    serialized_similarity_function_t cosine;
    serialized_similarity_function_t euclidean;
    serialized_similarity_function_t dot_product;
};

const serialized_similarity_backend scalar_backend = {
        .name = "scalar",
        .cosine = compute_serialized_scalar_cosine,
        .euclidean = compute_serialized_scalar_euclidean,
        .dot_product = compute_serialized_scalar_dot_product,
};

bool current_thread_has_single_cpu_affinity() {
#ifdef __linux__
    cpu_set_t cpus;
    if (sched_getaffinity(0, sizeof(cpus), &cpus) != 0) {
        return false;
    }
    size_t count = 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &cpus)) {
            ++count;
        }
    }
    return count == 1;
#else
    return false;
#endif
}

[[noreturn]] void throw_backend_unavailable(db::simd_optimization_mode backend, std::string_view reason) {
    throw exceptions::configuration_exception(fmt::format(
            "simd_optimization_options.vector_similarity={} is not available: {}",
            db::config::simd_optimization_mode_name(backend), reason));
}

#ifdef __x86_64__

struct x86_simd_support {
    bool ssse3;
    bool avx2_fma;
    bool avx512_float;
    bool avx512_vnni;
};

x86_simd_support detect_x86_simd_support() {
    __builtin_cpu_init();
    return {
        .ssse3 = __builtin_cpu_supports("ssse3"),
        .avx2_fma = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"),
        .avx512_float = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw"),
        .avx512_vnni = __builtin_cpu_supports("avx512vnni"),
    };
}

const x86_simd_support& get_x86_simd_support() {
    static const auto support = detect_x86_simd_support();
    return support;
}

arch_target("ssse3") __m128 load_be_floats_ssse3(const int8_t* p) {
    alignas(16) static constexpr int8_t byte_swap_mask_data[16] = {
        3, 2, 1, 0,
        7, 6, 5, 4,
        11, 10, 9, 8,
        15, 14, 13, 12,
    };
    const auto bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const auto mask = _mm_load_si128(reinterpret_cast<const __m128i*>(byte_swap_mask_data));
    return _mm_castsi128_ps(_mm_shuffle_epi8(bytes, mask));
}

arch_target("ssse3") float horizontal_sum_ssse3(__m128 values) {
    values = _mm_hadd_ps(values, values);
    values = _mm_hadd_ps(values, values);
    return _mm_cvtss_f32(values);
}

template <similarity_operation Operation>
arch_target("ssse3") float compute_serialized_ssse3(bytes_view v1, bytes_view v2, vector_dimension_t dimension) {
    __m128 dot_product = _mm_setzero_ps();
    __m128 squared_norm_a = _mm_setzero_ps();
    __m128 squared_norm_b = _mm_setzero_ps();
    __m128 squared_distance = _mm_setzero_ps();

    size_t i = 0;
    for (; i + 4 <= dimension; i += 4) {
        const auto a = load_be_floats_ssse3(v1.data() + i * sizeof(float));
        const auto b = load_be_floats_ssse3(v2.data() + i * sizeof(float));

        if constexpr (Operation == similarity_operation::cosine) {
            dot_product = _mm_add_ps(dot_product, _mm_mul_ps(a, b));
            squared_norm_a = _mm_add_ps(squared_norm_a, _mm_mul_ps(a, a));
            squared_norm_b = _mm_add_ps(squared_norm_b, _mm_mul_ps(b, b));
        } else if constexpr (Operation == similarity_operation::euclidean) {
            const auto diff = _mm_sub_ps(a, b);
            squared_distance = _mm_add_ps(squared_distance, _mm_mul_ps(diff, diff));
        } else {
            dot_product = _mm_add_ps(dot_product, _mm_mul_ps(a, b));
        }
    }

    float dot_product_tail = 0.0f;
    float squared_norm_a_tail = 0.0f;
    float squared_norm_b_tail = 0.0f;
    float squared_distance_tail = 0.0f;
    for (; i < dimension; ++i) {
        const float a = read_be_float(v1.data() + i * sizeof(float));
        const float b = read_be_float(v2.data() + i * sizeof(float));

        if constexpr (Operation == similarity_operation::cosine) {
            dot_product_tail += a * b;
            squared_norm_a_tail += a * a;
            squared_norm_b_tail += b * b;
        } else if constexpr (Operation == similarity_operation::euclidean) {
            const float diff = a - b;
            squared_distance_tail += diff * diff;
        } else {
            dot_product_tail += a * b;
        }
    }

    if constexpr (Operation == similarity_operation::cosine) {
        return finish_cosine_similarity(
                horizontal_sum_ssse3(dot_product) + dot_product_tail,
                horizontal_sum_ssse3(squared_norm_a) + squared_norm_a_tail,
                horizontal_sum_ssse3(squared_norm_b) + squared_norm_b_tail);
    } else if constexpr (Operation == similarity_operation::euclidean) {
        return finish_euclidean_similarity(horizontal_sum_ssse3(squared_distance) + squared_distance_tail);
    } else {
        return finish_dot_product_similarity(horizontal_sum_ssse3(dot_product) + dot_product_tail);
    }
}

arch_target("ssse3") float compute_serialized_ssse3_euclidean_four_accumulators(
        bytes_view v1, bytes_view v2, vector_dimension_t dimension) {
    __m128 squared_distance_0 = _mm_setzero_ps();
    __m128 squared_distance_1 = _mm_setzero_ps();
    __m128 squared_distance_2 = _mm_setzero_ps();
    __m128 squared_distance_3 = _mm_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= dimension; i += 16) {
        const auto a0 = load_be_floats_ssse3(v1.data() + i * sizeof(float));
        const auto b0 = load_be_floats_ssse3(v2.data() + i * sizeof(float));
        const auto a1 = load_be_floats_ssse3(v1.data() + (i + 4) * sizeof(float));
        const auto b1 = load_be_floats_ssse3(v2.data() + (i + 4) * sizeof(float));
        const auto a2 = load_be_floats_ssse3(v1.data() + (i + 8) * sizeof(float));
        const auto b2 = load_be_floats_ssse3(v2.data() + (i + 8) * sizeof(float));
        const auto a3 = load_be_floats_ssse3(v1.data() + (i + 12) * sizeof(float));
        const auto b3 = load_be_floats_ssse3(v2.data() + (i + 12) * sizeof(float));

        const auto diff0 = _mm_sub_ps(a0, b0);
        const auto diff1 = _mm_sub_ps(a1, b1);
        const auto diff2 = _mm_sub_ps(a2, b2);
        const auto diff3 = _mm_sub_ps(a3, b3);
        squared_distance_0 = _mm_add_ps(squared_distance_0, _mm_mul_ps(diff0, diff0));
        squared_distance_1 = _mm_add_ps(squared_distance_1, _mm_mul_ps(diff1, diff1));
        squared_distance_2 = _mm_add_ps(squared_distance_2, _mm_mul_ps(diff2, diff2));
        squared_distance_3 = _mm_add_ps(squared_distance_3, _mm_mul_ps(diff3, diff3));
    }

    for (; i + 8 <= dimension; i += 8) {
        const auto a0 = load_be_floats_ssse3(v1.data() + i * sizeof(float));
        const auto b0 = load_be_floats_ssse3(v2.data() + i * sizeof(float));
        const auto a1 = load_be_floats_ssse3(v1.data() + (i + 4) * sizeof(float));
        const auto b1 = load_be_floats_ssse3(v2.data() + (i + 4) * sizeof(float));

        const auto diff0 = _mm_sub_ps(a0, b0);
        const auto diff1 = _mm_sub_ps(a1, b1);
        squared_distance_0 = _mm_add_ps(squared_distance_0, _mm_mul_ps(diff0, diff0));
        squared_distance_1 = _mm_add_ps(squared_distance_1, _mm_mul_ps(diff1, diff1));
    }

    for (; i + 4 <= dimension; i += 4) {
        const auto a = load_be_floats_ssse3(v1.data() + i * sizeof(float));
        const auto b = load_be_floats_ssse3(v2.data() + i * sizeof(float));

        const auto diff = _mm_sub_ps(a, b);
        squared_distance_0 = _mm_add_ps(squared_distance_0, _mm_mul_ps(diff, diff));
    }

    float squared_distance_tail = 0.0f;
    for (; i < dimension; ++i) {
        const float a = read_be_float(v1.data() + i * sizeof(float));
        const float b = read_be_float(v2.data() + i * sizeof(float));

        const float diff = a - b;
        squared_distance_tail += diff * diff;
    }

    const auto squared_distance = _mm_add_ps(
            _mm_add_ps(squared_distance_0, squared_distance_1),
            _mm_add_ps(squared_distance_2, squared_distance_3));
    return finish_euclidean_similarity(horizontal_sum_ssse3(squared_distance) + squared_distance_tail);
}

arch_target("ssse3") float compute_serialized_ssse3_dot_product_four_accumulators(
        bytes_view v1, bytes_view v2, vector_dimension_t dimension) {
    __m128 dot_product_0 = _mm_setzero_ps();
    __m128 dot_product_1 = _mm_setzero_ps();
    __m128 dot_product_2 = _mm_setzero_ps();
    __m128 dot_product_3 = _mm_setzero_ps();

    size_t i = 0;
    for (; i + 16 <= dimension; i += 16) {
        const auto a0 = load_be_floats_ssse3(v1.data() + i * sizeof(float));
        const auto b0 = load_be_floats_ssse3(v2.data() + i * sizeof(float));
        const auto a1 = load_be_floats_ssse3(v1.data() + (i + 4) * sizeof(float));
        const auto b1 = load_be_floats_ssse3(v2.data() + (i + 4) * sizeof(float));
        const auto a2 = load_be_floats_ssse3(v1.data() + (i + 8) * sizeof(float));
        const auto b2 = load_be_floats_ssse3(v2.data() + (i + 8) * sizeof(float));
        const auto a3 = load_be_floats_ssse3(v1.data() + (i + 12) * sizeof(float));
        const auto b3 = load_be_floats_ssse3(v2.data() + (i + 12) * sizeof(float));

        dot_product_0 = _mm_add_ps(dot_product_0, _mm_mul_ps(a0, b0));
        dot_product_1 = _mm_add_ps(dot_product_1, _mm_mul_ps(a1, b1));
        dot_product_2 = _mm_add_ps(dot_product_2, _mm_mul_ps(a2, b2));
        dot_product_3 = _mm_add_ps(dot_product_3, _mm_mul_ps(a3, b3));
    }

    for (; i + 8 <= dimension; i += 8) {
        const auto a0 = load_be_floats_ssse3(v1.data() + i * sizeof(float));
        const auto b0 = load_be_floats_ssse3(v2.data() + i * sizeof(float));
        const auto a1 = load_be_floats_ssse3(v1.data() + (i + 4) * sizeof(float));
        const auto b1 = load_be_floats_ssse3(v2.data() + (i + 4) * sizeof(float));

        dot_product_0 = _mm_add_ps(dot_product_0, _mm_mul_ps(a0, b0));
        dot_product_1 = _mm_add_ps(dot_product_1, _mm_mul_ps(a1, b1));
    }

    for (; i + 4 <= dimension; i += 4) {
        const auto a = load_be_floats_ssse3(v1.data() + i * sizeof(float));
        const auto b = load_be_floats_ssse3(v2.data() + i * sizeof(float));

        dot_product_0 = _mm_add_ps(dot_product_0, _mm_mul_ps(a, b));
    }

    float dot_product_tail = 0.0f;
    for (; i < dimension; ++i) {
        const float a = read_be_float(v1.data() + i * sizeof(float));
        const float b = read_be_float(v2.data() + i * sizeof(float));

        dot_product_tail += a * b;
    }

    const auto dot_product = _mm_add_ps(
            _mm_add_ps(dot_product_0, dot_product_1),
            _mm_add_ps(dot_product_2, dot_product_3));
    return finish_dot_product_similarity(horizontal_sum_ssse3(dot_product) + dot_product_tail);
}

float compute_serialized_ssse3_cosine(const bytes_opt& v1, const bytes_opt& v2, vector_dimension_t dimension) {
    return compute_serialized_with<similarity_operation::cosine>(v1, v2, dimension, compute_serialized_ssse3<similarity_operation::cosine>);
}

float compute_serialized_ssse3_euclidean(const bytes_opt& v1, const bytes_opt& v2, vector_dimension_t dimension) {
    return compute_serialized_with<similarity_operation::euclidean>(
            v1, v2, dimension, compute_serialized_ssse3_euclidean_four_accumulators);
}

float compute_serialized_ssse3_dot_product(const bytes_opt& v1, const bytes_opt& v2, vector_dimension_t dimension) {
    return compute_serialized_with<similarity_operation::dot_product>(
            v1, v2, dimension, compute_serialized_ssse3_dot_product_four_accumulators);
}

const serialized_similarity_backend ssse3_backend = {
        .name = "ssse3",
        .cosine = compute_serialized_ssse3_cosine,
        .euclidean = compute_serialized_ssse3_euclidean,
        .dot_product = compute_serialized_ssse3_dot_product,
};

arch_target("avx2,fma,sse3") __m256i byte_swap_mask_avx2() {
    alignas(32) static constexpr int8_t byte_swap_mask_data[32] = {
        3, 2, 1, 0,
        7, 6, 5, 4,
        11, 10, 9, 8,
        15, 14, 13, 12,
        3, 2, 1, 0,
        7, 6, 5, 4,
        11, 10, 9, 8,
        15, 14, 13, 12,
    };
    return _mm256_load_si256(reinterpret_cast<const __m256i*>(byte_swap_mask_data));
}

arch_target("avx2,fma,sse3") __m256 load_be_floats_avx2(const int8_t* p, __m256i byte_swap_mask) {
    const auto bytes = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
    return _mm256_castsi256_ps(_mm256_shuffle_epi8(bytes, byte_swap_mask));
}

arch_target("avx2,fma,sse3") float horizontal_sum_avx2(__m256 values) {
    const __m128 low = _mm256_castps256_ps128(values);
    const __m128 high = _mm256_extractf128_ps(values, 1);
    __m128 sum = _mm_add_ps(low, high);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

template <similarity_operation Operation>
arch_target("avx2,fma,sse3") float compute_serialized_avx2(bytes_view v1, bytes_view v2, vector_dimension_t dimension) {
    __m256 dot_product_0 = _mm256_setzero_ps();
    __m256 dot_product_1 = _mm256_setzero_ps();
    __m256 squared_norm_a_0 = _mm256_setzero_ps();
    __m256 squared_norm_a_1 = _mm256_setzero_ps();
    __m256 squared_norm_b_0 = _mm256_setzero_ps();
    __m256 squared_norm_b_1 = _mm256_setzero_ps();
    __m256 squared_distance_0 = _mm256_setzero_ps();
    __m256 squared_distance_1 = _mm256_setzero_ps();
    const auto byte_swap_mask = byte_swap_mask_avx2();

    size_t i = 0;
    for (; i + 16 <= dimension; i += 16) {
        const auto a0 = load_be_floats_avx2(v1.data() + i * sizeof(float), byte_swap_mask);
        const auto b0 = load_be_floats_avx2(v2.data() + i * sizeof(float), byte_swap_mask);
        const auto a1 = load_be_floats_avx2(v1.data() + (i + 8) * sizeof(float), byte_swap_mask);
        const auto b1 = load_be_floats_avx2(v2.data() + (i + 8) * sizeof(float), byte_swap_mask);

        if constexpr (Operation == similarity_operation::cosine) {
            dot_product_0 = _mm256_fmadd_ps(a0, b0, dot_product_0);
            dot_product_1 = _mm256_fmadd_ps(a1, b1, dot_product_1);
            squared_norm_a_0 = _mm256_fmadd_ps(a0, a0, squared_norm_a_0);
            squared_norm_a_1 = _mm256_fmadd_ps(a1, a1, squared_norm_a_1);
            squared_norm_b_0 = _mm256_fmadd_ps(b0, b0, squared_norm_b_0);
            squared_norm_b_1 = _mm256_fmadd_ps(b1, b1, squared_norm_b_1);
        } else if constexpr (Operation == similarity_operation::euclidean) {
            const auto diff0 = _mm256_sub_ps(a0, b0);
            const auto diff1 = _mm256_sub_ps(a1, b1);
            squared_distance_0 = _mm256_fmadd_ps(diff0, diff0, squared_distance_0);
            squared_distance_1 = _mm256_fmadd_ps(diff1, diff1, squared_distance_1);
        } else {
            dot_product_0 = _mm256_fmadd_ps(a0, b0, dot_product_0);
            dot_product_1 = _mm256_fmadd_ps(a1, b1, dot_product_1);
        }
    }

    for (; i + 8 <= dimension; i += 8) {
        const auto a = load_be_floats_avx2(v1.data() + i * sizeof(float), byte_swap_mask);
        const auto b = load_be_floats_avx2(v2.data() + i * sizeof(float), byte_swap_mask);

        if constexpr (Operation == similarity_operation::cosine) {
            dot_product_0 = _mm256_fmadd_ps(a, b, dot_product_0);
            squared_norm_a_0 = _mm256_fmadd_ps(a, a, squared_norm_a_0);
            squared_norm_b_0 = _mm256_fmadd_ps(b, b, squared_norm_b_0);
        } else if constexpr (Operation == similarity_operation::euclidean) {
            const auto diff = _mm256_sub_ps(a, b);
            squared_distance_0 = _mm256_fmadd_ps(diff, diff, squared_distance_0);
        } else {
            dot_product_0 = _mm256_fmadd_ps(a, b, dot_product_0);
        }
    }

    float dot_product_tail = 0.0f;
    float squared_norm_a_tail = 0.0f;
    float squared_norm_b_tail = 0.0f;
    float squared_distance_tail = 0.0f;
    for (; i < dimension; ++i) {
        const float a = read_be_float(v1.data() + i * sizeof(float));
        const float b = read_be_float(v2.data() + i * sizeof(float));

        if constexpr (Operation == similarity_operation::cosine) {
            dot_product_tail += a * b;
            squared_norm_a_tail += a * a;
            squared_norm_b_tail += b * b;
        } else if constexpr (Operation == similarity_operation::euclidean) {
            const float diff = a - b;
            squared_distance_tail += diff * diff;
        } else {
            dot_product_tail += a * b;
        }
    }

    if constexpr (Operation == similarity_operation::cosine) {
        return finish_cosine_similarity(
                horizontal_sum_avx2(_mm256_add_ps(dot_product_0, dot_product_1)) + dot_product_tail,
                horizontal_sum_avx2(_mm256_add_ps(squared_norm_a_0, squared_norm_a_1)) + squared_norm_a_tail,
                horizontal_sum_avx2(_mm256_add_ps(squared_norm_b_0, squared_norm_b_1)) + squared_norm_b_tail);
    } else if constexpr (Operation == similarity_operation::euclidean) {
        return finish_euclidean_similarity(
                horizontal_sum_avx2(_mm256_add_ps(squared_distance_0, squared_distance_1)) + squared_distance_tail);
    } else {
        return finish_dot_product_similarity(horizontal_sum_avx2(_mm256_add_ps(dot_product_0, dot_product_1)) + dot_product_tail);
    }
}

arch_target("avx2,fma,sse3") float compute_serialized_avx2_euclidean_four_accumulators(
        bytes_view v1, bytes_view v2, vector_dimension_t dimension) {
    __m256 squared_distance_0 = _mm256_setzero_ps();
    __m256 squared_distance_1 = _mm256_setzero_ps();
    __m256 squared_distance_2 = _mm256_setzero_ps();
    __m256 squared_distance_3 = _mm256_setzero_ps();
    const auto byte_swap_mask = byte_swap_mask_avx2();

    size_t i = 0;
    for (; i + 32 <= dimension; i += 32) {
        const auto a0 = load_be_floats_avx2(v1.data() + i * sizeof(float), byte_swap_mask);
        const auto b0 = load_be_floats_avx2(v2.data() + i * sizeof(float), byte_swap_mask);
        const auto a1 = load_be_floats_avx2(v1.data() + (i + 8) * sizeof(float), byte_swap_mask);
        const auto b1 = load_be_floats_avx2(v2.data() + (i + 8) * sizeof(float), byte_swap_mask);
        const auto a2 = load_be_floats_avx2(v1.data() + (i + 16) * sizeof(float), byte_swap_mask);
        const auto b2 = load_be_floats_avx2(v2.data() + (i + 16) * sizeof(float), byte_swap_mask);
        const auto a3 = load_be_floats_avx2(v1.data() + (i + 24) * sizeof(float), byte_swap_mask);
        const auto b3 = load_be_floats_avx2(v2.data() + (i + 24) * sizeof(float), byte_swap_mask);

        const auto diff0 = _mm256_sub_ps(a0, b0);
        const auto diff1 = _mm256_sub_ps(a1, b1);
        const auto diff2 = _mm256_sub_ps(a2, b2);
        const auto diff3 = _mm256_sub_ps(a3, b3);
        squared_distance_0 = _mm256_fmadd_ps(diff0, diff0, squared_distance_0);
        squared_distance_1 = _mm256_fmadd_ps(diff1, diff1, squared_distance_1);
        squared_distance_2 = _mm256_fmadd_ps(diff2, diff2, squared_distance_2);
        squared_distance_3 = _mm256_fmadd_ps(diff3, diff3, squared_distance_3);
    }

    for (; i + 16 <= dimension; i += 16) {
        const auto a0 = load_be_floats_avx2(v1.data() + i * sizeof(float), byte_swap_mask);
        const auto b0 = load_be_floats_avx2(v2.data() + i * sizeof(float), byte_swap_mask);
        const auto a1 = load_be_floats_avx2(v1.data() + (i + 8) * sizeof(float), byte_swap_mask);
        const auto b1 = load_be_floats_avx2(v2.data() + (i + 8) * sizeof(float), byte_swap_mask);

        const auto diff0 = _mm256_sub_ps(a0, b0);
        const auto diff1 = _mm256_sub_ps(a1, b1);
        squared_distance_0 = _mm256_fmadd_ps(diff0, diff0, squared_distance_0);
        squared_distance_1 = _mm256_fmadd_ps(diff1, diff1, squared_distance_1);
    }

    for (; i + 8 <= dimension; i += 8) {
        const auto a = load_be_floats_avx2(v1.data() + i * sizeof(float), byte_swap_mask);
        const auto b = load_be_floats_avx2(v2.data() + i * sizeof(float), byte_swap_mask);

        const auto diff = _mm256_sub_ps(a, b);
        squared_distance_0 = _mm256_fmadd_ps(diff, diff, squared_distance_0);
    }

    float squared_distance_tail = 0.0f;
    for (; i < dimension; ++i) {
        const float a = read_be_float(v1.data() + i * sizeof(float));
        const float b = read_be_float(v2.data() + i * sizeof(float));

        const float diff = a - b;
        squared_distance_tail += diff * diff;
    }

    const auto squared_distance = _mm256_add_ps(
            _mm256_add_ps(squared_distance_0, squared_distance_1),
            _mm256_add_ps(squared_distance_2, squared_distance_3));
    return finish_euclidean_similarity(horizontal_sum_avx2(squared_distance) + squared_distance_tail);
}

arch_target("avx2,fma,sse3") float compute_serialized_avx2_dot_product_four_accumulators(
        bytes_view v1, bytes_view v2, vector_dimension_t dimension) {
    __m256 dot_product_0 = _mm256_setzero_ps();
    __m256 dot_product_1 = _mm256_setzero_ps();
    __m256 dot_product_2 = _mm256_setzero_ps();
    __m256 dot_product_3 = _mm256_setzero_ps();
    const auto byte_swap_mask = byte_swap_mask_avx2();

    size_t i = 0;
    for (; i + 32 <= dimension; i += 32) {
        const auto a0 = load_be_floats_avx2(v1.data() + i * sizeof(float), byte_swap_mask);
        const auto b0 = load_be_floats_avx2(v2.data() + i * sizeof(float), byte_swap_mask);
        const auto a1 = load_be_floats_avx2(v1.data() + (i + 8) * sizeof(float), byte_swap_mask);
        const auto b1 = load_be_floats_avx2(v2.data() + (i + 8) * sizeof(float), byte_swap_mask);
        const auto a2 = load_be_floats_avx2(v1.data() + (i + 16) * sizeof(float), byte_swap_mask);
        const auto b2 = load_be_floats_avx2(v2.data() + (i + 16) * sizeof(float), byte_swap_mask);
        const auto a3 = load_be_floats_avx2(v1.data() + (i + 24) * sizeof(float), byte_swap_mask);
        const auto b3 = load_be_floats_avx2(v2.data() + (i + 24) * sizeof(float), byte_swap_mask);

        dot_product_0 = _mm256_fmadd_ps(a0, b0, dot_product_0);
        dot_product_1 = _mm256_fmadd_ps(a1, b1, dot_product_1);
        dot_product_2 = _mm256_fmadd_ps(a2, b2, dot_product_2);
        dot_product_3 = _mm256_fmadd_ps(a3, b3, dot_product_3);
    }

    for (; i + 16 <= dimension; i += 16) {
        const auto a0 = load_be_floats_avx2(v1.data() + i * sizeof(float), byte_swap_mask);
        const auto b0 = load_be_floats_avx2(v2.data() + i * sizeof(float), byte_swap_mask);
        const auto a1 = load_be_floats_avx2(v1.data() + (i + 8) * sizeof(float), byte_swap_mask);
        const auto b1 = load_be_floats_avx2(v2.data() + (i + 8) * sizeof(float), byte_swap_mask);

        dot_product_0 = _mm256_fmadd_ps(a0, b0, dot_product_0);
        dot_product_1 = _mm256_fmadd_ps(a1, b1, dot_product_1);
    }

    for (; i + 8 <= dimension; i += 8) {
        const auto a = load_be_floats_avx2(v1.data() + i * sizeof(float), byte_swap_mask);
        const auto b = load_be_floats_avx2(v2.data() + i * sizeof(float), byte_swap_mask);

        dot_product_0 = _mm256_fmadd_ps(a, b, dot_product_0);
    }

    float dot_product_tail = 0.0f;
    for (; i < dimension; ++i) {
        const float a = read_be_float(v1.data() + i * sizeof(float));
        const float b = read_be_float(v2.data() + i * sizeof(float));

        dot_product_tail += a * b;
    }

    const auto dot_product = _mm256_add_ps(
            _mm256_add_ps(dot_product_0, dot_product_1),
            _mm256_add_ps(dot_product_2, dot_product_3));
    return finish_dot_product_similarity(horizontal_sum_avx2(dot_product) + dot_product_tail);
}

float compute_serialized_avx2_cosine(const bytes_opt& v1, const bytes_opt& v2, vector_dimension_t dimension) {
    return compute_serialized_with<similarity_operation::cosine>(v1, v2, dimension, compute_serialized_avx2<similarity_operation::cosine>);
}

float compute_serialized_avx2_euclidean(const bytes_opt& v1, const bytes_opt& v2, vector_dimension_t dimension) {
    return compute_serialized_with<similarity_operation::euclidean>(
            v1, v2, dimension, compute_serialized_avx2_euclidean_four_accumulators);
}

float compute_serialized_avx2_dot_product(const bytes_opt& v1, const bytes_opt& v2, vector_dimension_t dimension) {
    return compute_serialized_with<similarity_operation::dot_product>(
            v1, v2, dimension, compute_serialized_avx2_dot_product_four_accumulators);
}

const serialized_similarity_backend avx2_backend = {
        .name = "avx2",
        .cosine = compute_serialized_avx2_cosine,
        .euclidean = compute_serialized_avx2_euclidean,
        .dot_product = compute_serialized_avx2_dot_product,
};

arch_target("avx512f,avx512bw") __m512i byte_swap_mask_avx512() {
    alignas(64) static constexpr int8_t byte_swap_mask_data[64] = {
        3, 2, 1, 0,
        7, 6, 5, 4,
        11, 10, 9, 8,
        15, 14, 13, 12,
        3, 2, 1, 0,
        7, 6, 5, 4,
        11, 10, 9, 8,
        15, 14, 13, 12,
        3, 2, 1, 0,
        7, 6, 5, 4,
        11, 10, 9, 8,
        15, 14, 13, 12,
        3, 2, 1, 0,
        7, 6, 5, 4,
        11, 10, 9, 8,
        15, 14, 13, 12,
    };
    return _mm512_load_si512(reinterpret_cast<const __m512i*>(byte_swap_mask_data));
}

arch_target("avx512f,avx512bw") __m512 load_be_floats_avx512(const int8_t* p, __m512i byte_swap_mask) {
    const auto bytes = _mm512_loadu_si512(reinterpret_cast<const void*>(p));
    return _mm512_castsi512_ps(_mm512_shuffle_epi8(bytes, byte_swap_mask));
}

arch_target("avx512f,avx512bw") __m512 load_be_floats_avx512_masked(const int8_t* p, __m512i byte_swap_mask, size_t lanes) {
    const auto mask = static_cast<__mmask16>((uint32_t{1} << lanes) - 1);
    const auto bytes = _mm512_maskz_loadu_epi32(mask, reinterpret_cast<const void*>(p));
    return _mm512_castsi512_ps(_mm512_shuffle_epi8(bytes, byte_swap_mask));
}

arch_target("avx512f,avx512bw") float horizontal_sum_avx512(__m512 values) {
    return _mm512_reduce_add_ps(values);
}

template <similarity_operation Operation>
arch_target("avx512f,avx512bw") float compute_serialized_avx512(bytes_view v1, bytes_view v2, vector_dimension_t dimension) {
    __m512 dot_product_0 = _mm512_setzero_ps();
    __m512 dot_product_1 = _mm512_setzero_ps();
    __m512 squared_norm_a_0 = _mm512_setzero_ps();
    __m512 squared_norm_a_1 = _mm512_setzero_ps();
    __m512 squared_norm_b_0 = _mm512_setzero_ps();
    __m512 squared_norm_b_1 = _mm512_setzero_ps();
    __m512 squared_distance_0 = _mm512_setzero_ps();
    __m512 squared_distance_1 = _mm512_setzero_ps();
    const auto byte_swap_mask = byte_swap_mask_avx512();

    size_t i = 0;
    for (; i + 32 <= dimension; i += 32) {
        const auto a0 = load_be_floats_avx512(v1.data() + i * sizeof(float), byte_swap_mask);
        const auto b0 = load_be_floats_avx512(v2.data() + i * sizeof(float), byte_swap_mask);
        const auto a1 = load_be_floats_avx512(v1.data() + (i + 16) * sizeof(float), byte_swap_mask);
        const auto b1 = load_be_floats_avx512(v2.data() + (i + 16) * sizeof(float), byte_swap_mask);

        if constexpr (Operation == similarity_operation::cosine) {
            dot_product_0 = _mm512_fmadd_ps(a0, b0, dot_product_0);
            dot_product_1 = _mm512_fmadd_ps(a1, b1, dot_product_1);
            squared_norm_a_0 = _mm512_fmadd_ps(a0, a0, squared_norm_a_0);
            squared_norm_a_1 = _mm512_fmadd_ps(a1, a1, squared_norm_a_1);
            squared_norm_b_0 = _mm512_fmadd_ps(b0, b0, squared_norm_b_0);
            squared_norm_b_1 = _mm512_fmadd_ps(b1, b1, squared_norm_b_1);
        } else if constexpr (Operation == similarity_operation::euclidean) {
            const auto diff0 = _mm512_sub_ps(a0, b0);
            const auto diff1 = _mm512_sub_ps(a1, b1);
            squared_distance_0 = _mm512_fmadd_ps(diff0, diff0, squared_distance_0);
            squared_distance_1 = _mm512_fmadd_ps(diff1, diff1, squared_distance_1);
        } else {
            dot_product_0 = _mm512_fmadd_ps(a0, b0, dot_product_0);
            dot_product_1 = _mm512_fmadd_ps(a1, b1, dot_product_1);
        }
    }

    for (; i + 16 <= dimension; i += 16) {
        const auto a = load_be_floats_avx512(v1.data() + i * sizeof(float), byte_swap_mask);
        const auto b = load_be_floats_avx512(v2.data() + i * sizeof(float), byte_swap_mask);

        if constexpr (Operation == similarity_operation::cosine) {
            dot_product_0 = _mm512_fmadd_ps(a, b, dot_product_0);
            squared_norm_a_0 = _mm512_fmadd_ps(a, a, squared_norm_a_0);
            squared_norm_b_0 = _mm512_fmadd_ps(b, b, squared_norm_b_0);
        } else if constexpr (Operation == similarity_operation::euclidean) {
            const auto diff = _mm512_sub_ps(a, b);
            squared_distance_0 = _mm512_fmadd_ps(diff, diff, squared_distance_0);
        } else {
            dot_product_0 = _mm512_fmadd_ps(a, b, dot_product_0);
        }
    }

    if (i < dimension) {
        const auto a = load_be_floats_avx512_masked(v1.data() + i * sizeof(float), byte_swap_mask, dimension - i);
        const auto b = load_be_floats_avx512_masked(v2.data() + i * sizeof(float), byte_swap_mask, dimension - i);

        if constexpr (Operation == similarity_operation::cosine) {
            dot_product_0 = _mm512_fmadd_ps(a, b, dot_product_0);
            squared_norm_a_0 = _mm512_fmadd_ps(a, a, squared_norm_a_0);
            squared_norm_b_0 = _mm512_fmadd_ps(b, b, squared_norm_b_0);
        } else if constexpr (Operation == similarity_operation::euclidean) {
            const auto diff = _mm512_sub_ps(a, b);
            squared_distance_0 = _mm512_fmadd_ps(diff, diff, squared_distance_0);
        } else {
            dot_product_0 = _mm512_fmadd_ps(a, b, dot_product_0);
        }
    }

    if constexpr (Operation == similarity_operation::cosine) {
        return finish_cosine_similarity(
                horizontal_sum_avx512(_mm512_add_ps(dot_product_0, dot_product_1)),
                horizontal_sum_avx512(_mm512_add_ps(squared_norm_a_0, squared_norm_a_1)),
                horizontal_sum_avx512(_mm512_add_ps(squared_norm_b_0, squared_norm_b_1)));
    } else if constexpr (Operation == similarity_operation::euclidean) {
        return finish_euclidean_similarity(
                horizontal_sum_avx512(_mm512_add_ps(squared_distance_0, squared_distance_1)));
    } else {
        return finish_dot_product_similarity(horizontal_sum_avx512(_mm512_add_ps(dot_product_0, dot_product_1)));
    }
}

arch_target("avx512f,avx512bw") float compute_serialized_avx512_euclidean_four_accumulators(
        bytes_view v1, bytes_view v2, vector_dimension_t dimension) {
    __m512 squared_distance_0 = _mm512_setzero_ps();
    __m512 squared_distance_1 = _mm512_setzero_ps();
    __m512 squared_distance_2 = _mm512_setzero_ps();
    __m512 squared_distance_3 = _mm512_setzero_ps();
    const auto byte_swap_mask = byte_swap_mask_avx512();

    size_t i = 0;
    for (; i + 64 <= dimension; i += 64) {
        const auto a0 = load_be_floats_avx512(v1.data() + i * sizeof(float), byte_swap_mask);
        const auto b0 = load_be_floats_avx512(v2.data() + i * sizeof(float), byte_swap_mask);
        const auto a1 = load_be_floats_avx512(v1.data() + (i + 16) * sizeof(float), byte_swap_mask);
        const auto b1 = load_be_floats_avx512(v2.data() + (i + 16) * sizeof(float), byte_swap_mask);
        const auto a2 = load_be_floats_avx512(v1.data() + (i + 32) * sizeof(float), byte_swap_mask);
        const auto b2 = load_be_floats_avx512(v2.data() + (i + 32) * sizeof(float), byte_swap_mask);
        const auto a3 = load_be_floats_avx512(v1.data() + (i + 48) * sizeof(float), byte_swap_mask);
        const auto b3 = load_be_floats_avx512(v2.data() + (i + 48) * sizeof(float), byte_swap_mask);

        const auto diff0 = _mm512_sub_ps(a0, b0);
        const auto diff1 = _mm512_sub_ps(a1, b1);
        const auto diff2 = _mm512_sub_ps(a2, b2);
        const auto diff3 = _mm512_sub_ps(a3, b3);
        squared_distance_0 = _mm512_fmadd_ps(diff0, diff0, squared_distance_0);
        squared_distance_1 = _mm512_fmadd_ps(diff1, diff1, squared_distance_1);
        squared_distance_2 = _mm512_fmadd_ps(diff2, diff2, squared_distance_2);
        squared_distance_3 = _mm512_fmadd_ps(diff3, diff3, squared_distance_3);
    }

    for (; i + 32 <= dimension; i += 32) {
        const auto a0 = load_be_floats_avx512(v1.data() + i * sizeof(float), byte_swap_mask);
        const auto b0 = load_be_floats_avx512(v2.data() + i * sizeof(float), byte_swap_mask);
        const auto a1 = load_be_floats_avx512(v1.data() + (i + 16) * sizeof(float), byte_swap_mask);
        const auto b1 = load_be_floats_avx512(v2.data() + (i + 16) * sizeof(float), byte_swap_mask);

        const auto diff0 = _mm512_sub_ps(a0, b0);
        const auto diff1 = _mm512_sub_ps(a1, b1);
        squared_distance_0 = _mm512_fmadd_ps(diff0, diff0, squared_distance_0);
        squared_distance_1 = _mm512_fmadd_ps(diff1, diff1, squared_distance_1);
    }

    for (; i + 16 <= dimension; i += 16) {
        const auto a = load_be_floats_avx512(v1.data() + i * sizeof(float), byte_swap_mask);
        const auto b = load_be_floats_avx512(v2.data() + i * sizeof(float), byte_swap_mask);

        const auto diff = _mm512_sub_ps(a, b);
        squared_distance_0 = _mm512_fmadd_ps(diff, diff, squared_distance_0);
    }

    if (i < dimension) {
        const auto a = load_be_floats_avx512_masked(v1.data() + i * sizeof(float), byte_swap_mask, dimension - i);
        const auto b = load_be_floats_avx512_masked(v2.data() + i * sizeof(float), byte_swap_mask, dimension - i);

        const auto diff = _mm512_sub_ps(a, b);
        squared_distance_0 = _mm512_fmadd_ps(diff, diff, squared_distance_0);
    }

    const auto squared_distance = _mm512_add_ps(
            _mm512_add_ps(squared_distance_0, squared_distance_1),
            _mm512_add_ps(squared_distance_2, squared_distance_3));
    return finish_euclidean_similarity(horizontal_sum_avx512(squared_distance));
}

arch_target("avx512f,avx512bw") float compute_serialized_avx512_dot_product_four_accumulators(
        bytes_view v1, bytes_view v2, vector_dimension_t dimension) {
    __m512 dot_product_0 = _mm512_setzero_ps();
    __m512 dot_product_1 = _mm512_setzero_ps();
    __m512 dot_product_2 = _mm512_setzero_ps();
    __m512 dot_product_3 = _mm512_setzero_ps();
    const auto byte_swap_mask = byte_swap_mask_avx512();

    size_t i = 0;
    for (; i + 64 <= dimension; i += 64) {
        const auto a0 = load_be_floats_avx512(v1.data() + i * sizeof(float), byte_swap_mask);
        const auto b0 = load_be_floats_avx512(v2.data() + i * sizeof(float), byte_swap_mask);
        const auto a1 = load_be_floats_avx512(v1.data() + (i + 16) * sizeof(float), byte_swap_mask);
        const auto b1 = load_be_floats_avx512(v2.data() + (i + 16) * sizeof(float), byte_swap_mask);
        const auto a2 = load_be_floats_avx512(v1.data() + (i + 32) * sizeof(float), byte_swap_mask);
        const auto b2 = load_be_floats_avx512(v2.data() + (i + 32) * sizeof(float), byte_swap_mask);
        const auto a3 = load_be_floats_avx512(v1.data() + (i + 48) * sizeof(float), byte_swap_mask);
        const auto b3 = load_be_floats_avx512(v2.data() + (i + 48) * sizeof(float), byte_swap_mask);

        dot_product_0 = _mm512_fmadd_ps(a0, b0, dot_product_0);
        dot_product_1 = _mm512_fmadd_ps(a1, b1, dot_product_1);
        dot_product_2 = _mm512_fmadd_ps(a2, b2, dot_product_2);
        dot_product_3 = _mm512_fmadd_ps(a3, b3, dot_product_3);
    }

    for (; i + 32 <= dimension; i += 32) {
        const auto a0 = load_be_floats_avx512(v1.data() + i * sizeof(float), byte_swap_mask);
        const auto b0 = load_be_floats_avx512(v2.data() + i * sizeof(float), byte_swap_mask);
        const auto a1 = load_be_floats_avx512(v1.data() + (i + 16) * sizeof(float), byte_swap_mask);
        const auto b1 = load_be_floats_avx512(v2.data() + (i + 16) * sizeof(float), byte_swap_mask);

        dot_product_0 = _mm512_fmadd_ps(a0, b0, dot_product_0);
        dot_product_1 = _mm512_fmadd_ps(a1, b1, dot_product_1);
    }

    for (; i + 16 <= dimension; i += 16) {
        const auto a = load_be_floats_avx512(v1.data() + i * sizeof(float), byte_swap_mask);
        const auto b = load_be_floats_avx512(v2.data() + i * sizeof(float), byte_swap_mask);

        dot_product_0 = _mm512_fmadd_ps(a, b, dot_product_0);
    }

    if (i < dimension) {
        const auto a = load_be_floats_avx512_masked(v1.data() + i * sizeof(float), byte_swap_mask, dimension - i);
        const auto b = load_be_floats_avx512_masked(v2.data() + i * sizeof(float), byte_swap_mask, dimension - i);

        dot_product_0 = _mm512_fmadd_ps(a, b, dot_product_0);
    }

    const auto dot_product = _mm512_add_ps(
            _mm512_add_ps(dot_product_0, dot_product_1),
            _mm512_add_ps(dot_product_2, dot_product_3));
    return finish_dot_product_similarity(horizontal_sum_avx512(dot_product));
}

float compute_serialized_avx512_cosine(const bytes_opt& v1, const bytes_opt& v2, vector_dimension_t dimension) {
    return compute_serialized_with<similarity_operation::cosine>(v1, v2, dimension, compute_serialized_avx512<similarity_operation::cosine>);
}

float compute_serialized_avx512_euclidean(const bytes_opt& v1, const bytes_opt& v2, vector_dimension_t dimension) {
    return compute_serialized_with<similarity_operation::euclidean>(
            v1, v2, dimension, compute_serialized_avx512_euclidean_four_accumulators);
}

float compute_serialized_avx512_dot_product(const bytes_opt& v1, const bytes_opt& v2, vector_dimension_t dimension) {
    return compute_serialized_with<similarity_operation::dot_product>(
            v1, v2, dimension, compute_serialized_avx512_dot_product_four_accumulators);
}

const serialized_similarity_backend avx512_backend = {
        .name = "avx512",
        .cosine = compute_serialized_avx512_cosine,
        .euclidean = compute_serialized_avx512_euclidean,
        .dot_product = compute_serialized_avx512_dot_product,
};

const serialized_similarity_backend avx512_vnni_backend = {
        .name = "avx512_vnni",
        .cosine = compute_serialized_avx512_cosine,
        .euclidean = compute_serialized_avx512_euclidean,
        .dot_product = compute_serialized_avx512_dot_product,
};

#endif

#ifdef __aarch64__

struct arm_simd_support {
    bool neon;
    bool sve;
};

arm_simd_support detect_arm_simd_support() {
#ifdef __linux__
    const auto hwcap = getauxval(AT_HWCAP);
    return {
        .neon = (hwcap & HWCAP_ASIMD) != 0,
#ifdef HWCAP_SVE
        .sve = (hwcap & HWCAP_SVE) != 0,
#else
        .sve = false,
#endif
    };
#else
    return {
        .neon = true,
        .sve = false,
    };
#endif
}

const arm_simd_support& get_arm_simd_support() {
    static const auto support = detect_arm_simd_support();
    return support;
}

float32x4_t load_be_floats_neon(const int8_t* p) {
    auto bytes = vld1q_u8(reinterpret_cast<const uint8_t*>(p));
    if constexpr (std::endian::native == std::endian::little) {
        bytes = vrev32q_u8(bytes);
    }
    return vreinterpretq_f32_u8(bytes);
}

template <similarity_operation Operation>
float compute_serialized_neon(bytes_view v1, bytes_view v2, vector_dimension_t dimension) {
    float32x4_t dot_product_0 = vdupq_n_f32(0.0f);
    float32x4_t dot_product_1 = vdupq_n_f32(0.0f);
    float32x4_t dot_product_2 = vdupq_n_f32(0.0f);
    float32x4_t dot_product_3 = vdupq_n_f32(0.0f);
    float32x4_t squared_norm_a_0 = vdupq_n_f32(0.0f);
    float32x4_t squared_norm_a_1 = vdupq_n_f32(0.0f);
    float32x4_t squared_norm_a_2 = vdupq_n_f32(0.0f);
    float32x4_t squared_norm_a_3 = vdupq_n_f32(0.0f);
    float32x4_t squared_norm_b_0 = vdupq_n_f32(0.0f);
    float32x4_t squared_norm_b_1 = vdupq_n_f32(0.0f);
    float32x4_t squared_norm_b_2 = vdupq_n_f32(0.0f);
    float32x4_t squared_norm_b_3 = vdupq_n_f32(0.0f);
    float32x4_t squared_distance_0 = vdupq_n_f32(0.0f);
    float32x4_t squared_distance_1 = vdupq_n_f32(0.0f);
    float32x4_t squared_distance_2 = vdupq_n_f32(0.0f);
    float32x4_t squared_distance_3 = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 16 <= dimension; i += 16) {
        const auto a0 = load_be_floats_neon(v1.data() + i * sizeof(float));
        const auto b0 = load_be_floats_neon(v2.data() + i * sizeof(float));
        const auto a1 = load_be_floats_neon(v1.data() + (i + 4) * sizeof(float));
        const auto b1 = load_be_floats_neon(v2.data() + (i + 4) * sizeof(float));
        const auto a2 = load_be_floats_neon(v1.data() + (i + 8) * sizeof(float));
        const auto b2 = load_be_floats_neon(v2.data() + (i + 8) * sizeof(float));
        const auto a3 = load_be_floats_neon(v1.data() + (i + 12) * sizeof(float));
        const auto b3 = load_be_floats_neon(v2.data() + (i + 12) * sizeof(float));

        if constexpr (Operation == similarity_operation::cosine) {
            dot_product_0 = vmlaq_f32(dot_product_0, a0, b0);
            dot_product_1 = vmlaq_f32(dot_product_1, a1, b1);
            dot_product_2 = vmlaq_f32(dot_product_2, a2, b2);
            dot_product_3 = vmlaq_f32(dot_product_3, a3, b3);
            squared_norm_a_0 = vmlaq_f32(squared_norm_a_0, a0, a0);
            squared_norm_a_1 = vmlaq_f32(squared_norm_a_1, a1, a1);
            squared_norm_a_2 = vmlaq_f32(squared_norm_a_2, a2, a2);
            squared_norm_a_3 = vmlaq_f32(squared_norm_a_3, a3, a3);
            squared_norm_b_0 = vmlaq_f32(squared_norm_b_0, b0, b0);
            squared_norm_b_1 = vmlaq_f32(squared_norm_b_1, b1, b1);
            squared_norm_b_2 = vmlaq_f32(squared_norm_b_2, b2, b2);
            squared_norm_b_3 = vmlaq_f32(squared_norm_b_3, b3, b3);
        } else if constexpr (Operation == similarity_operation::euclidean) {
            const auto diff0 = vsubq_f32(a0, b0);
            const auto diff1 = vsubq_f32(a1, b1);
            const auto diff2 = vsubq_f32(a2, b2);
            const auto diff3 = vsubq_f32(a3, b3);
            squared_distance_0 = vmlaq_f32(squared_distance_0, diff0, diff0);
            squared_distance_1 = vmlaq_f32(squared_distance_1, diff1, diff1);
            squared_distance_2 = vmlaq_f32(squared_distance_2, diff2, diff2);
            squared_distance_3 = vmlaq_f32(squared_distance_3, diff3, diff3);
        } else {
            dot_product_0 = vmlaq_f32(dot_product_0, a0, b0);
            dot_product_1 = vmlaq_f32(dot_product_1, a1, b1);
            dot_product_2 = vmlaq_f32(dot_product_2, a2, b2);
            dot_product_3 = vmlaq_f32(dot_product_3, a3, b3);
        }
    }

    for (; i + 4 <= dimension; i += 4) {
        const auto a = load_be_floats_neon(v1.data() + i * sizeof(float));
        const auto b = load_be_floats_neon(v2.data() + i * sizeof(float));

        if constexpr (Operation == similarity_operation::cosine) {
            dot_product_0 = vmlaq_f32(dot_product_0, a, b);
            squared_norm_a_0 = vmlaq_f32(squared_norm_a_0, a, a);
            squared_norm_b_0 = vmlaq_f32(squared_norm_b_0, b, b);
        } else if constexpr (Operation == similarity_operation::euclidean) {
            const auto diff = vsubq_f32(a, b);
            squared_distance_0 = vmlaq_f32(squared_distance_0, diff, diff);
        } else {
            dot_product_0 = vmlaq_f32(dot_product_0, a, b);
        }
    }

    float dot_product_tail = 0.0f;
    float squared_norm_a_tail = 0.0f;
    float squared_norm_b_tail = 0.0f;
    float squared_distance_tail = 0.0f;
    for (; i < dimension; ++i) {
        const float a = read_be_float(v1.data() + i * sizeof(float));
        const float b = read_be_float(v2.data() + i * sizeof(float));

        if constexpr (Operation == similarity_operation::cosine) {
            dot_product_tail += a * b;
            squared_norm_a_tail += a * a;
            squared_norm_b_tail += b * b;
        } else if constexpr (Operation == similarity_operation::euclidean) {
            const float diff = a - b;
            squared_distance_tail += diff * diff;
        } else {
            dot_product_tail += a * b;
        }
    }

    if constexpr (Operation == similarity_operation::cosine) {
        const auto dot_product = vaddq_f32(vaddq_f32(dot_product_0, dot_product_1), vaddq_f32(dot_product_2, dot_product_3));
        const auto squared_norm_a = vaddq_f32(
                vaddq_f32(squared_norm_a_0, squared_norm_a_1),
                vaddq_f32(squared_norm_a_2, squared_norm_a_3));
        const auto squared_norm_b = vaddq_f32(
                vaddq_f32(squared_norm_b_0, squared_norm_b_1),
                vaddq_f32(squared_norm_b_2, squared_norm_b_3));
        return finish_cosine_similarity(
                vaddvq_f32(dot_product) + dot_product_tail,
                vaddvq_f32(squared_norm_a) + squared_norm_a_tail,
                vaddvq_f32(squared_norm_b) + squared_norm_b_tail);
    } else if constexpr (Operation == similarity_operation::euclidean) {
        const auto squared_distance = vaddq_f32(
                vaddq_f32(squared_distance_0, squared_distance_1),
                vaddq_f32(squared_distance_2, squared_distance_3));
        return finish_euclidean_similarity(vaddvq_f32(squared_distance) + squared_distance_tail);
    } else {
        const auto dot_product = vaddq_f32(vaddq_f32(dot_product_0, dot_product_1), vaddq_f32(dot_product_2, dot_product_3));
        return finish_dot_product_similarity(vaddvq_f32(dot_product) + dot_product_tail);
    }
}

float compute_serialized_neon_cosine(const bytes_opt& v1, const bytes_opt& v2, vector_dimension_t dimension) {
    return compute_serialized_with<similarity_operation::cosine>(v1, v2, dimension, compute_serialized_neon<similarity_operation::cosine>);
}

float compute_serialized_neon_euclidean(const bytes_opt& v1, const bytes_opt& v2, vector_dimension_t dimension) {
    return compute_serialized_with<similarity_operation::euclidean>(v1, v2, dimension, compute_serialized_neon<similarity_operation::euclidean>);
}

float compute_serialized_neon_dot_product(const bytes_opt& v1, const bytes_opt& v2, vector_dimension_t dimension) {
    return compute_serialized_with<similarity_operation::dot_product>(v1, v2, dimension, compute_serialized_neon<similarity_operation::dot_product>);
}

const serialized_similarity_backend neon_backend = {
        .name = "neon",
        .cosine = compute_serialized_neon_cosine,
        .euclidean = compute_serialized_neon_euclidean,
        .dot_product = compute_serialized_neon_dot_product,
};

#endif

const serialized_similarity_backend& select_backend(db::simd_optimization_mode requested) {
    if (requested == db::simd_optimization_mode::off) {
        return scalar_backend;
    }

#ifdef __x86_64__
    const auto& x86 = get_x86_simd_support();
    const auto avx512_safe = current_thread_has_single_cpu_affinity();
    if (requested == db::simd_optimization_mode::sse) {
        if (!x86.ssse3) {
            throw_backend_unavailable(requested, "SSSE3 is not supported by the current CPU");
        }
        return ssse3_backend;
    }
    if (requested == db::simd_optimization_mode::avx2) {
        if (!x86.avx2_fma) {
            throw_backend_unavailable(requested, "AVX2+FMA is not supported by the current CPU");
        }
        return avx2_backend;
    }
    if (requested == db::simd_optimization_mode::avx512) {
        if (!x86.avx512_float) {
            throw_backend_unavailable(requested, "AVX-512F+AVX-512BW is not supported by the current CPU");
        }
        if (!avx512_safe) {
            throw_backend_unavailable(requested, "the current thread is not pinned to a single CPU");
        }
        if (x86.avx512_vnni) {
            return avx512_vnni_backend;
        }
        return avx512_backend;
    }
    if (requested == db::simd_optimization_mode::neon || requested == db::simd_optimization_mode::sve) {
        throw_backend_unavailable(requested, "the process is not running on AArch64");
    }
    if (requested == db::simd_optimization_mode::automatic) {
        if (x86.avx512_float && avx512_safe) {
            if (x86.avx512_vnni) {
                return avx512_vnni_backend;
            }
            return avx512_backend;
        }
        if (x86.avx2_fma) {
            return avx2_backend;
        }
        if (x86.ssse3) {
            return ssse3_backend;
        }
        return scalar_backend;
    }
#elif defined(__aarch64__)
    const auto& arm = get_arm_simd_support();
    if (requested == db::simd_optimization_mode::neon) {
        if (!arm.neon) {
            throw_backend_unavailable(requested, "NEON/ASIMD is not supported by the current CPU");
        }
        return neon_backend;
    }
    if (requested == db::simd_optimization_mode::sve) {
        if (!arm.sve) {
            throw_backend_unavailable(requested, "SVE is not supported by the current CPU");
        }
        throw_backend_unavailable(requested, "the SVE backend is not compiled into this build");
    }
    if (requested == db::simd_optimization_mode::sse || requested == db::simd_optimization_mode::avx2 ||
            requested == db::simd_optimization_mode::avx512) {
        throw_backend_unavailable(requested, "the process is not running on x86_64");
    }
    if (requested == db::simd_optimization_mode::automatic) {
        if (arm.neon) {
            return neon_backend;
        }
        return scalar_backend;
    }
#else
    if (requested == db::simd_optimization_mode::sse || requested == db::simd_optimization_mode::avx2 ||
            requested == db::simd_optimization_mode::avx512 || requested == db::simd_optimization_mode::neon ||
            requested == db::simd_optimization_mode::sve) {
        throw_backend_unavailable(requested, "the binary was built without that architecture backend");
    }
    if (requested == db::simd_optimization_mode::automatic) {
        return scalar_backend;
    }
#endif

    throw exceptions::configuration_exception(fmt::format(
            "Invalid simd_optimization_options.vector_similarity value: {}.",
            db::config::simd_optimization_mode_name(requested)));
}

serialized_similarity_function_t function_for(const serialized_similarity_backend& backend, const function_name& name) {
    if (name == SIMILARITY_COSINE_FUNCTION_NAME) {
        return backend.cosine;
    }
    if (name == SIMILARITY_EUCLIDEAN_FUNCTION_NAME) {
        return backend.euclidean;
    }
    return backend.dot_product;
}

const serialized_similarity_backend& auto_backend_for_current_thread() {
    // Auto selection can query CPU features and thread affinity. Cache per thread
    // so the helper API does not repeat that work in the hot path.
    thread_local const auto* backend = &select_backend(db::simd_optimization_mode::automatic);
    return *backend;
}

} // namespace

std::vector<float> extract_float_vector(const bytes_opt& param, vector_dimension_t dimension) {
    const auto serialized = require_float_vector(param, dimension);

    std::vector<float> result(dimension);
    for (size_t i = 0; i < dimension; ++i) {
        result[i] = read_be_float(serialized.data() + i * sizeof(float));
    }

    return result;
}

// The computations of similarity scores match the exact formulas of Cassandra's (jVector's) implementation to ensure compatibility.
// There exist tests checking the compliance of the results.
// Reference:
// https://github.com/datastax/jvector/blob/f967f1c9249035b63b55a566fac7d4dc38380349/jvector-base/src/main/java/io/github/jbellis/jvector/vector/VectorSimilarityFunction.java#L36-L69

// You should only use this function if you need to preserve the original vectors and cannot normalize
// them in advance.
float compute_cosine_similarity(std::span<const float> v1, std::span<const float> v2) {
    #pragma clang fp contract(fast) reassociate(on) // Allow the compiler to optimize the loop.
    float dot_product = 0.0;
    float squared_norm_a = 0.0;
    float squared_norm_b = 0.0;

    for (size_t i = 0; i < v1.size(); ++i) {
        float a = v1[i];
        float b = v2[i];

        dot_product += a * b;
        squared_norm_a += a * a;
        squared_norm_b += b * b;
    }

    return finish_cosine_similarity(dot_product, squared_norm_a, squared_norm_b);
}

float compute_euclidean_similarity(std::span<const float> v1, std::span<const float> v2) {
    #pragma clang fp contract(fast) reassociate(on) // Allow the compiler to optimize the loop.
    float sum = 0.0;

    for (size_t i = 0; i < v1.size(); ++i) {
        float a = v1[i];
        float b = v2[i];

        float diff = a - b;
        sum += diff * diff;
    }

    return finish_euclidean_similarity(sum);
}

// Assumes that both vectors are L2-normalized.
// This similarity is intended as an optimized way to perform cosine similarity calculation.
float compute_dot_product_similarity(std::span<const float> v1, std::span<const float> v2) {
    #pragma clang fp contract(fast) reassociate(on) // Allow the compiler to optimize the loop.
    float dot_product = 0.0;

    for (size_t i = 0; i < v1.size(); ++i) {
        float a = v1[i];
        float b = v2[i];
        dot_product += a * b;
    }

    return finish_dot_product_similarity(dot_product);
}

float compute_serialized_cosine_similarity(const bytes_opt& v1, const bytes_opt& v2, vector_dimension_t dimension) {
    return auto_backend_for_current_thread().cosine(v1, v2, dimension);
}

float compute_serialized_euclidean_similarity(const bytes_opt& v1, const bytes_opt& v2, vector_dimension_t dimension) {
    return auto_backend_for_current_thread().euclidean(v1, v2, dimension);
}

float compute_serialized_dot_product_similarity(const bytes_opt& v1, const bytes_opt& v2, vector_dimension_t dimension) {
    return auto_backend_for_current_thread().dot_product(v1, v2, dimension);
}

sstring select_serialized_similarity_backend(db::simd_optimization_mode backend) {
    return sstring(select_backend(backend).name);
}

serialized_similarity_function_t select_serialized_similarity_function(const function_name& name, db::simd_optimization_mode backend) {
    return function_for(select_backend(backend), name);
}

} // namespace detail

thread_local const std::unordered_map<function_name, similarity_function_t> SIMILARITY_FUNCTIONS = {
        {SIMILARITY_COSINE_FUNCTION_NAME, detail::compute_cosine_similarity},
        {SIMILARITY_EUCLIDEAN_FUNCTION_NAME, detail::compute_euclidean_similarity},
        {SIMILARITY_DOT_PRODUCT_FUNCTION_NAME, detail::compute_dot_product_similarity},
};

vector_similarity_fct::vector_similarity_fct(const sstring& name, const std::vector<data_type>& arg_types, db::simd_optimization_mode backend)
    : native_scalar_function(name, float_type, arg_types)
    , _similarity_func(detail::select_serialized_similarity_function(_name, backend))
    , _dimension(static_cast<const vector_type_impl&>(*arg_types[0]).get_dimension()) {
}

std::vector<data_type> retrieve_vector_arg_types(const function_name& name, const std::vector<shared_ptr<assignment_testable>>& provided_args) {
    if (provided_args.size() != 2) {
        throw exceptions::invalid_request_exception(fmt::format("Invalid number of arguments for function {}(vector<float, n>, vector<float, n>)", name));
    }

    auto [first_result, first_dim_opt] = provided_args[0]->test_assignment_any_size_float_vector();
    auto [second_result, second_dim_opt] = provided_args[1]->test_assignment_any_size_float_vector();

    auto invalid_type_error_message = [&name](const shared_ptr<assignment_testable>& arg) {
        auto type = arg->assignment_testable_type_opt();
        const auto& source_context = arg->assignment_testable_source_context();
        if (type) {
            return fmt::format("Function {} requires a float vector argument, but found {} of type {}", name, source_context, type.value()->cql3_type_name());
        } else {
            return fmt::format("Function {} requires a float vector argument, but found {}", name, source_context);
        }
    };

    if (!is_assignable(first_result)) {
        throw exceptions::invalid_request_exception(invalid_type_error_message(provided_args[0]));
    }
    if (!is_assignable(second_result)) {
        throw exceptions::invalid_request_exception(invalid_type_error_message(provided_args[1]));
    }

    if (!first_dim_opt && !second_dim_opt) {
        throw exceptions::invalid_request_exception(fmt::format("Cannot infer type of argument {} for function {}(vector<float, n>, vector<float, n>)",
                provided_args[0]->assignment_testable_source_context(), name));
    }
    if (first_dim_opt && second_dim_opt) {
        if (*first_dim_opt != *second_dim_opt) {
            throw exceptions::invalid_request_exception(fmt::format(
                    "All arguments must have the same vector dimensions, but found vector<float, {}> and vector<float, {}>", *first_dim_opt, *second_dim_opt));
        }
    }

    vector_dimension_t dimension = first_dim_opt ? *first_dim_opt : *second_dim_opt;
    auto type = vector_type_impl::get_instance(float_type, dimension);
    return {type, type};
}

bytes_opt vector_similarity_fct::execute(std::span<const bytes_opt> parameters) {
    if (std::any_of(parameters.begin(), parameters.end(), [](const auto& param) {
            return !param;
        })) {
        return std::nullopt;
    }

    float result = _similarity_func(parameters[0], parameters[1], _dimension);
    return float_type->decompose(result);
}

} // namespace functions
} // namespace cql3
