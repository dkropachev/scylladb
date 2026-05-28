/*
 * Copyright (C) 2020-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */


#include "array-search.hh"

#include <bit>

#ifdef __x86_64__
#include <x86intrin.h>
#define arch_target(name) [[gnu::target(name)]]
#else
#define arch_target(name)
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "db/config.hh"

namespace utils {

arch_target("default") int array_search_gt_impl(int64_t val, const int64_t* array, const int capacity, const int size) {
    int i;

    for (i = 0; i < size; i++) {
        if (val < array[i])
            break;
    }

    return i;
}

static inline unsigned array_search_eq_impl(uint8_t val, const uint8_t* arr, unsigned len) {
    unsigned i;

    for (i = 0; i < len; i++) {
        if (arr[i] == val) {
            break;
        }
    }

    return i;
}

template <bool equal>
static inline size_t byte_search_scalar(uint8_t val, const int8_t* array, size_t size) {
    const auto signed_val = static_cast<int8_t>(val);
    for (size_t i = 0; i < size; ++i) {
        if ((array[i] == signed_val) == equal) {
            return i;
        }
    }
    return size;
}

#ifdef __x86_64__
static bool cpu_supports_avx2() noexcept {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
}

static bool cpu_supports_avx512() noexcept {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
}

static bool cpu_supports_sse2() noexcept {
    __builtin_cpu_init();
    return __builtin_cpu_supports("sse2");
}
#endif

arch_target("default") unsigned array_search_16_eq_impl(uint8_t val, const uint8_t* arr) {
    return array_search_eq_impl(val, arr, 16);
}

arch_target("default") unsigned array_search_32_eq_impl(uint8_t val, const uint8_t* arr) {
    return array_search_eq_impl(val, arr, 32);
}

arch_target("default") unsigned array_search_x32_eq_impl(uint8_t val, const uint8_t* arr, int nr) {
    return array_search_eq_impl(val, arr, 32 * nr);
}

arch_target("default") size_t byte_search_eq_impl(uint8_t val, const int8_t* array, size_t size) {
    return byte_search_scalar<true>(val, array, size);
}

arch_target("default") size_t byte_search_ne_impl(uint8_t val, const int8_t* array, size_t size) {
    return byte_search_scalar<false>(val, array, size);
}

#ifdef __x86_64__

/*
 * The AVX2 version doesn't take @size argument into account and expects
 * all the elements above it to be less than any possible value.
 *
 * To make it work without this requirement we'd need to:
 *  - limit the loop iterations to size instead of capacity
 *  - explicitly set to 1 all the mask's bits for elements >= size
 * both do make things up to 50% slower.
 */

arch_target("avx2") int array_search_gt_impl(int64_t val, const int64_t* array, const int capacity, const int size) {
    int cnt = 0;

    // 0. Load key into 256-bit ymm
    __m256i k = _mm256_set1_epi64x(val);
    for (int i = 0; i < capacity; i += 4) {
        // 4. Count the number of 1-s, each gt match gives 8 bits
        cnt += _mm_popcnt_u32(
                    // 3. Pack result into 4 bytes -- 1 byte from each comparison
                    _mm256_movemask_epi8(
                        // 2. Compare array[i] > key, 4 elements in one go
                        _mm256_cmpgt_epi64(
                            // 1. Load next 4 elements into ymm
                            _mm256_lddqu_si256((__m256i*)&array[i]), k
                        )
                    )
                ) / 8;
    }

    /*
     * 5. We need the index of the first gt value. Unused elements are < k
     *    for sure, so count from the tail of the used part.
     *
     *   <grumble>
     *    We might have done it the other way -- store the maximum in unused,
     *    check for key >= array[i] in the above loop and just return the cnt,
     *    but ...  AVX2 instructions set doesn't have the PCMPGE
     *
     *    SSE* set (predecessor) has cmpge, but eats 2 keys in one go
     *    AVX-512 (successor) has it back, and even eats 8 keys, but is
     *    not widely available
     *   </grumble>
     */
    return size - cnt;
}

/*
 * SSE4 version of searching in array for an exact match.
 */
arch_target("sse") unsigned array_search_16_eq_impl(uint8_t val, const uint8_t* arr) {
	auto a = _mm_set1_epi8(val);
	auto b = _mm_lddqu_si128((__m128i*)arr);
	auto c = _mm_cmpeq_epi8(a, b);
	unsigned int m = _mm_movemask_epi8(c);
	return __builtin_ctz(m | 0x10000);
}

/*
 * AVX2 version of searching in array for an exact match.
 */
arch_target("avx2") unsigned array_search_32_eq_impl(uint8_t val, const uint8_t* arr) {
    auto a = _mm256_set1_epi8(val);
    auto b = _mm256_lddqu_si256((__m256i*)arr);
    auto c = _mm256_cmpeq_epi8(a, b);
    unsigned long long m = _mm256_movemask_epi8(c);
    return __builtin_ctzll(m | 0x100000000ull);
}

arch_target("avx2") unsigned array_search_x32_eq_impl(uint8_t val, const uint8_t* arr, int nr) {
    unsigned len = 32 * nr;
    auto a = _mm256_set1_epi8(val);
    for (unsigned off = 0; off < len; off += 32) {
        auto b = _mm256_lddqu_si256((__m256i*)arr);
        auto c = _mm256_cmpeq_epi8(a, b);
        unsigned m = _mm256_movemask_epi8(c);
        if (m != 0) {
            return __builtin_ctz(m) + off;
        }
    }
    return len;
}

arch_target("sse2") static size_t byte_search_eq_sse2(uint8_t val, const int8_t* array, size_t size) {
    constexpr size_t vector_size = 16;
    const auto needle = _mm_set1_epi8(static_cast<char>(val));
    size_t i = 0;
    for (; i + vector_size <= size; i += vector_size) {
        const auto chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(array + i));
        const auto matches = _mm_cmpeq_epi8(chunk, needle);
        const auto mask = static_cast<unsigned>(_mm_movemask_epi8(matches));
        if (mask != 0) {
            return i + __builtin_ctz(mask);
        }
    }
    return i + byte_search_scalar<true>(val, array + i, size - i);
}

arch_target("sse2") static size_t byte_search_ne_sse2(uint8_t val, const int8_t* array, size_t size) {
    constexpr size_t vector_size = 16;
    const auto needle = _mm_set1_epi8(static_cast<char>(val));
    size_t i = 0;
    for (; i + vector_size <= size; i += vector_size) {
        const auto chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(array + i));
        const auto matches = _mm_cmpeq_epi8(chunk, needle);
        const auto mask = static_cast<unsigned>(_mm_movemask_epi8(matches)) ^ 0xffffu;
        if (mask != 0) {
            return i + __builtin_ctz(mask);
        }
    }
    return i + byte_search_scalar<false>(val, array + i, size - i);
}

arch_target("sse2") size_t byte_search_eq_impl(uint8_t val, const int8_t* array, size_t size) {
    return byte_search_eq_sse2(val, array, size);
}

arch_target("sse2") size_t byte_search_ne_impl(uint8_t val, const int8_t* array, size_t size) {
    return byte_search_ne_sse2(val, array, size);
}

arch_target("avx2") static size_t byte_search_eq_avx2(uint8_t val, const int8_t* array, size_t size) {
    constexpr size_t vector_size = 32;
    constexpr size_t tail_vector_size = 16;
    const auto needle = _mm256_set1_epi8(static_cast<char>(val));
    size_t i = 0;
    for (; i + 2 * vector_size <= size; i += 2 * vector_size) {
        const auto chunk0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(array + i));
        const auto matches0 = _mm256_cmpeq_epi8(chunk0, needle);
        const auto mask0 = static_cast<unsigned>(_mm256_movemask_epi8(matches0));
        if (mask0 != 0) {
            return i + __builtin_ctz(mask0);
        }

        const auto chunk1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(array + i + vector_size));
        const auto matches1 = _mm256_cmpeq_epi8(chunk1, needle);
        const auto mask1 = static_cast<unsigned>(_mm256_movemask_epi8(matches1));
        if (mask1 != 0) {
            return i + vector_size + __builtin_ctz(mask1);
        }
    }
    for (; i + vector_size <= size; i += vector_size) {
        const auto chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(array + i));
        const auto matches = _mm256_cmpeq_epi8(chunk, needle);
        const auto mask = static_cast<unsigned>(_mm256_movemask_epi8(matches));
        if (mask != 0) {
            return i + __builtin_ctz(mask);
        }
    }
    if (i + tail_vector_size <= size) {
        const auto tail_needle = _mm256_castsi256_si128(needle);
        const auto chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(array + i));
        const auto matches = _mm_cmpeq_epi8(chunk, tail_needle);
        const auto mask = static_cast<unsigned>(_mm_movemask_epi8(matches));
        if (mask != 0) {
            return i + __builtin_ctz(mask);
        }
        i += tail_vector_size;
    }
    return i + byte_search_scalar<true>(val, array + i, size - i);
}

arch_target("avx2") static size_t byte_search_ne_avx2(uint8_t val, const int8_t* array, size_t size) {
    constexpr size_t vector_size = 32;
    constexpr size_t tail_vector_size = 16;
    const auto needle = _mm256_set1_epi8(static_cast<char>(val));
    size_t i = 0;
    for (; i + 2 * vector_size <= size; i += 2 * vector_size) {
        const auto chunk0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(array + i));
        const auto matches0 = _mm256_cmpeq_epi8(chunk0, needle);
        const auto mask0 = static_cast<unsigned>(_mm256_movemask_epi8(matches0)) ^ 0xffffffffu;
        if (mask0 != 0) {
            return i + __builtin_ctz(mask0);
        }

        const auto chunk1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(array + i + vector_size));
        const auto matches1 = _mm256_cmpeq_epi8(chunk1, needle);
        const auto mask1 = static_cast<unsigned>(_mm256_movemask_epi8(matches1)) ^ 0xffffffffu;
        if (mask1 != 0) {
            return i + vector_size + __builtin_ctz(mask1);
        }
    }
    for (; i + vector_size <= size; i += vector_size) {
        const auto chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(array + i));
        const auto matches = _mm256_cmpeq_epi8(chunk, needle);
        const auto mask = static_cast<unsigned>(_mm256_movemask_epi8(matches)) ^ 0xffffffffu;
        if (mask != 0) {
            return i + __builtin_ctz(mask);
        }
    }
    if (i + tail_vector_size <= size) {
        const auto tail_needle = _mm256_castsi256_si128(needle);
        const auto chunk = _mm_loadu_si128(reinterpret_cast<const __m128i*>(array + i));
        const auto matches = _mm_cmpeq_epi8(chunk, tail_needle);
        const auto mask = static_cast<unsigned>(_mm_movemask_epi8(matches)) ^ 0xffffu;
        if (mask != 0) {
            return i + __builtin_ctz(mask);
        }
        i += tail_vector_size;
    }
    return i + byte_search_scalar<false>(val, array + i, size - i);
}

arch_target("avx2") size_t byte_search_eq_impl(uint8_t val, const int8_t* array, size_t size) {
    return byte_search_eq_avx2(val, array, size);
}

arch_target("avx2") size_t byte_search_ne_impl(uint8_t val, const int8_t* array, size_t size) {
    return byte_search_ne_avx2(val, array, size);
}

arch_target("avx512f,avx512bw") static size_t byte_search_eq_avx512(uint8_t val, const int8_t* array, size_t size) {
    constexpr size_t vector_size = 64;
    const auto needle = _mm512_set1_epi8(static_cast<char>(val));
    size_t i = 0;
    for (; i + 2 * vector_size <= size; i += 2 * vector_size) {
        const auto chunk0 = _mm512_loadu_si512(reinterpret_cast<const void*>(array + i));
        const auto mask0 = static_cast<uint64_t>(_mm512_cmpeq_epi8_mask(chunk0, needle));
        if (mask0 != 0) {
            return i + std::countr_zero(mask0);
        }

        const auto chunk1 = _mm512_loadu_si512(reinterpret_cast<const void*>(array + i + vector_size));
        const auto mask1 = static_cast<uint64_t>(_mm512_cmpeq_epi8_mask(chunk1, needle));
        if (mask1 != 0) {
            return i + vector_size + std::countr_zero(mask1);
        }
    }
    for (; i + vector_size <= size; i += vector_size) {
        const auto chunk = _mm512_loadu_si512(reinterpret_cast<const void*>(array + i));
        const auto mask = static_cast<uint64_t>(_mm512_cmpeq_epi8_mask(chunk, needle));
        if (mask != 0) {
            return i + std::countr_zero(mask);
        }
    }
    return i + byte_search_scalar<true>(val, array + i, size - i);
}

arch_target("avx512f,avx512bw") static size_t byte_search_ne_avx512(uint8_t val, const int8_t* array, size_t size) {
    constexpr size_t vector_size = 64;
    constexpr uint64_t lane_mask = ~uint64_t(0);
    const auto needle = _mm512_set1_epi8(static_cast<char>(val));
    size_t i = 0;
    for (; i + 2 * vector_size <= size; i += 2 * vector_size) {
        const auto chunk0 = _mm512_loadu_si512(reinterpret_cast<const void*>(array + i));
        const auto mask0 = static_cast<uint64_t>(_mm512_cmpeq_epi8_mask(chunk0, needle)) ^ lane_mask;
        if (mask0 != 0) {
            return i + std::countr_zero(mask0);
        }

        const auto chunk1 = _mm512_loadu_si512(reinterpret_cast<const void*>(array + i + vector_size));
        const auto mask1 = static_cast<uint64_t>(_mm512_cmpeq_epi8_mask(chunk1, needle)) ^ lane_mask;
        if (mask1 != 0) {
            return i + vector_size + std::countr_zero(mask1);
        }
    }
    for (; i + vector_size <= size; i += vector_size) {
        const auto chunk = _mm512_loadu_si512(reinterpret_cast<const void*>(array + i));
        const auto mask = static_cast<uint64_t>(_mm512_cmpeq_epi8_mask(chunk, needle)) ^ lane_mask;
        if (mask != 0) {
            return i + std::countr_zero(mask);
        }
    }
    return i + byte_search_scalar<false>(val, array + i, size - i);
}

#endif

#if defined(__aarch64__)
static size_t byte_search_eq_neon(uint8_t val, const int8_t* array, size_t size) {
    constexpr size_t vector_size = 16;
    const auto needle = vdupq_n_u8(val);
    const auto* bytes = reinterpret_cast<const uint8_t*>(array);
    size_t i = 0;
    for (; i + vector_size <= size; i += vector_size) {
        const auto chunk = vld1q_u8(bytes + i);
        const auto matches = vceqq_u8(chunk, needle);
        if (vmaxvq_u8(matches) != 0) {
            return i + byte_search_scalar<true>(val, array + i, vector_size);
        }
    }
    return i + byte_search_scalar<true>(val, array + i, size - i);
}

static size_t byte_search_ne_neon(uint8_t val, const int8_t* array, size_t size) {
    constexpr size_t vector_size = 16;
    const auto needle = vdupq_n_u8(val);
    const auto* bytes = reinterpret_cast<const uint8_t*>(array);
    size_t i = 0;
    for (; i + vector_size <= size; i += vector_size) {
        const auto chunk = vld1q_u8(bytes + i);
        const auto matches = vceqq_u8(chunk, needle);
        if (vminvq_u8(matches) != 0xff) {
            return i + byte_search_scalar<false>(val, array + i, vector_size);
        }
    }
    return i + byte_search_scalar<false>(val, array + i, size - i);
}
#endif

int array_search_gt(int64_t val, const int64_t* array, const int capacity, const int size) {
    return array_search_gt_impl(val, array, capacity, size);
}

unsigned array_search_16_eq(uint8_t val, const uint8_t* arr) {
    return array_search_16_eq_impl(val, arr);
}

unsigned array_search_32_eq(uint8_t val, const uint8_t* array) {
    return array_search_32_eq_impl(val, array);
}

unsigned array_search_x32_eq(uint8_t val, const uint8_t* array, int nr) {
    return array_search_x32_eq_impl(val, array, nr);
}

size_t byte_search_eq(uint8_t val, const int8_t* array, size_t size, db::simd_optimization_mode mode) {
    switch (mode) {
    case db::simd_optimization_mode::automatic:
#if defined(__aarch64__)
        return byte_search_eq_neon(val, array, size);
#else
        return byte_search_eq_impl(val, array, size);
#endif
    case db::simd_optimization_mode::off:
        return byte_search_scalar<true>(val, array, size);
#ifdef __x86_64__
    case db::simd_optimization_mode::sse:
        if (cpu_supports_sse2()) {
            return byte_search_eq_sse2(val, array, size);
        }
        break;
    case db::simd_optimization_mode::avx2:
        if (cpu_supports_avx2()) {
            return byte_search_eq_avx2(val, array, size);
        }
        break;
    case db::simd_optimization_mode::avx512:
        if (cpu_supports_avx512()) {
            return byte_search_eq_avx512(val, array, size);
        }
        break;
#elif defined(__aarch64__)
    case db::simd_optimization_mode::neon:
        return byte_search_eq_neon(val, array, size);
    case db::simd_optimization_mode::sse:
    case db::simd_optimization_mode::avx2:
    case db::simd_optimization_mode::avx512:
        break;
#else
    case db::simd_optimization_mode::sse:
    case db::simd_optimization_mode::avx2:
    case db::simd_optimization_mode::avx512:
        break;
#endif
#if !defined(__aarch64__)
    case db::simd_optimization_mode::neon:
#endif
    case db::simd_optimization_mode::sve:
        break;
    }
    return byte_search_scalar<true>(val, array, size);
}

size_t byte_search_ne(uint8_t val, const int8_t* array, size_t size, db::simd_optimization_mode mode) {
    switch (mode) {
    case db::simd_optimization_mode::automatic:
#if defined(__aarch64__)
        return byte_search_ne_neon(val, array, size);
#else
        return byte_search_ne_impl(val, array, size);
#endif
    case db::simd_optimization_mode::off:
        return byte_search_scalar<false>(val, array, size);
#ifdef __x86_64__
    case db::simd_optimization_mode::sse:
        if (cpu_supports_sse2()) {
            return byte_search_ne_sse2(val, array, size);
        }
        break;
    case db::simd_optimization_mode::avx2:
        if (cpu_supports_avx2()) {
            return byte_search_ne_avx2(val, array, size);
        }
        break;
    case db::simd_optimization_mode::avx512:
        if (cpu_supports_avx512()) {
            return byte_search_ne_avx512(val, array, size);
        }
        break;
#elif defined(__aarch64__)
    case db::simd_optimization_mode::neon:
        return byte_search_ne_neon(val, array, size);
    case db::simd_optimization_mode::sse:
    case db::simd_optimization_mode::avx2:
    case db::simd_optimization_mode::avx512:
        break;
#else
    case db::simd_optimization_mode::sse:
    case db::simd_optimization_mode::avx2:
    case db::simd_optimization_mode::avx512:
        break;
#endif
#if !defined(__aarch64__)
    case db::simd_optimization_mode::neon:
#endif
    case db::simd_optimization_mode::sve:
        break;
    }
    return byte_search_scalar<false>(val, array, size);
}

}
