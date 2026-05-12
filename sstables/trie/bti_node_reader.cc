/*
 * Copyright (C) 2024-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

#include "bti_node_reader.hh"
#include "bti_node_type.hh"

#include <algorithm>
#include <bit>
#include <cstdint>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#if defined(__aarch64__)
#include <arm_neon.h>
#endif

#include "db/config.hh"

namespace sstables::trie {

namespace {

static thread_local db::simd_optimization_mode bti_dense_node_simd_optimization_mode = db::simd_optimization_mode::automatic;
static thread_local db::simd_optimization_mode bti_sparse_node_simd_optimization_mode = db::simd_optimization_mode::automatic;

static int dense_first_nonzero_offset_scalar(const_bytes offsets, int first, int last, int bits_per_pointer) {
    for (int idx = first; idx < last; ++idx) {
        if (read_offset(offsets, idx, bits_per_pointer) != 0) {
            return idx;
        }
    }
    return last;
}

static int dense_last_nonzero_offset_scalar(const_bytes offsets, int first, int last, int bits_per_pointer) {
    for (int idx = last - 1; idx >= first; --idx) {
        if (read_offset(offsets, idx, bits_per_pointer) != 0) {
            return idx;
        }
    }
    return first - 1;
}

[[gnu::always_inline]]
static inline int sparse_transition_lower_bound_scalar(const std::byte* transitions, int n_children, std::byte key) {
    return std::lower_bound(transitions, transitions + n_children, key) - transitions;
}

[[gnu::always_inline]]
static inline int sparse_transition_lower_bound_linear(const std::byte* transitions, int n_children, std::byte key) {
    int idx = 0;
    while (idx < n_children && transitions[idx] < key) {
        ++idx;
    }
    return idx;
}

#if defined(__x86_64__) || defined(__i386__)
template <int BytesPerOffset>
[[gnu::target("sse2")]]
static int dense_first_nonzero_offset_sse2(const_bytes offsets, int first, int last) {
    constexpr int lane_bytes = 16;
    constexpr int offsets_per_lane = lane_bytes / BytesPerOffset;
    constexpr uint32_t all_bytes = (uint32_t(1) << lane_bytes) - 1;

    const auto zero = _mm_setzero_si128();
    int idx = first;
    for (; idx + offsets_per_lane <= last; idx += offsets_per_lane) {
        const auto v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(offsets.data() + idx * BytesPerOffset));
        const auto eq = _mm_cmpeq_epi8(v, zero);
        const auto nonzero_bytes = (~uint32_t(_mm_movemask_epi8(eq))) & all_bytes;
        if (nonzero_bytes != 0) {
            return idx + std::countr_zero(nonzero_bytes) / BytesPerOffset;
        }
    }
    return dense_first_nonzero_offset_scalar(offsets, idx, last, BytesPerOffset * 8);
}

template <int BytesPerOffset>
[[gnu::target("sse2")]]
static int dense_last_nonzero_offset_sse2(const_bytes offsets, int first, int last) {
    constexpr int lane_bytes = 16;
    constexpr int offsets_per_lane = lane_bytes / BytesPerOffset;
    constexpr uint32_t all_bytes = (uint32_t(1) << lane_bytes) - 1;

    const auto zero = _mm_setzero_si128();
    int idx = last;
    for (; idx - offsets_per_lane >= first; idx -= offsets_per_lane) {
        const int block_start = idx - offsets_per_lane;
        const auto v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(offsets.data() + block_start * BytesPerOffset));
        const auto eq = _mm_cmpeq_epi8(v, zero);
        const auto nonzero_bytes = (~uint32_t(_mm_movemask_epi8(eq))) & all_bytes;
        if (nonzero_bytes != 0) {
            return block_start + (std::bit_width(nonzero_bytes) - 1) / BytesPerOffset;
        }
    }
    return dense_last_nonzero_offset_scalar(offsets, first, idx, BytesPerOffset * 8);
}

template <int BytesPerOffset>
[[gnu::target("avx2")]]
static int dense_first_nonzero_offset_avx2(const_bytes offsets, int first, int last) {
    constexpr int lane_bytes = 32;
    constexpr int offsets_per_lane = lane_bytes / BytesPerOffset;
    constexpr uint32_t all_bytes = ~uint32_t(0);

    const auto zero = _mm256_setzero_si256();
    int idx = first;
    for (; idx + offsets_per_lane <= last; idx += offsets_per_lane) {
        const auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(offsets.data() + idx * BytesPerOffset));
        const auto eq = _mm256_cmpeq_epi8(v, zero);
        const auto nonzero_bytes = (~uint32_t(_mm256_movemask_epi8(eq))) & all_bytes;
        if (nonzero_bytes != 0) {
            return idx + std::countr_zero(nonzero_bytes) / BytesPerOffset;
        }
    }
    if constexpr (BytesPerOffset <= 2) {
        if (last - idx >= 16 / BytesPerOffset) {
            return dense_first_nonzero_offset_sse2<BytesPerOffset>(offsets, idx, last);
        }
    }
    return dense_first_nonzero_offset_scalar(offsets, idx, last, BytesPerOffset * 8);
}

template <int BytesPerOffset>
[[gnu::target("avx2")]]
static int dense_last_nonzero_offset_avx2(const_bytes offsets, int first, int last) {
    constexpr int lane_bytes = 32;
    constexpr int offsets_per_lane = lane_bytes / BytesPerOffset;
    constexpr uint32_t all_bytes = ~uint32_t(0);

    const auto zero = _mm256_setzero_si256();
    int idx = last;
    for (; idx - offsets_per_lane >= first; idx -= offsets_per_lane) {
        const int block_start = idx - offsets_per_lane;
        const auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(offsets.data() + block_start * BytesPerOffset));
        const auto eq = _mm256_cmpeq_epi8(v, zero);
        const auto nonzero_bytes = (~uint32_t(_mm256_movemask_epi8(eq))) & all_bytes;
        if (nonzero_bytes != 0) {
            return block_start + (std::bit_width(nonzero_bytes) - 1) / BytesPerOffset;
        }
    }
    if constexpr (BytesPerOffset <= 2) {
        if (idx - first >= 16 / BytesPerOffset) {
            return dense_last_nonzero_offset_sse2<BytesPerOffset>(offsets, first, idx);
        }
    }
    return dense_last_nonzero_offset_scalar(offsets, first, idx, BytesPerOffset * 8);
}

template <int BytesPerOffset>
[[gnu::target("avx512f,avx512bw")]]
static int dense_first_nonzero_offset_avx512(const_bytes offsets, int first, int last) {
    constexpr int lane_bytes = 64;
    constexpr int offsets_per_lane = lane_bytes / BytesPerOffset;
    constexpr uint64_t all_bytes = ~uint64_t(0);

    const auto zero = _mm512_setzero_si512();
    int idx = first;
    for (; idx + offsets_per_lane <= last; idx += offsets_per_lane) {
        const auto v = _mm512_loadu_si512(reinterpret_cast<const void*>(offsets.data() + idx * BytesPerOffset));
        const auto nonzero_bytes = (~uint64_t(_mm512_cmpeq_epi8_mask(v, zero))) & all_bytes;
        if (nonzero_bytes != 0) {
            return idx + std::countr_zero(nonzero_bytes) / BytesPerOffset;
        }
    }
    return dense_first_nonzero_offset_scalar(offsets, idx, last, BytesPerOffset * 8);
}

template <int BytesPerOffset>
[[gnu::target("avx512f,avx512bw")]]
static int dense_last_nonzero_offset_avx512(const_bytes offsets, int first, int last) {
    constexpr int lane_bytes = 64;
    constexpr int offsets_per_lane = lane_bytes / BytesPerOffset;
    constexpr uint64_t all_bytes = ~uint64_t(0);

    const auto zero = _mm512_setzero_si512();
    int idx = last;
    for (; idx - offsets_per_lane >= first; idx -= offsets_per_lane) {
        const int block_start = idx - offsets_per_lane;
        const auto v = _mm512_loadu_si512(reinterpret_cast<const void*>(offsets.data() + block_start * BytesPerOffset));
        const auto nonzero_bytes = (~uint64_t(_mm512_cmpeq_epi8_mask(v, zero))) & all_bytes;
        if (nonzero_bytes != 0) {
            return block_start + (std::bit_width(nonzero_bytes) - 1) / BytesPerOffset;
        }
    }
    return dense_last_nonzero_offset_scalar(offsets, first, idx, BytesPerOffset * 8);
}

static bool cpu_supports_avx512() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
}

[[gnu::target("sse2")]]
static int sparse_transition_lower_bound_sse2(const std::byte* transitions, int n_children, std::byte key) {
    constexpr int lane_bytes = 16;
    constexpr uint32_t lane_mask = (uint32_t(1) << lane_bytes) - 1;

    const auto sign_bit = _mm_set1_epi8(char(0x80));
    const auto adjusted_key = _mm_set1_epi8(char(uint8_t(key) ^ 0x80));
    int idx = 0;
    for (; idx + lane_bytes <= n_children; idx += lane_bytes) {
        const auto v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(transitions + idx));
        const auto adjusted_v = _mm_xor_si128(v, sign_bit);
        const auto less_than_key = _mm_cmpgt_epi8(adjusted_key, adjusted_v);
        const auto ge_key_mask = (~uint32_t(_mm_movemask_epi8(less_than_key))) & lane_mask;
        if (ge_key_mask != 0) {
            return idx + std::countr_zero(ge_key_mask);
        }
    }
    return idx + sparse_transition_lower_bound_scalar(transitions + idx, n_children - idx, key);
}

[[gnu::target("avx2")]]
static int sparse_transition_lower_bound_avx2(const std::byte* transitions, int n_children, std::byte key) {
    constexpr int lane_bytes = 32;

    const auto key_vector = _mm256_set1_epi8(char(uint8_t(key)));
    int idx = 0;
    for (; idx + lane_bytes <= n_children; idx += lane_bytes) {
        const auto v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(transitions + idx));
        const auto ge_key = _mm256_cmpeq_epi8(_mm256_min_epu8(v, key_vector), key_vector);
        const auto ge_key_mask = uint32_t(_mm256_movemask_epi8(ge_key));
        if (ge_key_mask != 0) {
            return idx + std::countr_zero(ge_key_mask);
        }
    }
    if (n_children - idx >= 16) {
        return idx + sparse_transition_lower_bound_sse2(transitions + idx, n_children - idx, key);
    }
    return idx + sparse_transition_lower_bound_scalar(transitions + idx, n_children - idx, key);
}

[[gnu::target("avx512f,avx512bw")]]
static int sparse_transition_lower_bound_avx512(const std::byte* transitions, int n_children, std::byte key) {
    constexpr int lane_bytes = 64;

    const auto key_vector = _mm512_set1_epi8(char(uint8_t(key)));
    int idx = 0;
    for (; idx + lane_bytes <= n_children; idx += lane_bytes) {
        const auto v = _mm512_loadu_si512(reinterpret_cast<const void*>(transitions + idx));
        const auto ge_key = _mm512_cmpeq_epi8_mask(_mm512_min_epu8(v, key_vector), key_vector);
        if (ge_key != 0) {
            return idx + std::countr_zero(uint64_t(ge_key));
        }
    }
    return idx + sparse_transition_lower_bound_scalar(transitions + idx, n_children - idx, key);
}

static bool cpu_supports_avx2() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
}

static bool cpu_supports_sse2() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("sse2");
}
#endif

#if defined(__aarch64__)
template <int BytesPerOffset>
static int dense_first_nonzero_offset_neon(const_bytes offsets, int first, int last) {
    constexpr int lane_bytes = 16;
    constexpr int offsets_per_lane = lane_bytes / BytesPerOffset;

    const auto zero = vdupq_n_u8(0);
    const auto* bytes = reinterpret_cast<const uint8_t*>(offsets.data());
    int idx = first;
    for (; idx + offsets_per_lane <= last; idx += offsets_per_lane) {
        const auto v = vld1q_u8(bytes + idx * BytesPerOffset);
        const auto eq = vceqq_u8(v, zero);
        if (vminvq_u8(eq) != 0xff) {
            return dense_first_nonzero_offset_scalar(offsets, idx, idx + offsets_per_lane, BytesPerOffset * 8);
        }
    }
    return dense_first_nonzero_offset_scalar(offsets, idx, last, BytesPerOffset * 8);
}

template <int BytesPerOffset>
static int dense_last_nonzero_offset_neon(const_bytes offsets, int first, int last) {
    constexpr int lane_bytes = 16;
    constexpr int offsets_per_lane = lane_bytes / BytesPerOffset;

    const auto zero = vdupq_n_u8(0);
    const auto* bytes = reinterpret_cast<const uint8_t*>(offsets.data());
    int idx = last;
    for (; idx - offsets_per_lane >= first; idx -= offsets_per_lane) {
        const int block_start = idx - offsets_per_lane;
        const auto v = vld1q_u8(bytes + block_start * BytesPerOffset);
        const auto eq = vceqq_u8(v, zero);
        if (vminvq_u8(eq) != 0xff) {
            return dense_last_nonzero_offset_scalar(offsets, block_start, idx, BytesPerOffset * 8);
        }
    }
    return dense_last_nonzero_offset_scalar(offsets, first, idx, BytesPerOffset * 8);
}

static int sparse_transition_lower_bound_neon(const std::byte* transitions, int n_children, std::byte key) {
    constexpr int lane_bytes = 16;

    const auto key_vector = vdupq_n_u8(uint8_t(key));
    const auto* bytes = reinterpret_cast<const uint8_t*>(transitions);
    int idx = 0;
    for (; idx + lane_bytes <= n_children; idx += lane_bytes) {
        const auto v = vld1q_u8(bytes + idx);
        const auto ge_key = vceqq_u8(vminq_u8(v, key_vector), key_vector);
        if (vmaxvq_u8(ge_key) != 0) {
            return idx + sparse_transition_lower_bound_scalar(transitions + idx, lane_bytes, key);
        }
    }
    return idx + sparse_transition_lower_bound_scalar(transitions + idx, n_children - idx, key);
}
#endif

template <int BytesPerOffset>
static int dense_first_nonzero_offset_fixed(const_bytes offsets, int first, int last, db::simd_optimization_mode mode) {
#if defined(__x86_64__) || defined(__i386__)
    if (mode == db::simd_optimization_mode::off) {
        return dense_first_nonzero_offset_scalar(offsets, first, last, BytesPerOffset * 8);
    }

    static const bool has_avx512 = cpu_supports_avx512();
    if ((mode == db::simd_optimization_mode::automatic || mode == db::simd_optimization_mode::avx512) && has_avx512) {
        return dense_first_nonzero_offset_avx512<BytesPerOffset>(offsets, first, last);
    }
    static const bool has_avx2 = cpu_supports_avx2();
    if ((mode == db::simd_optimization_mode::automatic || mode == db::simd_optimization_mode::avx2) && has_avx2) {
        return dense_first_nonzero_offset_avx2<BytesPerOffset>(offsets, first, last);
    }
    static const bool has_sse2 = cpu_supports_sse2();
    if ((mode == db::simd_optimization_mode::automatic || mode == db::simd_optimization_mode::sse) && has_sse2) {
        return dense_first_nonzero_offset_sse2<BytesPerOffset>(offsets, first, last);
    }
#elif defined(__aarch64__)
    if (mode == db::simd_optimization_mode::automatic || mode == db::simd_optimization_mode::neon) {
        return dense_first_nonzero_offset_neon<BytesPerOffset>(offsets, first, last);
    }
#endif
    return dense_first_nonzero_offset_scalar(offsets, first, last, BytesPerOffset * 8);
}

template <int BytesPerOffset>
static int dense_last_nonzero_offset_fixed(const_bytes offsets, int first, int last, db::simd_optimization_mode mode) {
#if defined(__x86_64__) || defined(__i386__)
    if (mode == db::simd_optimization_mode::off) {
        return dense_last_nonzero_offset_scalar(offsets, first, last, BytesPerOffset * 8);
    }

    static const bool has_avx512 = cpu_supports_avx512();
    if ((mode == db::simd_optimization_mode::automatic || mode == db::simd_optimization_mode::avx512) && has_avx512) {
        return dense_last_nonzero_offset_avx512<BytesPerOffset>(offsets, first, last);
    }
    static const bool has_avx2 = cpu_supports_avx2();
    if ((mode == db::simd_optimization_mode::automatic || mode == db::simd_optimization_mode::avx2) && has_avx2) {
        return dense_last_nonzero_offset_avx2<BytesPerOffset>(offsets, first, last);
    }
    static const bool has_sse2 = cpu_supports_sse2();
    if ((mode == db::simd_optimization_mode::automatic || mode == db::simd_optimization_mode::sse) && has_sse2) {
        return dense_last_nonzero_offset_sse2<BytesPerOffset>(offsets, first, last);
    }
#elif defined(__aarch64__)
    if (mode == db::simd_optimization_mode::automatic || mode == db::simd_optimization_mode::neon) {
        return dense_last_nonzero_offset_neon<BytesPerOffset>(offsets, first, last);
    }
#endif
    return dense_last_nonzero_offset_scalar(offsets, first, last, BytesPerOffset * 8);
}

static int dense_first_nonzero_offset(const_bytes offsets, int first, int last, int bits_per_pointer) {
    switch (bits_per_pointer) {
    case 8:
        return dense_first_nonzero_offset_fixed<1>(offsets, first, last, bti_dense_node_simd_optimization_mode);
    case 16:
        return dense_first_nonzero_offset_fixed<2>(offsets, first, last, bti_dense_node_simd_optimization_mode);
    case 32:
        return dense_first_nonzero_offset_fixed<4>(offsets, first, last, bti_dense_node_simd_optimization_mode);
    case 64:
        return dense_first_nonzero_offset_fixed<8>(offsets, first, last, bti_dense_node_simd_optimization_mode);
    default:
        return dense_first_nonzero_offset_scalar(offsets, first, last, bits_per_pointer);
    }
}

static int dense_last_nonzero_offset(const_bytes offsets, int first, int last, int bits_per_pointer) {
    switch (bits_per_pointer) {
    case 8:
        return dense_last_nonzero_offset_fixed<1>(offsets, first, last, bti_dense_node_simd_optimization_mode);
    case 16:
        return dense_last_nonzero_offset_fixed<2>(offsets, first, last, bti_dense_node_simd_optimization_mode);
    case 32:
        return dense_last_nonzero_offset_fixed<4>(offsets, first, last, bti_dense_node_simd_optimization_mode);
    case 64:
        return dense_last_nonzero_offset_fixed<8>(offsets, first, last, bti_dense_node_simd_optimization_mode);
    default:
        return dense_last_nonzero_offset_scalar(offsets, first, last, bits_per_pointer);
    }
}

[[gnu::noinline]]
static int sparse_transition_lower_bound_simd(const std::byte* transitions, int n_children, std::byte key, db::simd_optimization_mode mode) {
#if defined(__x86_64__) || defined(__i386__)
    if (mode == db::simd_optimization_mode::off) {
        return sparse_transition_lower_bound_scalar(transitions, n_children, key);
    }

    if ((mode == db::simd_optimization_mode::avx512 || (mode == db::simd_optimization_mode::automatic && n_children <= 128)) &&
            n_children >= 64) {
        static const bool has_avx512 = cpu_supports_avx512();
        if (has_avx512) {
            return sparse_transition_lower_bound_avx512(transitions, n_children, key);
        }
    }
    if ((mode == db::simd_optimization_mode::automatic || mode == db::simd_optimization_mode::avx2) && n_children >= 32) {
        static const bool has_avx2 = cpu_supports_avx2();
        if (has_avx2) {
            return sparse_transition_lower_bound_avx2(transitions, n_children, key);
        }
    }
    static const bool has_sse2 = cpu_supports_sse2();
    if ((mode == db::simd_optimization_mode::automatic || mode == db::simd_optimization_mode::sse) && has_sse2) {
        return sparse_transition_lower_bound_sse2(transitions, n_children, key);
    }
#elif defined(__aarch64__)
    if (mode == db::simd_optimization_mode::automatic || mode == db::simd_optimization_mode::neon) {
        return sparse_transition_lower_bound_neon(transitions, n_children, key);
    }
#endif
    return sparse_transition_lower_bound_scalar(transitions, n_children, key);
}

} // anonymous namespace

void set_bti_dense_node_simd_optimization_mode(db::simd_optimization_mode mode) noexcept {
    bti_dense_node_simd_optimization_mode = mode;
}

db::simd_optimization_mode get_bti_dense_node_simd_optimization_mode() noexcept {
    return bti_dense_node_simd_optimization_mode;
}

void set_bti_sparse_node_simd_optimization_mode(db::simd_optimization_mode mode) noexcept {
    bti_sparse_node_simd_optimization_mode = mode;
}

db::simd_optimization_mode get_bti_sparse_node_simd_optimization_mode() noexcept {
    return bti_sparse_node_simd_optimization_mode;
}

get_child_result bti_get_child(uint64_t pos, const_bytes sp, int child_idx, bool forward) {
    auto type = uint8_t(sp[0]) >> 4;
    trie::get_child_result result;
    auto single = [&](uint64_t offset) {
        result.offset = offset;
        result.idx = 0;
        return result;
    };
    auto sparse = [&] [[gnu::always_inline]] (int type) {
        auto bpp = bits_per_pointer_arr[type];
        result.idx = child_idx;
        result.offset = read_offset(sp.subspan(2 + int(sp[1])), child_idx, bpp);
        return result;
    };
    auto dense = [&] [[gnu::always_inline]] (int type) {
        auto bpp = bits_per_pointer_arr[type];
        auto dense_span = uint64_t(sp[2]) + 1;
        const int start_idx = child_idx;
        const int end_idx = int(dense_span);
        if (start_idx >= 0 && start_idx < end_idx) {
            auto offsets = sp.subspan(3);
            auto idx = forward
                ? dense_first_nonzero_offset(offsets, start_idx, end_idx, bpp)
                : dense_last_nonzero_offset(offsets, 0, start_idx + 1, bpp);
            if (idx >= 0 && idx < end_idx) {
                result.idx = idx;
                result.offset = read_offset(offsets, idx, bpp);
                return result;
            }
        }
        [[unlikely]] sstables::on_bti_parse_error(pos);
    };
    switch (type) {
    case PAYLOAD_ONLY:
        [[unlikely]] sstables::on_bti_parse_error(pos);
    case SINGLE_NOPAYLOAD_4:
        return single(uint64_t(sp[0]) & 0xf);
    case SINGLE_NOPAYLOAD_12:
        return single((uint64_t(sp[0]) & 0xf) << 8 | uint64_t(sp[1]));
    case SINGLE_8:
        return single(uint64_t(sp[2]));
    case SINGLE_16:
        return single(uint64_t(sp[2]) << 8 | uint64_t(sp[3]));
    case SPARSE_8:
        return sparse(type);
    case SPARSE_12:
        return sparse(type);
    case SPARSE_16:
        return sparse(type);
    case SPARSE_24:
        return sparse(type);
    case SPARSE_40:
        return sparse(type);
    case DENSE_12:
        return dense(type);
    case DENSE_16:
        return dense(type);
    case DENSE_24:
        return dense(type);
    case DENSE_32:
        return dense(type);
    case DENSE_40:
        return dense(type);
    case LONG_DENSE:
        return dense(type);
    }
    [[unlikely]] sstables::on_bti_parse_error(pos);
}

std::byte bti_get_child_transition(uint64_t pos, const_bytes raw, int idx) {
    auto type = uint8_t(raw[0]) >> 4;
    switch (type) {
    case PAYLOAD_ONLY:
        abort();
    case SINGLE_NOPAYLOAD_4:
        return raw[1];
    case SINGLE_8:
        return raw[1];
    case SINGLE_NOPAYLOAD_12:
        return raw[2];
    case SINGLE_16:
        return raw[1];
    case SPARSE_8:
    case SPARSE_12:
    case SPARSE_16:
    case SPARSE_24:
    case SPARSE_40:
        return raw[2 + idx];
    case DENSE_12:
    case DENSE_16:
    case DENSE_24:
    case DENSE_32:
    case DENSE_40:
    case LONG_DENSE:
        return std::byte(uint8_t(raw[1]) + idx);
    }
    [[unlikely]] sstables::on_bti_parse_error(pos);
}

load_final_node_result bti_read_node(int64_t pos, const_bytes sp) {
    load_final_node_result result;
    auto type = uint8_t(sp[0]) >> 4;
    auto single = [&](uint8_t payload_bits) {
        result.n_children = 1;
        result.payload_bits = payload_bits;
        return result;
    };
    auto sparse = [&] [[gnu::always_inline]] (int type) {
        int n_children = int(sp[1]);
        result.n_children = n_children;
        result.payload_bits = uint8_t(sp[0]) & 0xf;
        return result;
    };
    auto dense = [&] [[gnu::always_inline]] (int type) {
        auto dense_span = uint64_t(sp[2]) + 1;
        result.n_children = dense_span;
        result.payload_bits = uint8_t(sp[0]) & 0xf;
        return result;
    };
    switch (type) {
    case PAYLOAD_ONLY:
        result.payload_bits = uint8_t(sp[0]) & 0xf;
        result.n_children = 0;
        return result;
    case SINGLE_NOPAYLOAD_4:
        return single(0);
    case SINGLE_NOPAYLOAD_12:
        return single(0);
    case SINGLE_8:
        return single(uint8_t(sp[0]) & 0xf);
    case SINGLE_16:
        return single(uint8_t(sp[0]) & 0xf);
    case SPARSE_8:
        return sparse(type);
    case SPARSE_12:
        return sparse(type);
    case SPARSE_16:
        return sparse(type);
    case SPARSE_24:
        return sparse(type);
    case SPARSE_40:
        return sparse(type);
    case DENSE_12:
        return dense(type);
    case DENSE_16:
        return dense(type);
    case DENSE_24:
        return dense(type);
    case DENSE_32:
        return dense(type);
    case DENSE_40:
        return dense(type);
    case LONG_DENSE:
        return dense(type);
    }
    [[unlikely]] sstables::on_bti_parse_error(pos);
}

const_bytes bti_get_payload(int64_t pos, const_bytes sp) {
    auto type = uint8_t(sp[0]) >> 4;
    switch (type) {
    case PAYLOAD_ONLY:
        return sp.subspan(1);
    case SINGLE_NOPAYLOAD_4:
    case SINGLE_NOPAYLOAD_12:
        return sp.subspan(1 + div_ceil(bits_per_pointer_arr[type], 8));
    case SINGLE_8:
    case SINGLE_16:
        return sp.subspan(2 + div_ceil(bits_per_pointer_arr[type], 8));
    case SPARSE_8:
    case SPARSE_12:
    case SPARSE_16:
    case SPARSE_24:
    case SPARSE_40: {
        auto n_children = uint8_t(sp[1]);
        return sp.subspan(2 + div_ceil(n_children * (8 + bits_per_pointer_arr[type]), 8));
    }
    case DENSE_12:
    case DENSE_16:
    case DENSE_24:
    case DENSE_32:
    case DENSE_40:
    case LONG_DENSE: {
        auto dense_span = uint8_t(sp[2]) + 1;
        return sp.subspan(3 + div_ceil(dense_span * bits_per_pointer_arr[type], 8));
    }
    }
    [[unlikely]] sstables::on_bti_parse_error(pos);
}

node_traverse_result bti_walk_down_along_key(int64_t pos, const_bytes sp, const_bytes key) {
    auto type = uint8_t(sp[0]) >> 4;
    trie::node_traverse_result result;
    result.body_pos = pos;
    result.traversed_key_bytes = 0;
    auto single = [&](std::byte edge, uint64_t offset, uint8_t payload_bits) {
        result.n_children = 1;
        result.payload_bits = payload_bits;
        if (key[0] <= edge) {
            result.found_idx = 0;
            result.found_byte = int(edge);
            result.child_offset = offset;
        } else {
            result.found_idx = 1;
            result.found_byte = -1;
            result.child_offset = -1;
        }
        return result;
    };
    auto sparse = [&] [[gnu::always_inline]] (int type) {
        int n_children = int(sp[1]);
        int idx;
        if (n_children < 16) [[likely]] {
            idx = sparse_transition_lower_bound_linear(&sp[2], n_children, key[0]);
        } else {
            idx = sparse_transition_lower_bound_simd(&sp[2], n_children, key[0], bti_sparse_node_simd_optimization_mode);
        }
        result.n_children = n_children;
        result.payload_bits = uint8_t(sp[0]) & 0xf;
        result.found_idx = idx;
        if (idx < n_children) {
            auto bpp = bits_per_pointer_arr[type];
            result.child_offset = read_offset(sp.subspan(2 + n_children), idx, bpp);
            result.found_byte = int(sp[2 + idx]);
        } else {
            result.child_offset = -1;
            result.found_byte = -1;
        }
        return result;
    };
    auto dense = [&] [[gnu::always_inline]] (int type) {
        auto start = int(sp[1]);
        auto idx = std::max<int>(0, int(key[0]) - start);
        auto dense_span = uint64_t(sp[2]) + 1;
        auto bpp = bits_per_pointer_arr[type];
        result.n_children = dense_span;
        result.payload_bits = uint8_t(sp[0]) & 0xf;
        auto offsets = sp.subspan(3);
        idx = dense_first_nonzero_offset(offsets, idx, int(dense_span), bpp);
        if (idx < int(dense_span)) {
            result.child_offset = read_offset(offsets, idx, bpp);
            result.found_idx = idx;
            result.found_byte = start + idx;
            return result;
        }
        result.found_idx = dense_span;
        result.child_offset = -1;
        result.found_byte = -1;
        return result;
    };
    switch (type) {
    case PAYLOAD_ONLY:
        result.payload_bits = uint8_t(sp[0]) & 0xf;
        result.n_children = 0;
        result.found_idx = 0;
        result.found_byte = -1;
        result.child_offset = -1;
        return result;
    case SINGLE_NOPAYLOAD_4:
        return single(sp[1], uint64_t(sp[0]) & 0xf, 0);
    case SINGLE_NOPAYLOAD_12:
        return single(sp[2], (uint64_t(sp[0]) & 0xf) << 8 | uint64_t(sp[1]), 0);
    case SINGLE_8:
        return single(sp[1], uint64_t(sp[2]), uint8_t(sp[0]) & 0xf);
    case SINGLE_16:
        return single(sp[1], uint64_t(sp[2]) << 8 | uint64_t(sp[3]), uint8_t(sp[0]) & 0xf);
    case SPARSE_8:
        return sparse(type);
    case SPARSE_12:
        return sparse(type);
    case SPARSE_16:
        return sparse(type);
    case SPARSE_24:
        return sparse(type);
    case SPARSE_40:
        return sparse(type);
    case DENSE_12:
        return dense(type);
    case DENSE_16:
        return dense(type);
    case DENSE_24:
        return dense(type);
    case DENSE_32:
        return dense(type);
    case DENSE_40:
        return dense(type);
    case LONG_DENSE:
        return dense(type);
    }
    [[unlikely]] sstables::on_bti_parse_error(pos);
}

void bti_walk_down_along_key_batch(int64_t pos, const_bytes sp, const_bytes key_bytes, std::span<node_traverse_result> results) {
    SCYLLA_ASSERT(key_bytes.size() == results.size());

    auto type = uint8_t(sp[0]) >> 4;
    switch (type) {
    case SPARSE_8:
    case SPARSE_12:
    case SPARSE_16:
    case SPARSE_24:
    case SPARSE_40:
        break;
    default:
        for (size_t i = 0; i < key_bytes.size(); ++i) {
            results[i] = bti_walk_down_along_key(pos, sp, key_bytes.subspan(i, 1));
        }
        return;
    }

    if (key_bytes.size() < 32 || !std::is_sorted(key_bytes.begin(), key_bytes.end())) {
        for (size_t i = 0; i < key_bytes.size(); ++i) {
            results[i] = bti_walk_down_along_key(pos, sp, key_bytes.subspan(i, 1));
        }
        return;
    }

    int n_children = int(sp[1]);
    const auto bpp = bits_per_pointer_arr[type];
    auto offsets = sp.subspan(2 + n_children);
    int idx = 0;
    for (size_t i = 0; i < key_bytes.size(); ++i) {
        auto key = key_bytes[i];
        while (idx < n_children && sp[2 + idx] < key) {
            ++idx;
        }
        auto& result = results[i];
        result.n_children = n_children;
        result.payload_bits = uint8_t(sp[0]) & 0xf;
        result.found_idx = idx;
        result.traversed_key_bytes = 0;
        result.body_pos = pos;
        if (idx < n_children) {
            result.child_offset = read_offset(offsets, idx, bpp);
            result.found_byte = int(sp[2 + idx]);
        } else {
            result.child_offset = -1;
            result.found_byte = -1;
        }
    }
}

node_traverse_sidemost_result bti_walk_down_leftmost_path(int64_t pos, const_bytes sp) {
    auto type = uint8_t(sp[0]) >> 4;
    trie::node_traverse_sidemost_result result;
    result.body_pos = pos;
    auto single = [&](uint64_t offset, uint8_t payload_bits) {
        result.n_children = 1;
        result.payload_bits = payload_bits;
        result.child_offset = offset;
        return result;
    };
    auto sparse = [&] [[gnu::always_inline]] (int type) {
        int n_children = int(sp[1]);
        auto bpp = bits_per_pointer_arr[type];
        result.n_children = n_children;
        result.payload_bits = uint8_t(sp[0]) & 0xf;
        result.child_offset = read_offset(sp.subspan(2 + n_children), 0, bpp);
        return result;
    };
    auto dense = [&] [[gnu::always_inline]] (int type) {
        auto dense_span = uint64_t(sp[2]) + 1;
        auto bpp = bits_per_pointer_arr[type];
        result.n_children = dense_span;
        result.payload_bits = uint8_t(sp[0]) & 0xf;
        result.child_offset = read_offset(sp.subspan(3), 0, bpp);
        return result;
    };
    switch (type) {
    case PAYLOAD_ONLY:
        result.payload_bits = uint8_t(sp[0]) & 0xf;
        result.n_children = 0;
        result.child_offset = -1;
        return result;
    case SINGLE_NOPAYLOAD_4:
        return single(uint64_t(sp[0]) & 0xf, 0);
    case SINGLE_NOPAYLOAD_12:
        return single((uint64_t(sp[0]) & 0xf) << 8 | uint64_t(sp[1]), 0);
    case SINGLE_8:
        return single(uint64_t(sp[2]), uint8_t(sp[0]) & 0xf);
    case SINGLE_16:
        return single(uint64_t(sp[2]) << 8 | uint64_t(sp[3]), uint8_t(sp[0]) & 0xf);
    case SPARSE_8:
        return sparse(type);
    case SPARSE_12:
        return sparse(type);
    case SPARSE_16:
        return sparse(type);
    case SPARSE_24:
        return sparse(type);
    case SPARSE_40:
        return sparse(type);
    case DENSE_12:
        return dense(type);
    case DENSE_16:
        return dense(type);
    case DENSE_24:
        return dense(type);
    case DENSE_32:
        return dense(type);
    case DENSE_40:
        return dense(type);
    case LONG_DENSE:
        return dense(type);
    }
    [[unlikely]] sstables::on_bti_parse_error(pos);
}

node_traverse_sidemost_result bti_walk_down_rightmost_path(int64_t pos, const_bytes sp) {
    auto type = uint8_t(sp[0]) >> 4;
    trie::node_traverse_sidemost_result result;
    result.body_pos = pos;
    auto single = [&](uint64_t offset, uint8_t payload_bits) {
        result.n_children = 1;
        result.payload_bits = payload_bits;
        result.child_offset = offset;
        return result;
    };
    auto sparse = [&] [[gnu::always_inline]] (int type) {
        int n_children = int(sp[1]);
        auto bpp = bits_per_pointer_arr[type];
        result.n_children = n_children;
        result.payload_bits = uint8_t(sp[0]) & 0xf;
        result.child_offset = read_offset(sp.subspan(2 + n_children), n_children - 1, bpp);
        return result;
    };
    auto dense = [&] [[gnu::always_inline]] (int type) {
        auto dense_span = uint64_t(sp[2]) + 1;
        auto bpp = bits_per_pointer_arr[type];
        result.n_children = dense_span;
        result.payload_bits = uint8_t(sp[0]) & 0xf;
        result.child_offset = read_offset(sp.subspan(3), dense_span - 1, bpp);
        return result;
    };
    switch (type) {
    case PAYLOAD_ONLY:
        result.payload_bits = uint8_t(sp[0]) & 0xf;
        result.n_children = 0;
        result.child_offset = -1;
        return result;
    case SINGLE_NOPAYLOAD_4:
        return single(uint64_t(sp[0]) & 0xf, 0);
    case SINGLE_NOPAYLOAD_12:
        return single((uint64_t(sp[0]) & 0xf) << 8 | uint64_t(sp[1]), 0);
    case SINGLE_8:
        return single(uint64_t(sp[2]), uint8_t(sp[0]) & 0xf);
    case SINGLE_16:
        return single(uint64_t(sp[2]) << 8 | uint64_t(sp[3]), uint8_t(sp[0]) & 0xf);
    case SPARSE_8:
        return sparse(type);
    case SPARSE_12:
        return sparse(type);
    case SPARSE_16:
        return sparse(type);
    case SPARSE_24:
        return sparse(type);
    case SPARSE_40:
        return sparse(type);
    case DENSE_12:
        return dense(type);
    case DENSE_16:
        return dense(type);
    case DENSE_24:
        return dense(type);
    case DENSE_32:
        return dense(type);
    case DENSE_40:
        return dense(type);
    case LONG_DENSE:
        return dense(type);
    }
    [[unlikely]] sstables::on_bti_parse_error(pos);
}

bti_node_reader::bti_node_reader(cached_file& f)
    : _file(f) {
}

bool bti_node_reader::cached(int64_t pos) const {
    return _cached_page && _cached_page->pos() / cached_file::page_size == pos / cached_file::page_size;
}

seastar::future<> bti_node_reader::load(int64_t pos, const reader_permit& permit, const tracing::trace_state_ptr& trace_ptr) {
    if (cached(pos)) {
        return make_ready_future<>();
    }
    return _file.get().get_shared_page(pos, permit, trace_ptr).then([this](cached_file::page_read_result page) {
        _cached_page = std::move(page.ptr);
    });
}

trie::load_final_node_result bti_node_reader::read_node(int64_t pos) {
    SCYLLA_ASSERT(cached(pos));
    auto sp = _cached_page->get_view().subspan(pos % cached_file::page_size);
    return bti_read_node(pos, sp);
}

trie::node_traverse_result bti_node_reader::walk_down_along_key(int64_t pos, const_bytes key) {
    SCYLLA_ASSERT(cached(pos));
    auto sp = _cached_page->get_view().subspan(pos % cached_file::page_size);
    return bti_walk_down_along_key(pos, sp, key);
}

void bti_node_reader::walk_down_along_key_batch(int64_t pos, const_bytes key_bytes, std::span<node_traverse_result> results) {
    SCYLLA_ASSERT(cached(pos));
    auto sp = _cached_page->get_view().subspan(pos % cached_file::page_size);
    bti_walk_down_along_key_batch(pos, sp, key_bytes, results);
}

trie::node_traverse_sidemost_result bti_node_reader::walk_down_leftmost_path(int64_t pos) {
    SCYLLA_ASSERT(cached(pos));
    auto sp = _cached_page->get_view().subspan(pos % cached_file::page_size);
    return bti_walk_down_leftmost_path(pos, sp);
}

trie::node_traverse_sidemost_result bti_node_reader::walk_down_rightmost_path(int64_t pos) {
    SCYLLA_ASSERT(cached(pos));
    auto sp = _cached_page->get_view().subspan(pos % cached_file::page_size);
    return bti_walk_down_rightmost_path(pos, sp);
}

trie::get_child_result bti_node_reader::get_child(int64_t pos, int child_idx, bool forward) const {
    SCYLLA_ASSERT(cached(pos));
    auto sp = _cached_page->get_view().subspan(pos % cached_file::page_size);
    return bti_get_child(pos, sp, child_idx, forward);
}

const_bytes bti_node_reader::get_payload(int64_t pos) const {
    SCYLLA_ASSERT(cached(pos));
    auto sp = _cached_page->get_view().subspan(pos % cached_file::page_size);
    return bti_get_payload(pos, sp);
}

} // namespace sstables::trie
