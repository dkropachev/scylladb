/*
 * Copyright (C) 2024-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

#include "bti_key_translation.hh"

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
#include "sstables/mx/types.hh"

namespace sstables::trie {

namespace {

static size_t first_mismatch_scalar(std::span<const std::byte> a, std::span<const std::byte> b, size_t first, size_t last) noexcept {
    for (size_t i = first; i < last; ++i) {
        if (a[i] != b[i]) {
            return i;
        }
    }
    return last;
}

static size_t first_mismatch_scalar(std::span<const std::byte> a, std::span<const std::byte> b) noexcept {
    return first_mismatch_scalar(a, b, 0, a.size());
}

#if defined(__x86_64__) || defined(__i386__)
[[gnu::target("sse2")]]
static size_t first_mismatch_sse2(std::span<const std::byte> a, std::span<const std::byte> b) noexcept {
    constexpr size_t lane_size = 16;
    constexpr uint32_t all_equal = (uint32_t(1) << lane_size) - 1;

    size_t i = 0;
    const size_t size = a.size();
    for (; i + lane_size <= size; i += lane_size) {
        const auto av = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a.data() + i));
        const auto bv = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b.data() + i));
        const auto eq = _mm_cmpeq_epi8(av, bv);
        const uint32_t mask = static_cast<uint32_t>(_mm_movemask_epi8(eq));
        if (mask != all_equal) {
            return i + std::countr_zero((~mask) & all_equal);
        }
    }
    return first_mismatch_scalar(a, b, i, size);
}

[[gnu::target("avx2")]]
static size_t first_mismatch_avx2(std::span<const std::byte> a, std::span<const std::byte> b) noexcept {
    constexpr size_t lane_size = 32;
    constexpr uint32_t all_equal = ~uint32_t(0);

    size_t i = 0;
    const size_t size = a.size();
    for (; i + lane_size <= size; i += lane_size) {
        const auto av = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.data() + i));
        const auto bv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b.data() + i));
        const auto eq = _mm256_cmpeq_epi8(av, bv);
        const uint32_t mask = static_cast<uint32_t>(_mm256_movemask_epi8(eq));
        if (mask != all_equal) {
            return i + std::countr_zero(~mask);
        }
    }
    return first_mismatch_scalar(a, b, i, size);
}

[[gnu::target("avx512f,avx512bw")]]
static size_t first_mismatch_avx512(std::span<const std::byte> a, std::span<const std::byte> b) noexcept {
    constexpr size_t lane_size = 64;
    constexpr uint64_t all_equal = ~uint64_t(0);

    size_t i = 0;
    const size_t size = a.size();
    for (; i + lane_size <= size; i += lane_size) {
        const auto av = _mm512_loadu_si512(reinterpret_cast<const void*>(a.data() + i));
        const auto bv = _mm512_loadu_si512(reinterpret_cast<const void*>(b.data() + i));
        const uint64_t mask = static_cast<uint64_t>(_mm512_cmpeq_epi8_mask(av, bv));
        if (mask != all_equal) {
            return i + std::countr_zero(~mask);
        }
    }
    return first_mismatch_scalar(a, b, i, size);
}

static bool cpu_supports_avx512() noexcept {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw");
}

static bool cpu_supports_avx2() noexcept {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
}

static bool cpu_supports_sse2() noexcept {
    __builtin_cpu_init();
    return __builtin_cpu_supports("sse2");
}
#endif

#if defined(__aarch64__)
static size_t first_mismatch_neon(std::span<const std::byte> a, std::span<const std::byte> b) noexcept {
    constexpr size_t lane_size = 16;

    size_t i = 0;
    const size_t size = a.size();
    for (; i + lane_size <= size; i += lane_size) {
        const auto av = vld1q_u8(reinterpret_cast<const uint8_t*>(a.data() + i));
        const auto bv = vld1q_u8(reinterpret_cast<const uint8_t*>(b.data() + i));
        const auto eq = vceqq_u8(av, bv);
        if (vminvq_u8(eq) != 0xff) {
            return first_mismatch_scalar(a, b, i, i + lane_size);
        }
    }
    return first_mismatch_scalar(a, b, i, size);
}
#endif

} // anonymous namespace

size_t first_mismatch_offset(std::span<const std::byte> a, std::span<const std::byte> b, db::simd_optimization_mode mode) noexcept {
    const size_t size = std::min(a.size(), b.size());
    a = a.first(size);
    b = b.first(size);

#if defined(__x86_64__) || defined(__i386__)
    if (mode == db::simd_optimization_mode::off) {
        return first_mismatch_scalar(a, b);
    }

    static const bool has_avx512 = cpu_supports_avx512();
    if (mode == db::simd_optimization_mode::avx512 && has_avx512) {
        return first_mismatch_avx512(a, b);
    }
    static const bool has_avx2 = cpu_supports_avx2();
    if ((mode == db::simd_optimization_mode::automatic || mode == db::simd_optimization_mode::avx2) && has_avx2) {
        return first_mismatch_avx2(a, b);
    }
    static const bool has_sse2 = cpu_supports_sse2();
    if ((mode == db::simd_optimization_mode::automatic || mode == db::simd_optimization_mode::sse) && has_sse2) {
        return first_mismatch_sse2(a, b);
    }
    return first_mismatch_scalar(a, b);
#elif defined(__aarch64__)
    if (mode == db::simd_optimization_mode::automatic || mode == db::simd_optimization_mode::neon) {
        return first_mismatch_neon(a, b);
    }
    return first_mismatch_scalar(a, b);
#else
    return first_mismatch_scalar(a, b);
#endif
}

lazy_comparable_bytes_from_ring_position::lazy_comparable_bytes_from_ring_position(const schema& s, dht::ring_position_view rpv)
    : _s(s)
    // Note about `dht::maximum_token()`: this isn't a real token.
    // It's just a fake element greater than any real token.
    // It is never written used during writes, but might appear as a search key during reads.
    // So we need to encode it in a way that preserves the ordering, doesn't matter to what exactly.
    // Here we arbitrarily encode identically to the greatest real token, with weight 1.
    // (Weight was handled in the constructor).
    //
    // dht::minimum_token() also isn't a real token, but it doesn't need special handling,
    // because its `unbias()` returns something smaller than any real token's `unbias()`.
    , _weight(rpv.token().is_maximum() ? bound_weight::after_all_prefixed : bound_weight(rpv.weight()))
    // If there's no key, we just eagerly translate the "key part" to the terminator byte.
    , _pk(rpv.key()
        ? decltype(_pk)(*rpv.key())
        : decltype(_pk)(comparable_bytes(managed_bytes{bytes::value_type(bound_weight_to_terminator(_weight))})))
{
    if (std::holds_alternative<comparable_bytes>(_pk)) {
        // If we initialized `_pk` to a `comparable_bytes`, we must initialize `_size` accordingly.
        // (operator++ expects it).
        // This branch is only entered if there's no partition key, only a token.
        // So the entire encoding is the nine bytes in _token_buf (leading 0x40, token), and the one terminator byte in `comparable_bytes`.
        _size = _token_buf.size() + 1;
    }
    init_first_fragment(rpv.token().is_maximum() ? dht::token::last() : rpv.token());
}

lazy_comparable_bytes_from_ring_position::lazy_comparable_bytes_from_ring_position(const schema& s, dht::decorated_key dk)
    : _s(s)
    , _weight(bound_weight::equal)
    , _pk(std::move(dk._key))
{
    init_first_fragment(dk._token);
}

void lazy_comparable_bytes_from_ring_position::init_first_fragment(dht::token dht_token) {
    auto p = _token_buf.data();
    // 0x40 is the key field separator as defined by the BTI byte-comparable encoding.
    // See also https://github.com/apache/cassandra/blob/fad1f7457032544ab6a7b40c5d38ecb8b25899bb/src/java/org/apache/cassandra/utils/bytecomparable/ByteComparable.md#multi-component-sequences-partition-or-clustering-keys-tuples-bounds-and-nulls
    *p++ = std::byte(0x40);
    uint64_t token = dht_token.unbias();
    seastar::write_be<uint64_t>(reinterpret_cast<char*>(p), token);
}

void lazy_comparable_bytes_from_ring_position::trim(const size_t n) {
    SCYLLA_ASSERT(n <= _size);
    _size = n;
}

std::byte bound_weight_to_terminator(bound_weight b) {
    // 0x20, 0x38, 0x60 are terminators defined by the BTI byte-comparable encoding.
    // See also https://github.com/apache/cassandra/blob/fad1f7457032544ab6a7b40c5d38ecb8b25899bb/src/java/org/apache/cassandra/utils/bytecomparable/ByteComparable.md#multi-component-sequences-partition-or-clustering-keys-tuples-bounds-and-nulls
    switch (b) {
    case bound_weight::after_all_prefixed:
        return std::byte(0x60);
    case bound_weight::before_all_prefixed:
        return std::byte(0x20);
    case bound_weight::equal:
        return std::byte(0x38);
    }
}

[[maybe_unused]]
static bound_weight convert_bound_to_bound_weight(sstables::bound_kind_m b) {
    switch (b) {
    case sstables::bound_kind_m::excl_end:
    case sstables::bound_kind_m::incl_start:
    case sstables::bound_kind_m::excl_end_incl_start:
        return bound_weight::before_all_prefixed;
    case sstables::bound_kind_m::clustering:
    case sstables::bound_kind_m::static_clustering:
        return bound_weight::equal;
    case sstables::bound_kind_m::excl_start:
    case sstables::bound_kind_m::incl_end_excl_start:
    case sstables::bound_kind_m::incl_end:
        return bound_weight::after_all_prefixed;
    default:
        abort();
    }
}

static std::byte pip_to_terminator(const position_in_partition_view& pipv) {
    if (pipv.region() < partition_region::clustered) [[unlikely]] {
        // This isn't defined by the BTI format docs, it's just something smaller than
        // any actual BTI-encoded clustering position.
        // It could be reasonably used during reads, but it should never be written.
        return std::byte(0x00);
    } else if (pipv.region() > partition_region::clustered) [[unlikely]] {
        // This isn't defined by the BTI format docs, it's just something greater than
        // any actual BTI-encoded clustering position.
        // It could be reasonably used during reads, but it should never be written.
        return std::byte(0xff);
    } else {
        return bound_weight_to_terminator(pipv.get_bound_weight());
    }
}

lazy_comparable_bytes_from_clustering_position::lazy_comparable_bytes_from_clustering_position(const schema& s, position_in_partition_view pipv)
    : _encoded(comparable_bytes_from_compound(
        *s.clustering_key_prefix_type(),
        pipv.has_key() ? pipv.key().representation() : managed_bytes_view(),
        pip_to_terminator(pipv)
    ))
    , _size(_encoded.size())
{}

lazy_comparable_bytes_from_clustering_position::lazy_comparable_bytes_from_clustering_position(const schema& s, const sstables::clustering_info& ci)
    : _encoded(comparable_bytes_from_compound(
        *s.clustering_key_prefix_type(),
        ci.clustering.representation(),
        bound_weight_to_terminator(convert_bound_to_bound_weight(ci.kind))
    ))
    , _size(_encoded.size())
{}

void lazy_comparable_bytes_from_clustering_position::trim(unsigned n) {
    SCYLLA_ASSERT(n <= _size);
    _size = n;
}

} // namespace sstables::trie
