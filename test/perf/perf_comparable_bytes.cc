/*
 * Copyright (C) 2026-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

#include <seastar/testing/perf_tests.hh>
#include <seastar/testing/test_runner.hh>
#include <seastar/util/defer.hh>

#include <algorithm>

#include "db/config.hh"
#include "types/comparable_bytes.hh"
#include "types/types.hh"

class comparable_bytes_zero_escaping {
    static constexpr size_t value_size = 4096;
    static constexpr size_t count = 512;

    managed_bytes _no_zeros;
    managed_bytes _sparse_zeros;
    comparable_bytes _encoded_no_zeros;
    comparable_bytes _encoded_sparse_zeros;

    static managed_bytes make_no_zeros() {
        bytes data(bytes::initialized_later{}, value_size);
        std::fill(data.begin(), data.end(), int8_t('x'));
        return managed_bytes(std::move(data));
    }

    static managed_bytes make_sparse_zeros() {
        bytes data(bytes::initialized_later{}, value_size);
        std::fill(data.begin(), data.end(), int8_t('x'));
        for (size_t i = 127; i < data.size(); i += 257) {
            data[i] = int8_t(0);
        }
        return managed_bytes(std::move(data));
    }

public:
    comparable_bytes_zero_escaping()
        : _no_zeros(make_no_zeros())
        , _sparse_zeros(make_sparse_zeros())
        , _encoded_no_zeros(*bytes_type, managed_bytes_view(_no_zeros))
        , _encoded_sparse_zeros(*bytes_type, managed_bytes_view(_sparse_zeros)) {
    }

    const managed_bytes& no_zeros() const {
        return _no_zeros;
    }

    const managed_bytes& sparse_zeros() const {
        return _sparse_zeros;
    }

    const comparable_bytes& encoded_no_zeros() const {
        return _encoded_no_zeros;
    }

    const comparable_bytes& encoded_sparse_zeros() const {
        return _encoded_sparse_zeros;
    }

    static constexpr size_t iterations() {
        return count;
    }

    size_t encode_no_zeros(db::simd_optimization_mode mode) const {
        const auto previous_mode = get_comparable_bytes_simd_optimization_mode();
        auto restore_mode = defer([previous_mode] {
            set_comparable_bytes_simd_optimization_mode(previous_mode);
        });
        set_comparable_bytes_simd_optimization_mode(mode);
        for (size_t i = 0; i < iterations(); ++i) {
            auto view = managed_bytes_view(no_zeros());
            auto encoded = comparable_bytes(*bytes_type, view);
            perf_tests::do_not_optimize(encoded.size());
        }
        return iterations();
    }

    size_t encode_sparse_zeros(db::simd_optimization_mode mode) const {
        const auto previous_mode = get_comparable_bytes_simd_optimization_mode();
        auto restore_mode = defer([previous_mode] {
            set_comparable_bytes_simd_optimization_mode(previous_mode);
        });
        set_comparable_bytes_simd_optimization_mode(mode);
        for (size_t i = 0; i < iterations(); ++i) {
            auto view = managed_bytes_view(sparse_zeros());
            auto encoded = comparable_bytes(*bytes_type, view);
            perf_tests::do_not_optimize(encoded.size());
        }
        return iterations();
    }

    size_t decode_no_zeros(db::simd_optimization_mode mode) const {
        const auto previous_mode = get_comparable_bytes_simd_optimization_mode();
        auto restore_mode = defer([previous_mode] {
            set_comparable_bytes_simd_optimization_mode(previous_mode);
        });
        set_comparable_bytes_simd_optimization_mode(mode);
        for (size_t i = 0; i < iterations(); ++i) {
            auto decoded = encoded_no_zeros().to_serialized_bytes(*bytes_type);
            perf_tests::do_not_optimize(decoded->size());
        }
        return iterations();
    }

    size_t decode_sparse_zeros(db::simd_optimization_mode mode) const {
        const auto previous_mode = get_comparable_bytes_simd_optimization_mode();
        auto restore_mode = defer([previous_mode] {
            set_comparable_bytes_simd_optimization_mode(previous_mode);
        });
        set_comparable_bytes_simd_optimization_mode(mode);
        for (size_t i = 0; i < iterations(); ++i) {
            auto decoded = encoded_sparse_zeros().to_serialized_bytes(*bytes_type);
            perf_tests::do_not_optimize(decoded->size());
        }
        return iterations();
    }
};

PERF_TEST_F(comparable_bytes_zero_escaping, encode_blob_no_zeros_4k) {
    return encode_no_zeros(db::simd_optimization_mode::automatic);
}

PERF_TEST_F(comparable_bytes_zero_escaping, encode_blob_no_zeros_4k_avx2) {
    return encode_no_zeros(db::simd_optimization_mode::avx2);
}

PERF_TEST_F(comparable_bytes_zero_escaping, encode_blob_no_zeros_4k_avx512) {
    return encode_no_zeros(db::simd_optimization_mode::avx512);
}

PERF_TEST_F(comparable_bytes_zero_escaping, encode_blob_sparse_zeros_4k) {
    return encode_sparse_zeros(db::simd_optimization_mode::automatic);
}

PERF_TEST_F(comparable_bytes_zero_escaping, encode_blob_sparse_zeros_4k_avx2) {
    return encode_sparse_zeros(db::simd_optimization_mode::avx2);
}

PERF_TEST_F(comparable_bytes_zero_escaping, encode_blob_sparse_zeros_4k_avx512) {
    return encode_sparse_zeros(db::simd_optimization_mode::avx512);
}

PERF_TEST_F(comparable_bytes_zero_escaping, decode_blob_no_zeros_4k) {
    return decode_no_zeros(db::simd_optimization_mode::automatic);
}

PERF_TEST_F(comparable_bytes_zero_escaping, decode_blob_no_zeros_4k_avx2) {
    return decode_no_zeros(db::simd_optimization_mode::avx2);
}

PERF_TEST_F(comparable_bytes_zero_escaping, decode_blob_no_zeros_4k_avx512) {
    return decode_no_zeros(db::simd_optimization_mode::avx512);
}

PERF_TEST_F(comparable_bytes_zero_escaping, decode_blob_sparse_zeros_4k) {
    return decode_sparse_zeros(db::simd_optimization_mode::automatic);
}

PERF_TEST_F(comparable_bytes_zero_escaping, decode_blob_sparse_zeros_4k_avx2) {
    return decode_sparse_zeros(db::simd_optimization_mode::avx2);
}

PERF_TEST_F(comparable_bytes_zero_escaping, decode_blob_sparse_zeros_4k_avx512) {
    return decode_sparse_zeros(db::simd_optimization_mode::avx512);
}
