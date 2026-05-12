/*
 * Copyright (C) 2020-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

#include <seastar/testing/perf_tests.hh>
#include <seastar/testing/test_runner.hh>

#include <algorithm>
#include <cstdint>

#include "db/config.hh"
#include "schema/schema.hh"
#include "schema/schema_builder.hh"
#include "sstables/mx/types.hh"
#include "sstables/trie/bti_key_translation.hh"

struct lcb_mismatch_test {
    schema_ptr _s;
    std::vector<sstables::clustering_info> keys;
    lcb_mismatch_test() {
        int n_columns = 16;
        auto builder = schema_builder("ks", "t")
            .with_column("pk", int32_type, column_kind::partition_key);
        for (int i = 0; i <= n_columns; ++i) {
            builder.with_column(bytes(fmt::format("c{}", i).c_str()), int32_type, column_kind::clustering_key);
        }
        _s = builder.build();
        std::vector<data_value> components;
        for (int i = 0; i < n_columns - 1; ++i) {
            components.push_back(data_value(int32_t(0)));
        }
        for (int i = 0; i < 100; ++i) {
            components.push_back(data_value(int32_t(i)));
            keys.push_back({
                sstables::clustering_info{
                    clustering_key_prefix::from_deeply_exploded(*_s, components),
                    sstables::bound_kind_m::clustering
                }
            });
            components.pop_back();
        }
    }
};

PERF_TEST_F(lcb_mismatch_test, lcb_mismatch) {
    for (size_t i = 1; i < keys.size(); ++i) {
        auto a = sstables::trie::lazy_comparable_bytes_from_clustering_position(*_s, keys[i - 1]);
        auto b = sstables::trie::lazy_comparable_bytes_from_clustering_position(*_s, keys[i]);
        auto [offset, ptr] = sstables::trie::lcb_mismatch(a.begin(), b.begin(), db::simd_optimization_mode::automatic);
        perf_tests::do_not_optimize(offset);
        perf_tests::do_not_optimize(ptr);
    }
}

struct blob_lcb_mismatch_test {
    static constexpr size_t value_size = 4096;
    static constexpr size_t key_count = 100;

    schema_ptr _s;
    std::vector<sstables::clustering_info> no_zero_keys;
    std::vector<sstables::clustering_info> sparse_zero_keys;

    blob_lcb_mismatch_test() {
        _s = schema_builder("ks", "blob_t")
            .with_column("pk", int32_type, column_kind::partition_key)
            .with_column("c", bytes_type, column_kind::clustering_key)
            .build();

        no_zero_keys = make_keys(false);
        sparse_zero_keys = make_keys(true);
    }

    static bytes make_blob(size_t idx, bool sparse_zeros) {
        bytes data(bytes::initialized_later{}, value_size);
        std::fill(data.begin(), data.end(), int8_t('x'));
        if (sparse_zeros) {
            for (size_t i = 127; i < data.size() - sizeof(uint64_t); i += 257) {
                data[i] = int8_t(0);
            }
        }
        auto p = data.end() - sizeof(uint64_t);
        write_be(reinterpret_cast<char*>(&*p), static_cast<uint64_t>(idx));
        return data;
    }

    std::vector<sstables::clustering_info> make_keys(bool sparse_zeros) const {
        std::vector<sstables::clustering_info> keys;
        keys.reserve(key_count);
        for (size_t i = 0; i < key_count; ++i) {
            keys.push_back({
                sstables::clustering_info{
                    clustering_key_prefix::from_deeply_exploded(*_s, {data_value(make_blob(i, sparse_zeros))}),
                    sstables::bound_kind_m::clustering
                }
            });
        }
        return keys;
    }

    void test_keys(const std::vector<sstables::clustering_info>& keys, db::simd_optimization_mode mode) {
        for (size_t i = 1; i < keys.size(); ++i) {
            auto a = sstables::trie::lazy_comparable_bytes_from_clustering_position(*_s, keys[i - 1]);
            auto b = sstables::trie::lazy_comparable_bytes_from_clustering_position(*_s, keys[i]);
            auto [offset, ptr] = sstables::trie::lcb_mismatch(a.begin(), b.begin(), mode);
            perf_tests::do_not_optimize(offset);
            perf_tests::do_not_optimize(ptr);
        }
    }
};

PERF_TEST_F(blob_lcb_mismatch_test, blob_no_zeros_4k) {
    test_keys(no_zero_keys, db::simd_optimization_mode::automatic);
}

PERF_TEST_F(blob_lcb_mismatch_test, blob_no_zeros_4k_avx2) {
    test_keys(no_zero_keys, db::simd_optimization_mode::avx2);
}

PERF_TEST_F(blob_lcb_mismatch_test, blob_no_zeros_4k_avx512) {
    test_keys(no_zero_keys, db::simd_optimization_mode::avx512);
}

PERF_TEST_F(blob_lcb_mismatch_test, blob_sparse_zeros_4k) {
    test_keys(sparse_zero_keys, db::simd_optimization_mode::automatic);
}

PERF_TEST_F(blob_lcb_mismatch_test, blob_sparse_zeros_4k_avx2) {
    test_keys(sparse_zero_keys, db::simd_optimization_mode::avx2);
}

PERF_TEST_F(blob_lcb_mismatch_test, blob_sparse_zeros_4k_avx512) {
    test_keys(sparse_zero_keys, db::simd_optimization_mode::avx512);
}
