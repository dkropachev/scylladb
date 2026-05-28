/*
 * Copyright (C) 2020-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

#include <seastar/testing/perf_tests.hh>
#include <seastar/testing/test_runner.hh>
#include <seastar/util/defer.hh>

#include <cstdint>
#include <vector>

#include "db/config.hh"
#include "schema/schema.hh"
#include "schema/schema_builder.hh"
#include "sstables/mx/types.hh"
#include "sstables/trie/bti_key_translation.hh"
#include "sstables/trie/bti_node_reader.hh"
#include "sstables/trie/bti_node_type.hh"

namespace {

using sstables::trie::bits_per_pointer_arr;
using sstables::trie::node_type;
using sstables::trie::SPARSE_16;
using sstables::trie::SPARSE_40;

void append_be(std::vector<std::byte>& out, uint64_t value, int bytes) {
    for (int shift = (bytes - 1) * 8; shift >= 0; shift -= 8) {
        out.push_back(std::byte((value >> shift) & 0xff));
    }
}

std::vector<uint8_t> make_sparse_transitions(int n_children) {
    std::vector<uint8_t> transitions;
    transitions.reserve(n_children);
    for (int i = 0; i < n_children; ++i) {
        transitions.push_back(i * 256 / n_children);
    }
    return transitions;
}

std::vector<std::byte> make_sparse_node(node_type type, int n_children) {
    auto transitions = make_sparse_transitions(n_children);
    auto bpp = bits_per_pointer_arr[type];

    std::vector<std::byte> node;
    node.reserve(2 + n_children + n_children * bpp / 8);
    node.push_back(std::byte(type << 4));
    node.push_back(std::byte(n_children));
    for (auto transition : transitions) {
        node.push_back(std::byte(transition));
    }
    for (int i = 0; i < n_children; ++i) {
        append_be(node, uint64_t(i + 1), bpp / 8);
    }
    return node;
}

std::vector<std::byte> make_all_lookup_keys() {
    std::vector<std::byte> keys;
    keys.reserve(256);
    for (int i = 0; i < 256; ++i) {
        keys.push_back(std::byte(i));
    }
    return keys;
}

std::vector<std::byte> make_lookup_keys(std::initializer_list<uint8_t> keys) {
    std::vector<std::byte> result;
    result.reserve(keys.size());
    for (auto key : keys) {
        result.push_back(std::byte(key));
    }
    return result;
}

void run_sparse_lookup(const std::vector<std::byte>& node, const std::vector<std::byte>& keys) {
    uint64_t sum = 0;
    auto sp = const_bytes(node.data(), node.size());
    for (auto key : keys) {
        auto result = sstables::trie::bti_walk_down_along_key(0, sp, std::span<const std::byte>(&key, 1));
        sum += uint64_t(result.found_idx) + uint64_t(result.found_byte) + uint64_t(result.child_offset);
    }
    perf_tests::do_not_optimize(sum);
}

void run_sparse_lookup(const std::vector<std::byte>& node, const std::vector<std::byte>& keys, db::simd_optimization_mode mode) {
    const auto previous_mode = sstables::trie::get_bti_sparse_node_simd_optimization_mode();
    auto restore_mode = defer([previous_mode] {
        sstables::trie::set_bti_sparse_node_simd_optimization_mode(previous_mode);
    });
    sstables::trie::set_bti_sparse_node_simd_optimization_mode(mode);

    uint64_t sum = 0;
    auto sp = const_bytes(node.data(), node.size());
    for (auto key : keys) {
        auto result = sstables::trie::bti_walk_down_along_key(0, sp, std::span<const std::byte>(&key, 1));
        sum += uint64_t(result.found_idx) + uint64_t(result.found_byte) + uint64_t(result.child_offset);
    }
    perf_tests::do_not_optimize(sum);
}

void run_sparse_lookup_batch(const std::vector<std::byte>& node, const std::vector<std::byte>& keys, std::vector<sstables::trie::node_traverse_result>& results) {
    uint64_t sum = 0;
    auto sp = const_bytes(node.data(), node.size());
    sstables::trie::bti_walk_down_along_key_batch(0, sp, const_bytes(keys.data(), keys.size()), std::span(results.data(), keys.size()));
    for (size_t i = 0; i < keys.size(); ++i) {
        const auto& result = results[i];
        sum += uint64_t(result.found_idx) + uint64_t(result.found_byte) + uint64_t(result.child_offset);
    }
    perf_tests::do_not_optimize(sum);
}

void run_sparse_lookup_single_output(const std::vector<std::byte>& node, const std::vector<std::byte>& keys, std::vector<sstables::trie::node_traverse_result>& results) {
    uint64_t sum = 0;
    auto sp = const_bytes(node.data(), node.size());
    for (size_t i = 0; i < keys.size(); ++i) {
        results[i] = sstables::trie::bti_walk_down_along_key(0, sp, const_bytes(&keys[i], 1));
        const auto& result = results[i];
        sum += uint64_t(result.found_idx) + uint64_t(result.found_byte) + uint64_t(result.child_offset);
    }
    perf_tests::do_not_optimize(sum);
}

struct sparse_node_lookup_test {
    std::vector<std::byte> all_keys = make_all_lookup_keys();
    std::vector<std::byte> edge_keys = make_lookup_keys({0x00, 0x01, 0x7f, 0x80, 0xfe, 0xff});
    std::vector<sstables::trie::node_traverse_result> all_key_results = std::vector<sstables::trie::node_traverse_result>(all_keys.size());
    std::vector<sstables::trie::node_traverse_result> edge_key_results = std::vector<sstables::trie::node_traverse_result>(edge_keys.size());
    std::vector<std::byte> sparse16_n8 = make_sparse_node(SPARSE_16, 8);
    std::vector<std::byte> sparse16_n12 = make_sparse_node(SPARSE_16, 12);
    std::vector<std::byte> sparse16_n15 = make_sparse_node(SPARSE_16, 15);
    std::vector<std::byte> sparse16_n16 = make_sparse_node(SPARSE_16, 16);
    std::vector<std::byte> sparse16_n17 = make_sparse_node(SPARSE_16, 17);
    std::vector<std::byte> sparse16_n24 = make_sparse_node(SPARSE_16, 24);
    std::vector<std::byte> sparse16_n31 = make_sparse_node(SPARSE_16, 31);
    std::vector<std::byte> sparse16_n32 = make_sparse_node(SPARSE_16, 32);
    std::vector<std::byte> sparse16_n48 = make_sparse_node(SPARSE_16, 48);
    std::vector<std::byte> sparse16_n64 = make_sparse_node(SPARSE_16, 64);
    std::vector<std::byte> sparse16_n128 = make_sparse_node(SPARSE_16, 128);
    std::vector<std::byte> sparse16_n255 = make_sparse_node(SPARSE_16, 255);
    std::vector<std::byte> sparse40_n128 = make_sparse_node(SPARSE_40, 128);
};

} // anonymous namespace

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
        auto [offset, ptr] = sstables::trie::lcb_mismatch(a.begin(), b.begin());
        perf_tests::do_not_optimize(offset);
        perf_tests::do_not_optimize(ptr);
    }
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n8_all_keys) {
    run_sparse_lookup(sparse16_n8, all_keys);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n8_all_keys_batch) {
    run_sparse_lookup_batch(sparse16_n8, all_keys, all_key_results);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n8_all_keys_single_output) {
    run_sparse_lookup_single_output(sparse16_n8, all_keys, all_key_results);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n12_all_keys) {
    run_sparse_lookup(sparse16_n12, all_keys);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n15_all_keys) {
    run_sparse_lookup(sparse16_n15, all_keys);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n16_all_keys) {
    run_sparse_lookup(sparse16_n16, all_keys);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n16_all_keys_batch) {
    run_sparse_lookup_batch(sparse16_n16, all_keys, all_key_results);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n16_all_keys_single_output) {
    run_sparse_lookup_single_output(sparse16_n16, all_keys, all_key_results);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n17_all_keys) {
    run_sparse_lookup(sparse16_n17, all_keys);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n24_all_keys) {
    run_sparse_lookup(sparse16_n24, all_keys);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n31_all_keys) {
    run_sparse_lookup(sparse16_n31, all_keys);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n32_all_keys) {
    run_sparse_lookup(sparse16_n32, all_keys);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n32_all_keys_batch) {
    run_sparse_lookup_batch(sparse16_n32, all_keys, all_key_results);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n32_all_keys_single_output) {
    run_sparse_lookup_single_output(sparse16_n32, all_keys, all_key_results);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n48_all_keys) {
    run_sparse_lookup(sparse16_n48, all_keys);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n64_all_keys) {
    run_sparse_lookup(sparse16_n64, all_keys);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n64_all_keys_avx2) {
    run_sparse_lookup(sparse16_n64, all_keys, db::simd_optimization_mode::avx2);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n64_all_keys_avx512) {
    run_sparse_lookup(sparse16_n64, all_keys, db::simd_optimization_mode::avx512);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n64_all_keys_batch) {
    run_sparse_lookup_batch(sparse16_n64, all_keys, all_key_results);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n64_all_keys_single_output) {
    run_sparse_lookup_single_output(sparse16_n64, all_keys, all_key_results);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n128_all_keys) {
    run_sparse_lookup(sparse16_n128, all_keys);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n128_all_keys_avx2) {
    run_sparse_lookup(sparse16_n128, all_keys, db::simd_optimization_mode::avx2);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n128_all_keys_avx512) {
    run_sparse_lookup(sparse16_n128, all_keys, db::simd_optimization_mode::avx512);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n128_all_keys_batch) {
    run_sparse_lookup_batch(sparse16_n128, all_keys, all_key_results);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n128_all_keys_single_output) {
    run_sparse_lookup_single_output(sparse16_n128, all_keys, all_key_results);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n255_all_keys) {
    run_sparse_lookup(sparse16_n255, all_keys);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n255_all_keys_avx2) {
    run_sparse_lookup(sparse16_n255, all_keys, db::simd_optimization_mode::avx2);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n255_all_keys_avx512) {
    run_sparse_lookup(sparse16_n255, all_keys, db::simd_optimization_mode::avx512);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n255_all_keys_batch) {
    run_sparse_lookup_batch(sparse16_n255, all_keys, all_key_results);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n255_all_keys_single_output) {
    run_sparse_lookup_single_output(sparse16_n255, all_keys, all_key_results);
}

PERF_TEST_F(sparse_node_lookup_test, sparse40_n128_all_keys) {
    run_sparse_lookup(sparse40_n128, all_keys);
}

PERF_TEST_F(sparse_node_lookup_test, sparse40_n128_all_keys_batch) {
    run_sparse_lookup_batch(sparse40_n128, all_keys, all_key_results);
}

PERF_TEST_F(sparse_node_lookup_test, sparse40_n128_all_keys_single_output) {
    run_sparse_lookup_single_output(sparse40_n128, all_keys, all_key_results);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n128_edge_keys) {
    run_sparse_lookup(sparse16_n128, edge_keys);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n128_edge_keys_batch) {
    run_sparse_lookup_batch(sparse16_n128, edge_keys, edge_key_results);
}

PERF_TEST_F(sparse_node_lookup_test, sparse16_n128_edge_keys_single_output) {
    run_sparse_lookup_single_output(sparse16_n128, edge_keys, edge_key_results);
}
