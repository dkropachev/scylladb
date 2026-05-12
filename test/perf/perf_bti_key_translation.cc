/*
 * Copyright (C) 2020-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

#include <seastar/testing/perf_tests.hh>
#include <seastar/testing/test_runner.hh>
#include <seastar/util/defer.hh>

#include <algorithm>
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
using sstables::trie::const_bytes;
using sstables::trie::DENSE_16;
using sstables::trie::DENSE_32;
using sstables::trie::get_child_result;
using sstables::trie::LONG_DENSE;
using sstables::trie::node_type;
using sstables::trie::read_offset;
using sstables::trie::SPARSE_16;
using sstables::trie::SPARSE_40;

constexpr int dense_span = 256;

void append_be(std::vector<std::byte>& out, uint64_t value, int bytes) {
    for (int shift = (bytes - 1) * 8; shift >= 0; shift -= 8) {
        out.push_back(std::byte((value >> shift) & 0xff));
    }
}

uint64_t offset_for(node_type type, int slot) {
    switch (type) {
    case DENSE_16:
        return slot + 1;
    case DENSE_32:
        return (uint64_t(1) << 16) + slot + 1;
    case LONG_DENSE:
        return (uint64_t(1) << 40) + slot + 1;
    default:
        abort();
    }
}

std::vector<std::byte> make_dense_node(node_type type, int stride) {
    const auto bpp = bits_per_pointer_arr[type];
    std::vector<std::byte> node;
    node.reserve(3 + dense_span * bpp / 8);
    node.push_back(std::byte(type << 4));
    node.push_back(std::byte(0));
    node.push_back(std::byte(dense_span - 1));

    for (int slot = 0; slot < dense_span; ++slot) {
        const bool occupied = slot % stride == 0 || slot == dense_span - 1;
        append_be(node, occupied ? offset_for(type, slot) : 0, bpp / 8);
    }
    return node;
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

std::vector<int> make_lookup_starts(bool forward) {
    std::vector<int> starts;
    starts.reserve(dense_span);
    if (forward) {
        for (int i = 0; i < dense_span; ++i) {
            starts.push_back(i);
        }
    } else {
        for (int i = dense_span - 1; i >= 0; --i) {
            starts.push_back(i);
        }
    }
    return starts;
}

get_child_result bti_get_child_scalar(const_bytes sp, int child_idx, bool forward) {
    const auto type = uint8_t(sp[0]) >> 4;
    const auto bpp = bits_per_pointer_arr[type];
    const auto offsets = sp.subspan(3);
    const auto end_idx = int(uint8_t(sp[2])) + 1;
    const int increment = forward ? 1 : -1;

    for (int idx = child_idx; idx < end_idx && idx >= 0; idx += increment) {
        if (auto off = read_offset(offsets, idx, bpp)) {
            return {
                .idx = idx,
                .offset = off,
            };
        }
    }
    abort();
}

void run_dense_lookup(const std::vector<std::byte>& node, const std::vector<int>& starts, bool forward, bool scalar) {
    uint64_t sum = 0;
    auto sp = const_bytes(node.data(), node.size());
    for (auto start : starts) {
        auto result = scalar
            ? bti_get_child_scalar(sp, start, forward)
            : sstables::trie::bti_get_child(0, sp, start, forward);
        sum += result.offset + result.idx;
    }
    perf_tests::do_not_optimize(sum);
}

void run_dense_lookup(const std::vector<std::byte>& node, const std::vector<int>& starts, bool forward, db::simd_optimization_mode mode) {
    const auto previous_mode = sstables::trie::get_bti_dense_node_simd_optimization_mode();
    auto restore_mode = defer([previous_mode] {
        sstables::trie::set_bti_dense_node_simd_optimization_mode(previous_mode);
    });
    sstables::trie::set_bti_dense_node_simd_optimization_mode(mode);

    uint64_t sum = 0;
    auto sp = const_bytes(node.data(), node.size());
    for (auto start : starts) {
        auto result = sstables::trie::bti_get_child(0, sp, start, forward);
        sum += result.offset + result.idx;
    }
    perf_tests::do_not_optimize(sum);
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

struct dense_node_scan_test {
    std::vector<std::byte> dense16_stride16 = make_dense_node(DENSE_16, 16);
    std::vector<std::byte> dense16_stride4 = make_dense_node(DENSE_16, 4);
    std::vector<std::byte> dense32_stride16 = make_dense_node(DENSE_32, 16);
    std::vector<std::byte> dense32_stride4 = make_dense_node(DENSE_32, 4);
    std::vector<std::byte> dense64_stride16 = make_dense_node(LONG_DENSE, 16);
    std::vector<std::byte> dense64_stride4 = make_dense_node(LONG_DENSE, 4);
    std::vector<int> forward_starts = make_lookup_starts(true);
    std::vector<int> reverse_starts = make_lookup_starts(false);
};

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

PERF_TEST_F(dense_node_scan_test, dense16_stride16_forward_scalar) {
    run_dense_lookup(dense16_stride16, forward_starts, true, true);
}

PERF_TEST_F(dense_node_scan_test, dense16_stride16_forward_simd) {
    run_dense_lookup(dense16_stride16, forward_starts, true, false);
}

PERF_TEST_F(dense_node_scan_test, dense16_stride16_reverse_scalar) {
    run_dense_lookup(dense16_stride16, reverse_starts, false, true);
}

PERF_TEST_F(dense_node_scan_test, dense16_stride16_reverse_simd) {
    run_dense_lookup(dense16_stride16, reverse_starts, false, false);
}

PERF_TEST_F(dense_node_scan_test, dense16_stride4_forward_scalar) {
    run_dense_lookup(dense16_stride4, forward_starts, true, true);
}

PERF_TEST_F(dense_node_scan_test, dense16_stride4_forward_simd) {
    run_dense_lookup(dense16_stride4, forward_starts, true, false);
}

PERF_TEST_F(dense_node_scan_test, dense16_stride4_reverse_scalar) {
    run_dense_lookup(dense16_stride4, reverse_starts, false, true);
}

PERF_TEST_F(dense_node_scan_test, dense16_stride4_reverse_simd) {
    run_dense_lookup(dense16_stride4, reverse_starts, false, false);
}

PERF_TEST_F(dense_node_scan_test, dense32_stride16_forward_scalar) {
    run_dense_lookup(dense32_stride16, forward_starts, true, true);
}

PERF_TEST_F(dense_node_scan_test, dense32_stride16_forward_simd) {
    run_dense_lookup(dense32_stride16, forward_starts, true, false);
}

PERF_TEST_F(dense_node_scan_test, dense32_stride16_forward_avx2) {
    run_dense_lookup(dense32_stride16, forward_starts, true, db::simd_optimization_mode::avx2);
}

PERF_TEST_F(dense_node_scan_test, dense32_stride16_forward_avx512) {
    run_dense_lookup(dense32_stride16, forward_starts, true, db::simd_optimization_mode::avx512);
}

PERF_TEST_F(dense_node_scan_test, dense32_stride16_reverse_scalar) {
    run_dense_lookup(dense32_stride16, reverse_starts, false, true);
}

PERF_TEST_F(dense_node_scan_test, dense32_stride16_reverse_simd) {
    run_dense_lookup(dense32_stride16, reverse_starts, false, false);
}

PERF_TEST_F(dense_node_scan_test, dense32_stride16_reverse_avx2) {
    run_dense_lookup(dense32_stride16, reverse_starts, false, db::simd_optimization_mode::avx2);
}

PERF_TEST_F(dense_node_scan_test, dense32_stride16_reverse_avx512) {
    run_dense_lookup(dense32_stride16, reverse_starts, false, db::simd_optimization_mode::avx512);
}

PERF_TEST_F(dense_node_scan_test, dense32_stride4_forward_scalar) {
    run_dense_lookup(dense32_stride4, forward_starts, true, true);
}

PERF_TEST_F(dense_node_scan_test, dense32_stride4_forward_simd) {
    run_dense_lookup(dense32_stride4, forward_starts, true, false);
}

PERF_TEST_F(dense_node_scan_test, dense32_stride4_reverse_scalar) {
    run_dense_lookup(dense32_stride4, reverse_starts, false, true);
}

PERF_TEST_F(dense_node_scan_test, dense32_stride4_reverse_simd) {
    run_dense_lookup(dense32_stride4, reverse_starts, false, false);
}

PERF_TEST_F(dense_node_scan_test, dense64_stride16_forward_scalar) {
    run_dense_lookup(dense64_stride16, forward_starts, true, true);
}

PERF_TEST_F(dense_node_scan_test, dense64_stride16_forward_simd) {
    run_dense_lookup(dense64_stride16, forward_starts, true, false);
}

PERF_TEST_F(dense_node_scan_test, dense64_stride16_forward_avx2) {
    run_dense_lookup(dense64_stride16, forward_starts, true, db::simd_optimization_mode::avx2);
}

PERF_TEST_F(dense_node_scan_test, dense64_stride16_forward_avx512) {
    run_dense_lookup(dense64_stride16, forward_starts, true, db::simd_optimization_mode::avx512);
}

PERF_TEST_F(dense_node_scan_test, dense64_stride16_reverse_scalar) {
    run_dense_lookup(dense64_stride16, reverse_starts, false, true);
}

PERF_TEST_F(dense_node_scan_test, dense64_stride16_reverse_simd) {
    run_dense_lookup(dense64_stride16, reverse_starts, false, false);
}

PERF_TEST_F(dense_node_scan_test, dense64_stride16_reverse_avx2) {
    run_dense_lookup(dense64_stride16, reverse_starts, false, db::simd_optimization_mode::avx2);
}

PERF_TEST_F(dense_node_scan_test, dense64_stride16_reverse_avx512) {
    run_dense_lookup(dense64_stride16, reverse_starts, false, db::simd_optimization_mode::avx512);
}

PERF_TEST_F(dense_node_scan_test, dense64_stride4_forward_scalar) {
    run_dense_lookup(dense64_stride4, forward_starts, true, true);
}

PERF_TEST_F(dense_node_scan_test, dense64_stride4_forward_simd) {
    run_dense_lookup(dense64_stride4, forward_starts, true, false);
}

PERF_TEST_F(dense_node_scan_test, dense64_stride4_reverse_scalar) {
    run_dense_lookup(dense64_stride4, reverse_starts, false, true);
}

PERF_TEST_F(dense_node_scan_test, dense64_stride4_reverse_simd) {
    run_dense_lookup(dense64_stride4, reverse_starts, false, false);
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
