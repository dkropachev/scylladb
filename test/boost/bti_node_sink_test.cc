/*
 * Copyright (C) 2024-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

#include <seastar/testing/thread_test_case.hh>
#include <seastar/util/defer.hh>

#include <algorithm>
#include <array>
#include <fmt/ranges.h>
#include <ranges>
#include "db/config.hh"
#include "test/lib/log.hh"
#include "test/lib/random_utils.hh"
#include "utils/memory_data_sink.hh"
#include "sstables/trie/bti_node_reader.hh"
#include "sstables/trie/bti_node_sink.hh"
// For instantiation of `writer_node::recalc_sizes`.
#include "sstables/trie/writer_node.impl.hh" // IWYU pragma: keep

// Calling BOOST_REQUIRE unconditionally is stupidly expensive.
// Checking the condition first, and only calling the BOOST assertions
// if the condition is false, makes the test orders of magnitude faster.
#define REQUIRE(a) do if (!(a)) BOOST_REQUIRE(a); while (0)
#define REQUIRE_EQUAL(a, b) do if (!((a) == (b))) BOOST_REQUIRE_EQUAL(a, b); while (0)
#define REQUIRE_GE(a, b) do if (!((a) >= (b))) BOOST_REQUIRE_GE(a, b); while (0)
#define REQUIRE_LE(a, b) do if (!((a) <= (b))) BOOST_REQUIRE_LE(a, b); while (0)

using namespace sstables::trie;

inline const_bytes string_as_bytes(std::string_view sv) {
    return std::as_bytes(std::span(sv.data(), sv.size()));
}

inline std::string_view bytes_as_string(const_bytes sv) {
    return {reinterpret_cast<const char*>(sv.data()), sv.size()};
}

static std::vector<std::byte> linearize(const memory_data_sink_buffers& bufs) {
    std::vector<std::byte> retval;
    for (const auto& frag : bufs.buffers()) {
        auto v = std::as_bytes(std::span(frag));
        retval.insert(retval.end(), v.begin(), v.end());
    }
    return retval;
}

std::vector<uint8_t> unpack_bitstring(const_bytes packed) {
    std::vector<uint8_t> unpacked;
    for (const auto byte : packed) {
        for (int i = 7; i >= 0; --i) {
            unpacked.push_back((uint8_t(byte) >> i) & 1);
        }
    }
    return unpacked;
}

// Test read_offset() on a random blob, by unpacking the bits
// of the blob and the bits of read_offset() and checking that
// the relevant bitstrings are equal.
SEASTAR_THREAD_TEST_CASE(test_read_offset) {
    auto test_blob_buf = tests::random::get_bytes(256);
    auto test_string = std::as_bytes(std::span(test_blob_buf));
    auto test_bitstring = unpack_bitstring(test_string);

    for (const int width : {8, 12, 16, 24, 32, 48, 56, 64}) {
        for (int idx = 0; idx * width + width <= int(test_string.size()); ++idx) {
            uint64_t result = read_offset(test_string, idx, width);
            auto read_blob = unpack_bitstring(object_representation(seastar::cpu_to_be(result)));
            auto actual_bitstring = std::span(read_blob).subspan(64 - width, width);
            auto expected_bitstring = std::span(test_bitstring).subspan(idx * width, width);
            REQUIRE(std::ranges::equal(actual_bitstring, expected_bitstring));
        }
    }
}

std::vector<std::byte> serialize_body(const writer_node& node, sink_pos pos, node_type type) {
    memory_data_sink_buffers bufs;
    constexpr size_t page_size = 4096;
    sstables::file_writer fw(data_sink(std::make_unique<memory_data_sink>(bufs)));

    bti_node_sink serializer(fw, page_size);
    auto sz = serializer.serialized_size_body_type(node, type);
    serializer.write_body(node, pos, type);
    fw.close();
    REQUIRE(fw.offset() <= page_size);
    REQUIRE(fw.offset() == uint64_t(sz.value));

    return linearize(bufs);
}

struct serialize_chain_result {
    std::vector<std::byte> serialized;
    size_t starting_point;
};
serialize_chain_result serialize_chain(const writer_node& node, node_size body_offset) {
    memory_data_sink_buffers bufs;
    constexpr size_t page_size = 4096;
    sstables::file_writer fw(data_sink(std::make_unique<memory_data_sink>(bufs)));

    bti_node_sink serializer(fw, page_size);
    auto sz = serializer.serialized_size_chain(node, body_offset);
    auto starting_point = serializer.write_chain(node, body_offset);
    fw.close();
    REQUIRE(fw.offset() <= page_size);
    REQUIRE(fw.offset() == uint64_t(sz.value));

    return {linearize(bufs), starting_point.value};
}

struct payload_result {
    uint8_t bits;
    const_bytes bytes;
};

struct deserialize_node_result {
    std::vector<uint64_t> offsets;
    std::vector<std::byte> transitions;
    payload_result payload;
};

struct lookup_result {
    int idx;
    int byte;
    uint64_t offset;
};

inline lookup_result get_child(int64_t pos, const_bytes raw, int idx, bool forward) {
    auto n_children = bti_read_node(pos, raw).n_children;
    auto found_child = bti_get_child(pos, raw, idx, forward);
    lookup_result result;
    result.idx = found_child.idx;
    result.offset = found_child.offset;
    if (0 <= result.idx && result.idx < n_children) {
        result.byte = int(bti_get_child_transition(pos, raw, result.idx));
    } else {
        result.byte = -1;
    }
    return result;
}

inline payload_result get_payload(int64_t pos, const_bytes raw) {
    auto bits = bti_read_node(pos, raw).payload_bits;
    auto bytes = bti_get_payload(pos, raw);
    return {bits, bytes};
}

inline int get_n_children(int64_t pos, const_bytes raw) {
    return bti_read_node(pos, raw).n_children;
}

deserialize_node_result deserialize_body(int64_t pos, const_bytes raw) {
    deserialize_node_result result;
    auto n_children = get_n_children(pos, raw);
    for (int i = 0; i < n_children; ++i) {
        auto child = get_child(pos, raw, i, true);
        i = child.idx;
        result.offsets.push_back(child.offset);
        REQUIRE_GE(child.byte, 0);
        result.transitions.push_back(std::byte(child.byte));
    }
    result.payload = get_payload(pos, raw);
    return result;
}

struct deserialize_chain_result {
    std::vector<std::byte> transition;
    node_size body_offset;
};

deserialize_chain_result deserialize_chain(const_bytes raw, size_t start_point) {
    REQUIRE(start_point < raw.size());
    deserialize_chain_result result;
    while (true) {
        auto n_children = get_n_children(start_point, raw.subspan(start_point));
        REQUIRE(n_children == 1);
        auto child = get_child(start_point, raw.subspan(start_point), 0, true);
        REQUIRE(child.idx == 0);
        REQUIRE(child.offset > 0);
        REQUIRE(child.offset < 4096);
        REQUIRE_GE(child.byte, 0);
        result.transition.push_back(std::byte(child.byte));
        if (child.offset > start_point) {
            REQUIRE(start_point == 0);
            result.body_offset = node_size(child.offset);
            return result;
        }
        start_point -= child.offset;
    }
}

bool eligible(node_type type, const writer_node& node, sink_pos pos) {
    auto max_offset = max_offset_from_child(node, pos);
    REQUIRE(max_offset.valid());
    auto width = std::bit_width<uint64_t>(max_offset.value);
    switch (type) {
    case PAYLOAD_ONLY:
        return node.get_children().size() == 0;
    case SINGLE_NOPAYLOAD_4:
        return node.get_children().size() == 1 && width <= 4 && node._payload._payload_bits == 0;
    case SINGLE_8:
        return node.get_children().size() == 1 && width <= 8;
    case SINGLE_NOPAYLOAD_12:
        return node.get_children().size() == 1 && width <= 12 && node._payload._payload_bits == 0;
    case SINGLE_16:
        return node.get_children().size() == 1 && width <= 16;
    case SPARSE_8:
        return node.get_children().size() >= 1 && node.get_children().size() < 256 && width <= 8;
    case SPARSE_12:
        return node.get_children().size() >= 1 && node.get_children().size() < 256 && width <= 12;
    case SPARSE_16:
        return node.get_children().size() >= 1 && node.get_children().size() < 256 && width <= 16;
    case SPARSE_24:
        return node.get_children().size() >= 1 && node.get_children().size() < 256 && width <= 24;
    case SPARSE_40:
        return node.get_children().size() >= 1 && node.get_children().size() < 256 && width <= 40;
    case DENSE_12:
        return node.get_children().size() >= 1 && width <= 12;
    case DENSE_16:
        return node.get_children().size() >= 1 && width <= 16;
    case DENSE_24:
        return node.get_children().size() >= 1 && width <= 24;
    case DENSE_32:
        return node.get_children().size() >= 1 && width <= 32;
    case DENSE_40:
        return node.get_children().size() >= 1 && width <= 40;
    case LONG_DENSE:
        return node.get_children().size() >= 1;
    default: abort();
    }
}

// Generates multiple interesting sets of child edges for a trie node,
// (which try to stress various conditionals in the encoding).
std::vector<std::vector<uint8_t>> get_some_interesting_transition_sets() {
    std::vector<std::vector<uint8_t>> result;
    // 0 children. Important edge case.
    result.push_back({});
    // 1 child, at both extremes and in the middle. (Extremes are useful to ensure that there is no weird wrapping).
    result.push_back({0x00});
    result.push_back({0x7f});
    result.push_back({0xff});
    // 2 children, at both extremes, and also with a gap between them (to test unused child slots in DENSE nodes).
    result.push_back({0x00, 0x01});
    result.push_back({0xfe, 0xff});
    result.push_back({0x00, 0xff});
    // 256 children, with all possible transition bytes.
    // Edge case.
    auto full = std::ranges::iota_view(0x00, 0x100) | std::ranges::to<std::vector<uint8_t>>();
    result.push_back(full);
    // 255 children, with all possible transition bytes except one.
    auto almost_full = full;
    almost_full.erase(almost_full.begin() + 100);
    result.push_back(almost_full);
    return result;
}

// For the given number of children and the given max supported integer width,
// generates a few sets of interesting child offsets.
// (Where "interesting" in this case means that they are just close to extremes of the supported integer
// range. This checks that no bits are lost).
std::vector<std::vector<int64_t>> get_some_interesting_child_offsets(int width, int n_children) {
    std::vector<std::vector<int64_t>> result;
    int64_t max = (int64_t(1) << width) - 1;
    auto clamp_to_legal = [=] (int64_t v) {
        return std::clamp<int64_t>(v, 1, max);
    };
    auto clamped_iota = [=] (int64_t a, int64_t b) {
        return std::ranges::iota_view(a, b) | std::views::transform(clamp_to_legal) | std::ranges::to<std::vector>();
    };
    result.emplace_back(clamped_iota(1, n_children + 1));
    result.emplace_back(clamped_iota(max - n_children + 1, max + 1));
    return result;
}

writer_node::ptr<writer_node> make_node(
    sink_pos pos,
    const_bytes transition,
    std::span<const uint8_t> child_transitions,
    std::span<const int64_t> child_offsets,
    std::optional<trie_payload> payload,
    bump_allocator& alctr
) {
    auto node = writer_node::create(transition, alctr);
    for (size_t i = 0; i < child_transitions.size(); ++i) {
        std::byte transition[] = {std::byte(child_transitions[i])};
        auto child = node->add_child(transition, alctr);
        child->_pos = pos - sink_offset(child_offsets[i]);
        child->_node_size = node_size(1);
        child->_branch_size = sink_offset(0);
        child->_first_transition_byte = transition[0];
    }
    if (payload) {
        node->set_payload(*payload);
    }
    return node;
}

void test_one_body(
    std::span<const uint8_t> child_transitions,
    std::span<const int64_t> offsets,
    std::span<const std::byte> incoming_transition,
    const std::optional<trie_payload>& payload_opt,
    sink_pos pos
) {
    testlog.trace("transitions={} offsets={} payload={}", child_transitions, offsets, bool(payload_opt));
    SCYLLA_ASSERT(child_transitions.size() == offsets.size());
    bump_allocator alctr(128 * 1024);
    auto node = make_node(pos, incoming_transition, child_transitions, offsets, payload_opt, alctr);
    for (node_type type = node_type(0); type < NODE_TYPE_COUNT; type = node_type(int(type) + 1)) {
        testlog.trace("type={}", int(type));
        if (eligible(type, *node, pos)) {
            auto serialized = serialize_body(*node, pos, type);

            testlog.trace("serialized={}", fmt_hex(serialized));
            auto deserialized = deserialize_body(pos.value, serialized);
            testlog.trace("deserialized_transitions={:x}", fmt::join(deserialized.transitions, ", "));
            REQUIRE_EQUAL(deserialized.offsets.size(), child_transitions.size());
            for (size_t i = 0; i < child_transitions.size(); ++i) {
                REQUIRE_EQUAL(child_transitions[i], int(deserialized.transitions[i]));
                REQUIRE_EQUAL(uint64_t(offsets[i]), deserialized.offsets[i]);
            }
            if (payload_opt) {
                REQUIRE_EQUAL(deserialized.payload.bits, payload_opt->_payload_bits);
                REQUIRE(std::ranges::equal(
                    deserialized.payload.bytes.subspan(0, payload_opt->blob().size()),
                    payload_opt->blob()));
            } else {
                REQUIRE_EQUAL(deserialized.payload.bits, 0);
            }

            int n_slots = 0;

            testlog.trace("Test bti_read_node");
            {
                auto result = bti_read_node(pos.value, serialized);
                REQUIRE_GE(result.n_children, child_transitions.size());
                n_slots = result.n_children;
                REQUIRE_EQUAL(result.payload_bits, payload_opt ? payload_opt->_payload_bits : 0);
            }

            testlog.trace("Test bti_walk_down_leftmost_path");
            {
                auto result = bti_walk_down_leftmost_path(pos.value, serialized);
                REQUIRE_EQUAL(result.body_pos, pos.value);
                REQUIRE_EQUAL(result.n_children, n_slots);
                REQUIRE_EQUAL(result.child_offset, offsets.empty() ? -1 : offsets.front());
                REQUIRE_EQUAL(result.payload_bits, payload_opt ? payload_opt->_payload_bits : 0);
            }

            testlog.trace("Test bti_walk_down_rightmost_path");
            {
                auto result = bti_walk_down_rightmost_path(pos.value, serialized);
                REQUIRE_EQUAL(result.body_pos, pos.value);
                REQUIRE_EQUAL(result.n_children, n_slots);
                REQUIRE_EQUAL(result.child_offset, offsets.empty() ? -1 : offsets.back());
                REQUIRE_EQUAL(result.payload_bits, payload_opt ? payload_opt->_payload_bits : 0);
            }

            testlog.trace("Test bti_walk_down_along_key");
            {
                for (int key_byte = 0; key_byte < 256; ++key_byte) {
                    //testlog.trace("key_byte={}", key_byte);
                    auto k = std::byte(key_byte);
                    auto result = bti_walk_down_along_key(pos.value, serialized, std::span<const std::byte>(&k, 1));

                    auto target_child = std::ranges::lower_bound(child_transitions, uint8_t(k)) - child_transitions.begin();
                    auto target_byte = target_child < int(child_transitions.size()) ? child_transitions[target_child] : -1;
                    auto target_offset = target_child < int(child_transitions.size()) ? offsets[target_child] : -1;

                    REQUIRE_EQUAL(result.found_byte, target_byte);
                    REQUIRE_EQUAL(result.traversed_key_bytes, 0);
                    REQUIRE_EQUAL(result.n_children, n_slots);
                    REQUIRE_LE(result.found_idx, result.n_children);
                    REQUIRE_GE(result.found_idx, 0);
                    REQUIRE_EQUAL(result.child_offset, target_offset);
                    REQUIRE_EQUAL(result.body_pos, pos.value);
                    REQUIRE_EQUAL(result.payload_bits, payload_opt ? payload_opt->_payload_bits : 0);

                    if (result.found_idx < result.n_children) {
                        auto child = get_child(pos.value, serialized, result.found_idx, true);
                        REQUIRE_EQUAL(result.found_byte, child.byte);
                        REQUIRE_EQUAL(result.found_idx, child.idx);
                        REQUIRE_EQUAL(result.child_offset, int64_t(child.offset));
                    }
                }
            }

            testlog.trace("Test bti_get_child");
            {
                int closest_occupied_slot = -1;
                int n_observed_children = 0;
                testlog.trace("Forwards");
                for (int i = n_slots - 1; i >= 0; --i) {
                    //testlog.trace("i={}", i);
                    auto result = bti_get_child(pos.value, serialized, i, true);
                    if (result.idx == i) {
                        closest_occupied_slot = i;
                        ++n_observed_children;
                        REQUIRE_EQUAL(int64_t(result.offset), offsets[offsets.size() - n_observed_children]);
                    } else {
                        REQUIRE_EQUAL(result.idx, closest_occupied_slot);
                    }
                }
                REQUIRE_EQUAL(n_observed_children, int(offsets.size()));
                testlog.trace("Backwards");
                closest_occupied_slot = -1;
                n_observed_children = 0;
                for (int i = 0; i < n_slots; ++i) {
                    //testlog.trace("i={}", i);
                    auto result = bti_get_child(pos.value, serialized, i, false);
                    if (result.idx == i) {
                        closest_occupied_slot = i;
                        ++n_observed_children;
                        REQUIRE_EQUAL(int64_t(result.offset), offsets[n_observed_children - 1]);
                    } else {
                        REQUIRE_EQUAL(result.idx, closest_occupied_slot);
                    }
                }
                REQUIRE_EQUAL(n_observed_children, int(offsets.size()));
            }
        }
    }
}

void test_dense_lookup_shape(
    node_type type,
    std::span<const uint8_t> child_transitions,
    std::span<const int64_t> offsets,
    sink_pos pos
) {
    auto whatever = string_as_bytes("dense lookup shape");
    bump_allocator alctr(128 * 1024);
    auto node = make_node(pos, whatever, child_transitions, offsets, std::nullopt, alctr);
    auto serialized = serialize_body(*node, pos, type);
    auto first_transition = child_transitions.front();
    auto n_slots = child_transitions.back() - first_transition + 1;

    {
        auto result = bti_read_node(pos.value, serialized);
        REQUIRE_EQUAL(result.n_children, n_slots);
    }

    for (int slot = 0; slot < n_slots; ++slot) {
        auto transition = uint8_t(first_transition + slot);
        auto next_child = std::ranges::lower_bound(child_transitions, transition) - child_transitions.begin();
        auto previous_child = std::ranges::upper_bound(child_transitions, transition) - child_transitions.begin() - 1;

        auto next = bti_get_child(pos.value, serialized, slot, true);
        REQUIRE_EQUAL(next.idx, child_transitions[next_child] - first_transition);
        REQUIRE_EQUAL(next.offset, uint64_t(offsets[next_child]));

        auto previous = bti_get_child(pos.value, serialized, slot, false);
        REQUIRE_EQUAL(previous.idx, child_transitions[previous_child] - first_transition);
        REQUIRE_EQUAL(previous.offset, uint64_t(offsets[previous_child]));
    }

    for (int key_byte = 0; key_byte < 256; ++key_byte) {
        auto key = std::byte(key_byte);
        auto result = bti_walk_down_along_key(pos.value, serialized, std::span<const std::byte>(&key, 1));
        auto target_child = std::ranges::lower_bound(child_transitions, uint8_t(key_byte)) - child_transitions.begin();
        if (target_child < int(child_transitions.size())) {
            REQUIRE_EQUAL(result.found_idx, child_transitions[target_child] - first_transition);
            REQUIRE_EQUAL(result.found_byte, child_transitions[target_child]);
            REQUIRE_EQUAL(result.child_offset, offsets[target_child]);
        } else {
            REQUIRE_EQUAL(result.found_idx, n_slots);
            REQUIRE_EQUAL(result.found_byte, -1);
            REQUIRE_EQUAL(result.child_offset, -1);
        }
    }
}

// Covers #5: SIMD dense-node scans for optimized packed offset widths,
// including first, middle, last, and missing child transitions.
SEASTAR_THREAD_TEST_CASE(test_dense_lookup_optimized_offset_widths) {
    const auto previous_mode = get_bti_dense_node_simd_optimization_mode();
    auto restore_mode = defer([previous_mode] {
        set_bti_dense_node_simd_optimization_mode(previous_mode);
    });

    const std::array<uint8_t, 4> transitions_with_full_span = {0x00, 0x1f, 0x7f, 0xff};
    const std::array<uint8_t, 3> transitions_with_tail_miss = {0x00, 0x1f, 0x7f};

    const std::array<int64_t, 4> offsets_16 = {1, 17, 1024, (int64_t(1) << 16) - 1};
    const std::array<int64_t, 3> offsets_16_tail = {1, 17, 1024};
    const std::array<int64_t, 4> offsets_32 = {1, int64_t(1) << 16, int64_t(1) << 24, (int64_t(1) << 32) - 1};
    const std::array<int64_t, 3> offsets_32_tail = {1, int64_t(1) << 16, int64_t(1) << 24};
    const std::array<int64_t, 4> offsets_64 = {1, int64_t(1) << 40, int64_t(1) << 48, int64_t(1) << 55};
    const std::array<int64_t, 3> offsets_64_tail = {1, int64_t(1) << 40, int64_t(1) << 48};

    for (auto mode : {
            db::simd_optimization_mode::automatic,
            db::simd_optimization_mode::off,
            db::simd_optimization_mode::sse,
            db::simd_optimization_mode::avx2,
            db::simd_optimization_mode::avx512,
            db::simd_optimization_mode::neon,
            db::simd_optimization_mode::sve}) {
        set_bti_dense_node_simd_optimization_mode(mode);
        test_dense_lookup_shape(DENSE_16, transitions_with_full_span, offsets_16, sink_pos((int64_t(1) << 16) + 4096));
        test_dense_lookup_shape(DENSE_16, transitions_with_tail_miss, offsets_16_tail, sink_pos((int64_t(1) << 16) + 4096));
        test_dense_lookup_shape(DENSE_32, transitions_with_full_span, offsets_32, sink_pos((int64_t(1) << 32) + 4096));
        test_dense_lookup_shape(DENSE_32, transitions_with_tail_miss, offsets_32_tail, sink_pos((int64_t(1) << 32) + 4096));
        test_dense_lookup_shape(LONG_DENSE, transitions_with_full_span, offsets_64, sink_pos((int64_t(1) << 56) + 4096));
        test_dense_lookup_shape(LONG_DENSE, transitions_with_tail_miss, offsets_64_tail, sink_pos((int64_t(1) << 56) + 4096));
    }
}

// Tests the encoding of `writer_node`'s "body"
// (see the comment at the declaration fo bti_node_sink::write_body for what "body" means).
//
// Tries to cover all BTI node types and interesting node "shapes"
// (e.g. with and without gaps in the list of occupied child indexes).
//
// Generates various "interesting" nodes. For each such node, it BTI-encodes it,
// and checks that all BTI node traversal routines give the expected result
// on it.
SEASTAR_THREAD_TEST_CASE(test_body) {
    std::vector<std::vector<uint8_t>> interesting_transition_sets = get_some_interesting_transition_sets();

    // Arbitrary, but large enough to cover all interesting widths.
    auto pos = sink_pos((uint64_t(1) << 60) + 1);
    auto whatever = string_as_bytes("hahaha");
    const auto custom_payload = trie_payload(0x7, string_as_bytes("lololo"));

    for (int width = 1; width < 50; ++width)
    for (const auto& child_transitions : interesting_transition_sets)
    for (const auto& offsets : get_some_interesting_child_offsets(width, child_transitions.size()))
    for (bool payload : {true, false}) {
        auto payload_opt = payload ? std::optional<trie_payload>(custom_payload) : std::optional<trie_payload>();
        test_one_body(child_transitions, offsets, whatever, payload_opt, pos);
    }
}

static std::vector<uint8_t> make_evenly_spaced_transitions(int n_children) {
    std::vector<uint8_t> transitions;
    transitions.reserve(n_children);
    for (int i = 0; i < n_children; ++i) {
        transitions.push_back(i * 256 / n_children);
    }
    return transitions;
}

// Covers the sparse transition lookup thresholds and SIMD lane boundaries from
// https://github.com/dkropachev/scylladb/issues/14.
SEASTAR_THREAD_TEST_CASE(test_sparse_walk_down_along_key_boundaries) {
    const auto previous_mode = get_bti_sparse_node_simd_optimization_mode();
    auto restore_mode = defer([previous_mode] {
        set_bti_sparse_node_simd_optimization_mode(previous_mode);
    });

    constexpr std::array sparse_types = {SPARSE_8, SPARSE_12, SPARSE_16, SPARSE_24, SPARSE_40};
    constexpr std::array child_counts = {1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255};

    const auto pos = sink_pos((uint64_t(1) << 40) + 255);
    const auto incoming_transition = string_as_bytes("issue-14");

    for (auto mode : {
            db::simd_optimization_mode::automatic,
            db::simd_optimization_mode::off,
            db::simd_optimization_mode::sse,
            db::simd_optimization_mode::avx2,
            db::simd_optimization_mode::avx512,
            db::simd_optimization_mode::neon,
            db::simd_optimization_mode::sve})
    for (const auto type : sparse_types)
    for (const auto n_children : child_counts) {
        set_bti_sparse_node_simd_optimization_mode(mode);
        auto child_transitions = make_evenly_spaced_transitions(n_children);
        std::vector<int64_t> offsets;
        offsets.reserve(n_children);
        for (int i = 0; i < n_children; ++i) {
            offsets.push_back(i + 1);
        }

        bump_allocator alctr(128 * 1024);
        auto node = make_node(pos, incoming_transition, child_transitions, offsets, std::nullopt, alctr);
        auto serialized = serialize_body(*node, pos, type);

        std::array<std::byte, 256> batch_keys;
        std::array<node_traverse_result, 256> batch_results;
        for (int i = 0; i < 256; ++i) {
            batch_keys[i] = std::byte((i * 73) & 0xff);
        }
        bti_walk_down_along_key_batch(pos.value, serialized, const_bytes(batch_keys.data(), batch_keys.size()), batch_results);
        for (int i = 0; i < 256; ++i) {
            auto key = batch_keys[i];
            auto result = batch_results[i];

            auto target_child = std::ranges::lower_bound(child_transitions, uint8_t(key)) - child_transitions.begin();
            auto target_byte = target_child < n_children ? child_transitions[target_child] : -1;
            auto target_offset = target_child < n_children ? offsets[target_child] : -1;

            REQUIRE_EQUAL(result.found_idx, target_child);
            REQUIRE_EQUAL(result.found_byte, target_byte);
            REQUIRE_EQUAL(result.child_offset, target_offset);
            REQUIRE_EQUAL(result.traversed_key_bytes, 0);
            REQUIRE_EQUAL(result.n_children, n_children);
            REQUIRE_EQUAL(result.body_pos, pos.value);
            REQUIRE_EQUAL(result.payload_bits, 0);
        }

        std::array small_batch_keys = {std::byte(0x00), std::byte(0x01), std::byte(0x7f), std::byte(0x80), std::byte(0xfe), std::byte(0xff)};
        std::array<node_traverse_result, 6> small_batch_results;
        bti_walk_down_along_key_batch(pos.value, serialized, const_bytes(small_batch_keys.data(), small_batch_keys.size()), small_batch_results);
        for (size_t i = 0; i < small_batch_keys.size(); ++i) {
            auto key = small_batch_keys[i];
            auto result = small_batch_results[i];
            auto expected = bti_walk_down_along_key(pos.value, serialized, std::span<const std::byte>(&key, 1));

            REQUIRE_EQUAL(result.found_idx, expected.found_idx);
            REQUIRE_EQUAL(result.found_byte, expected.found_byte);
            REQUIRE_EQUAL(result.child_offset, expected.child_offset);
            REQUIRE_EQUAL(result.traversed_key_bytes, expected.traversed_key_bytes);
            REQUIRE_EQUAL(result.n_children, expected.n_children);
            REQUIRE_EQUAL(result.body_pos, expected.body_pos);
            REQUIRE_EQUAL(result.payload_bits, expected.payload_bits);
        }

        for (int key_byte = 0; key_byte < 256; ++key_byte) {
            auto key = std::byte(key_byte);
            auto result = bti_walk_down_along_key(pos.value, serialized, std::span<const std::byte>(&key, 1));

            auto target_child = std::ranges::lower_bound(child_transitions, uint8_t(key)) - child_transitions.begin();
            auto target_byte = target_child < n_children ? child_transitions[target_child] : -1;
            auto target_offset = target_child < n_children ? offsets[target_child] : -1;

            REQUIRE_EQUAL(result.found_idx, target_child);
            REQUIRE_EQUAL(result.found_byte, target_byte);
            REQUIRE_EQUAL(result.child_offset, target_offset);
            REQUIRE_EQUAL(result.traversed_key_bytes, 0);
            REQUIRE_EQUAL(result.n_children, n_children);
            REQUIRE_EQUAL(result.body_pos, pos.value);
            REQUIRE_EQUAL(result.payload_bits, 0);
        }
    }
}

static uint64_t bitwidth_mask(uint64_t width) {
    if (width == 0) {
        return 0;
    }
    return uint64_t(-1) >> (64 - width);
}

static uint64_t get_random_int_bitwidth_weighted(uint64_t max) {
    if (max == 0) {
        return 0;
    }
    auto x = tests::random::get_int<uint64_t>(1, max, tests::random::gen());
    auto max_width = std::bit_width(x);
    auto width = tests::random::get_int<uint64_t>(1, max_width, tests::random::gen());
    return x & bitwidth_mask(width);
}

// Like `test_body` but with randomized parameters.
SEASTAR_THREAD_TEST_CASE(test_body_randomized) {
    for (uint64_t trial = 0; trial < 1337; ++trial) {
        auto n_children = get_random_int_bitwidth_weighted(256);

        std::vector<int64_t> offsets;
        auto max_offset_width = tests::random::get_int<uint64_t>(1, 63, tests::random::gen());
        for (uint64_t i = 0; i < n_children; ++i) {
            auto off = get_random_int_bitwidth_weighted(bitwidth_mask(max_offset_width));
            offsets.push_back(std::max<int64_t>(1, off));
        }

        std::array<uint8_t, 256> possible_transitions;
        std::ranges::iota(possible_transitions, 0);

        std::array<uint8_t, 256> transitions_buf;
        auto child_transitions = std::span(transitions_buf.begin(), n_children);
        std::ranges::sample(possible_transitions, child_transitions.begin(), child_transitions.size(), tests::random::gen());
        std::ranges::sort(child_transitions);

        std::optional<trie_payload> payload_opt;
        bool has_payload = tests::random::get_int<int>(0, 1, tests::random::gen());
        if (has_payload) {
            auto bits = tests::random::get_int<int>(1, 15, tests::random::gen());
            auto length = tests::random::get_int<int>(1, trie_payload::MAX_PAYLOAD_SIZE, tests::random::gen());
            auto bytes = tests::random::get_bytes(length);
            payload_opt.emplace(bits, std::as_bytes(std::span(bytes)));
        }

        auto max_offset = offsets.size() ? std::ranges::max(offsets) : 0;
        auto pos = tests::random::get_int<int64_t>(max_offset, std::numeric_limits<int64_t>::max(), tests::random::gen());

        // The transition doesn't matter for body serialization.
        auto incoming_transition = string_as_bytes("hahaha");

        test_one_body(child_transitions, offsets, incoming_transition, payload_opt, sink_pos(pos));
    }
}

// Tests the encoding of `writer_node`'s "chain"
// (see the comment at the declaration fo bti_node_sink::write_body for what "chain" means).
// Tries to cover all BTI node types and interesting node "shapes"
// (e.g. with and without gaps in the list of occupied child indexes).
SEASTAR_THREAD_TEST_CASE(test_chain) {
    std::vector<bytes> interesting_transitions;
    // 65 is supposed to be long enough to cover the SIMD code paths.
    for (int i = 2; i < 65; ++i) {
        interesting_transitions.push_back(tests::random::get_bytes(i));
    }

    std::vector<node_size> interesting_body_offsets;
    interesting_body_offsets.emplace_back(1);
    interesting_body_offsets.emplace_back(2);
    interesting_body_offsets.emplace_back(15);
    interesting_body_offsets.emplace_back(16);
    interesting_body_offsets.emplace_back(17);

    auto test_one = [] (const_bytes transition, node_size body_offset) {
        bump_allocator alctr(128 * 1024);
        auto node = writer_node::create(transition, alctr);
        auto [serialized, starting_point] = serialize_chain(*node, body_offset);
        auto deserialized = deserialize_chain(serialized, starting_point);
        REQUIRE(deserialized.body_offset.value == body_offset.value);
        REQUIRE(std::ranges::equal(deserialized.transition, transition.subspan(1)));
    };

    for (const auto& off : interesting_body_offsets) {
        auto some_transition = std::as_bytes(std::span(interesting_transitions.back()));
        test_one(some_transition, off);
    }
    for (const auto& transition : interesting_transitions) {
        auto transition_view = std::as_bytes(std::span(transition));
        auto some_offset = interesting_body_offsets.back();
        test_one(transition_view, some_offset);
    }
}

// Reproduces an issue with `max_offset_from_child` which I encountered
// during development and which wasn't detected by the previous unit tests.
// (It was only detected by randomized integration tests.
// `mutation_source_test`, IIRC).
//
// Specifically: I forgot to add an `if` for the special case when
// `chain_length == 2 && child->_node_size >= 16`.
// and this was only detected by random integration tests.
// The test hits that branch and would fail without it.
SEASTAR_THREAD_TEST_CASE(test_max_offset_from_child_consistent_across_write) {
    auto test_one = [] (const_bytes transition, size_t payload_size) {
        testlog.trace("Testing transition={} payload_size={}", fmt_hex(transition), payload_size);
        // Dummy output stream.
        memory_data_sink_buffers bufs;
        sstables::file_writer fw(data_sink(std::make_unique<memory_data_sink>(bufs)));
        fw.write(to_bytes_view(std::string_view("Let's offset the stream to make the test just a bit more general.")));

        // BTI node serializer.
        constexpr size_t page_size = 4096;
        bump_allocator alctr(128 * 1024);
        bti_node_sink serializer(fw, page_size);
        auto starting_pos = serializer.pos();

        // Create a parent and a child. Their node size and branch size are uninitialized.
        auto parent = writer_node::create(string_as_bytes("abc"), alctr);
        auto child = parent->add_child(transition, alctr);
        child->set_payload(trie_payload(1, string_as_bytes(std::string(payload_size, 'z'))));

        // max_offset_from_child needs to know the sizes (node size, branch size) of children.
        // We could compute them manually, but we can use `recalc_sizes` too.
        // But we have to set `_has_out_of_page_descendants`` to force `recalc_sizes`
        // to reculculate the sizes.
        child->_has_out_of_page_descendants = true;
        auto child_size = writer_node::recalc_sizes(child, serializer, starting_pos);
        child->_has_out_of_page_descendants = false;

        // Check that the size prediction done by max_offset_from_child
        // is consistent with the actual distance to the child after it is written.
        auto before = max_offset_from_child(*parent, serializer.pos() + child_size);
        writer_node::write(child, serializer, true);
        auto after = max_offset_from_child(*parent, serializer.pos());
        REQUIRE_EQUAL(before.value, after.value);
    };

    // Pick various transition lenghts and payload sizes to exercise all
    // branches in max_offset_from_child()
    std::vector<bytes> interesting_transitions;
    for (int i = 1; i < 32; ++i) {
        interesting_transitions.push_back(tests::random::get_bytes(i));
    }
    std::vector<size_t> interesting_payload_sizes;
    for (size_t i = 1; i <= 20; ++i) {
        interesting_payload_sizes.push_back(i);
    }

    for (const auto& transition : interesting_transitions)
    for (const auto& payload_size : interesting_payload_sizes) {
        auto transition_view = std::as_bytes(std::span(transition));
        test_one(transition_view, payload_size);
    }
}
