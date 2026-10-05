/*
 * Copyright (C) 2026-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

#pragma once

#include <atomic>
#include <optional>

#include "gc_clock.hh"
#include "schema/schema.hh"

namespace db::system_table_response_cache {

// CQL response bodies are cached on coordinator shards. Replica writes can run on
// another shard, so a process-wide generation protects every shard's entries.
struct state {
    std::atomic<uint64_t> generation{0};
    std::atomic<uint32_t> writers{0};
};

inline state global_state;

/// Whether writes to this table must invalidate cached CQL response bodies.
inline bool covers(const schema& s) noexcept {
    return s.ks_name() == "system" && (s.cf_name() == "local" || s.cf_name() == "peers");
}

/// Invalidate cached responses across shards while a covered table is being changed.
/// The guard spans the write so readers cannot capture a stable generation mid-write.
class write_guard {
    bool _covered;
public:
    explicit write_guard(const schema& s) noexcept : _covered(covers(s)) {
        if (_covered) {
            global_state.writers.fetch_add(1);
            global_state.generation.fetch_add(1);
        }
    }

    write_guard(const write_guard&) = delete;
    write_guard& operator=(const write_guard&) = delete;

    ~write_guard() {
        if (_covered) {
            global_state.generation.fetch_add(1);
            global_state.writers.fetch_sub(1);
        }
    }
};

/// Return the current generation only when no covered-table write is in progress.
inline std::optional<uint64_t> stable_generation() noexcept {
    const auto generation = global_state.generation.load();
    if (global_state.writers.load() || global_state.generation.load() != generation) {
        return std::nullopt;
    }
    return generation;
}

/// The table generation and gc_clock second used to validate a cached response.
struct snapshot {
    uint64_t generation;
    gc_clock::time_point second;

    /// Whether neither a covered-table write nor a TTL second boundary has passed.
    bool valid() const noexcept {
        return gc_clock::now() == second && stable_generation() == generation;
    }
};

}
