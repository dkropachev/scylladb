/*
 * Copyright 2026-present ScyllaDB
 */

/*
 * SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
 */

#include "db/timeout_clock.hh"
#include "gms/inet_address_serializer.hh"
#include "idl/uuid.idl.hh"

struct timeout_config {
    db::timeout_clock::duration read_timeout;
    db::timeout_clock::duration write_timeout;
    db::timeout_clock::duration range_read_timeout;
    db::timeout_clock::duration counter_write_timeout;
    db::timeout_clock::duration truncate_timeout;
    db::timeout_clock::duration cas_timeout;
    db::timeout_clock::duration other_timeout;
};

namespace service {

struct forwarded_client_state {
    sstring keyspace;
    std::optional<sstring> username;
    timeout_config timeout_config;
    uint64_t protocol_extensions_mask;
    gms::inet_address remote_address;
    uint16_t remote_port;
    std::optional<locator::host_id> tablet_routing_source_host [[version 2026.3]] = std::nullopt;
    std::optional<unsigned> tablet_routing_source_shard [[version 2026.3]] = std::nullopt;
};

}
