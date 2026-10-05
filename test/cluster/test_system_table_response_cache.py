# Copyright (C) 2026-present ScyllaDB
# SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1

import asyncio
from uuid import uuid4

import pytest
from cassandra import ConsistencyLevel

from test.pylib.scylla_cluster_manager import ScyllaClusterManager
from test.pylib.rest_client import inject_error_one_shot


async def test_prepared_system_table_responses_follow_writes(manager: ScyllaClusterManager) -> None:
    """Cached prepared responses reflect writes to system.local and system.peers."""
    await manager.server_add()
    cql = manager.get_cql()

    local = cql.prepare("SELECT data_center, rack, host_id, rpc_address FROM system.local WHERE key = ?")
    peers = cql.prepare("SELECT data_center, rack, host_id, peer FROM system.peers")
    local.consistency_level = ConsistencyLevel.ONE
    peers.consistency_level = ConsistencyLevel.ONE
    original_dc = (await cql.run_async(local, ("local",)))[0].data_center

    try:
        for _ in range(20):
            assert (await cql.run_async(local, ("local",)))[0].data_center == original_dc
        assert await cql.run_async(local, ("unknown",)) == []

        await cql.run_async("UPDATE system.local SET data_center = 'response_cache_test' WHERE key = 'local'")
        for _ in range(20):
            assert (await cql.run_async(local, ("local",)))[0].data_center == "response_cache_test"

        peer = "127.1.1.1"
        await cql.run_async(
            "INSERT INTO system.peers (peer, host_id, data_center, rack) VALUES (%s, %s, %s, %s)",
            (peer, uuid4(), "dc1", "rack1"),
        )
        for _ in range(20):
            assert any(str(row.peer) == peer and row.rack == "rack1" for row in await cql.run_async(peers))

        await cql.run_async("UPDATE system.peers SET rack = 'rack2' WHERE peer = %s", (peer,))
        for _ in range(20):
            assert any(str(row.peer) == peer and row.rack == "rack2" for row in await cql.run_async(peers))

        await cql.run_async("DELETE FROM system.peers WHERE peer = %s", (peer,))
        for _ in range(20):
            assert all(str(row.peer) != peer for row in await cql.run_async(peers))
    finally:
        await cql.run_async("UPDATE system.local SET data_center = %s WHERE key = 'local'", (original_dc,))
        await cql.run_async("DELETE FROM system.peers WHERE peer = '127.1.1.1'")


@pytest.mark.skip_mode(mode="release", reason="error injections are not supported in release mode")
@pytest.mark.parametrize("pause_at", ["before_validation", "after_validation"])
async def test_cached_response_rechecked_after_write(manager: ScyllaClusterManager, pause_at: str) -> None:
    """A write while a cached request is paused must invalidate its saved body."""
    server = await manager.server_add()
    cql = manager.get_cql()
    host = cql.cluster.metadata.get_host(server.ip_addr)
    local = cql.prepare("SELECT data_center FROM system.local WHERE key = ?")
    local.consistency_level = ConsistencyLevel.ONE
    original_dc = (await cql.run_async(local, ("local",), host=host))[0].data_center

    try:
        for _ in range(20):
            await cql.run_async(local, ("local",), host=host)

        injection_name = f"cached_system_table_response_{pause_at}"
        injection = await inject_error_one_shot(manager.api, server.ip_addr, injection_name)
        entered = asyncio.create_task(manager.api.wait_for_injection_enter(server.ip_addr, injection_name))
        try:
            while True:
                pending = asyncio.ensure_future(cql.run_async(local, ("local",), host=host))
                done, _ = await asyncio.wait((pending, entered), return_when=asyncio.FIRST_COMPLETED)
                if entered in done:
                    await entered
                    break
                await pending
            await cql.run_async("UPDATE system.local SET data_center = 'response_cache_race' WHERE key = 'local'", host=host)
        finally:
            await injection.message()

        assert (await pending)[0].data_center == "response_cache_race"
    finally:
        await cql.run_async("UPDATE system.local SET data_center = %s WHERE key = 'local'", (original_dc,), host=host)


async def test_cached_response_counts_as_select(manager: ScyllaClusterManager) -> None:
    """Cached prepared reads must count toward CQL SELECT metrics."""
    server = await manager.server_add()
    cql = manager.get_cql()
    host = cql.cluster.metadata.get_host(server.ip_addr)
    local = cql.prepare("SELECT data_center FROM system.local WHERE key = ?")
    local.consistency_level = ConsistencyLevel.ONE
    await cql.run_async(local, ("local",), host=host)

    before = await manager.metrics.query(server.ip_addr)
    for _ in range(20):
        await cql.run_async(local, ("local",), host=host)
    after = await manager.metrics.query(server.ip_addr)

    assert (after.get("scylla_transport_system_table_response_cache_hits") or 0) > (before.get("scylla_transport_system_table_response_cache_hits") or 0), \
            after.lines_by_prefix("scylla_transport_system_table")
    assert (after.get("scylla_cql_reads_per_ks", {"ks": "system", "who": "user"}) or 0) >= \
            (before.get("scylla_cql_reads_per_ks", {"ks": "system", "who": "user"}) or 0) + 20
