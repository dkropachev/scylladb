#
# Copyright (C) 2026-present ScyllaDB
#
# SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1
#

import asyncio
import logging
import pathlib

import pytest

from test.pylib.host_registry import Host
from test.pylib.internal_types import IPAddress, ServerInfo, ServerNum
from test.pylib.scylla_cluster import ScyllaCluster


class FakeHostRegistry:
    def __init__(self) -> None:
        self.next_host_id = 0
        self.released_hosts: list[Host] = []

    async def lease_host(self) -> Host:
        self.next_host_id += 1
        return Host(f"127.0.0.{self.next_host_id}")

    async def release_host(self, host: Host) -> None:
        self.released_hosts.append(host)


class BlockingHostRegistry(FakeHostRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.lease_started = asyncio.Event()
        self.finish_lease = asyncio.Event()

    async def lease_host(self) -> Host:
        self.lease_started.set()
        await self.finish_lease.wait()
        return await super().lease_host()


class SlowInstallingServer:
    def __init__(self, server_id: ServerNum, ip_addr: IPAddress, seeds: list[IPAddress]) -> None:
        self.server_id = server_id
        self.ip_addr = ip_addr
        self.rpc_address = ip_addr
        self.seeds = seeds
        self.datacenter = "dc1"
        self.rack = "rack1"
        self.workdir = pathlib.Path(f"server-{server_id}")
        self.install_started = asyncio.Event()
        self.finish_install = asyncio.Event()
        self.pause_after_before_start = False
        self.before_start_checked = asyncio.Event()
        self.continue_after_before_start = asyncio.Event()
        self.start_finished = asyncio.Event()
        self.block_stop_until_released = False
        self.first_stop_started = asyncio.Event()
        self.second_stop_started = asyncio.Event()
        self.finish_stop = asyncio.Event()
        self.stop_finished = asyncio.Event()
        self.uninstall_started = asyncio.Event()
        self.install_finished = False
        self.uninstalled_before_install_finished = False
        self.started = False
        self.stop_calls = 0
        self.stop_gracefully_calls = 0

    async def install(self) -> None:
        self.install_started.set()
        await self.finish_install.wait()
        self.install_finished = True

    async def install_and_start(self, api, expected_error=None, expected_server_up_state=None, before_start=None) -> None:
        await self.install()
        try:
            if before_start is not None:
                before_start()
            self.before_start_checked.set()
            if self.pause_after_before_start:
                await self.continue_after_before_start.wait()
            if before_start is not None:
                before_start()
            self.started = True
            self.start_finished.set()
        except BaseException:
            await self.stop()
            raise

    async def stop(self) -> None:
        self.stop_calls += 1
        if self.block_stop_until_released:
            if self.stop_calls == 1:
                self.first_stop_started.set()
            elif self.stop_calls == 2:
                self.second_stop_started.set()
            await self.finish_stop.wait()
        self.started = False
        self.stop_finished.set()

    async def stop_gracefully(self) -> None:
        self.stop_gracefully_calls += 1
        self.started = False

    async def uninstall(self) -> None:
        self.uninstall_started.set()
        self.uninstalled_before_install_finished = not self.install_finished

    def server_info(self) -> ServerInfo:
        return ServerInfo(self.server_id, self.ip_addr, self.rpc_address, self.datacenter, self.rack, 0)


def make_cluster(host_registry: FakeHostRegistry | None = None) -> tuple[ScyllaCluster, FakeHostRegistry, dict[str, SlowInstallingServer]]:
    if host_registry is None:
        host_registry = FakeHostRegistry()
    created_servers: dict[str, SlowInstallingServer] = {}
    next_server_id = 0

    def create_server(params: ScyllaCluster.CreateServerParams) -> SlowInstallingServer:
        nonlocal next_server_id
        next_server_id += 1
        server = SlowInstallingServer(ServerNum(next_server_id), params.ip_addr, list(params.seeds))
        created_servers["server"] = server
        created_servers[f"server{next_server_id}"] = server
        return server

    cluster = ScyllaCluster(logging.getLogger("test_scylla_cluster"), host_registry, replicas=0, create_server=create_server)
    cluster.is_running = True
    return cluster, host_registry, created_servers


async def start_adding_server(cluster: ScyllaCluster, created_servers: dict[str, SlowInstallingServer]) -> tuple[asyncio.Task, SlowInstallingServer]:
    add_task = asyncio.create_task(cluster.add_server())
    while "server" not in created_servers:
        await asyncio.sleep(0)
    server = created_servers["server"]
    await server.install_started.wait()
    assert cluster.starting[server.server_id] is server
    return add_task, server


async def wait_for_add_task_done_or_server_created(add_task: asyncio.Task, created_servers: dict[str, SlowInstallingServer]) -> None:
    while not add_task.done() and "server" not in created_servers:
        await asyncio.sleep(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_running", [False, True])
async def test_cluster_stop_cancels_add_server_waiting_for_ip(is_running: bool) -> None:
    """Regression test for stopping a cluster while add_server() is still waiting for an IP lease."""
    host_registry = BlockingHostRegistry()
    cluster, _, created_servers = make_cluster(host_registry)
    cluster.is_running = is_running

    add_task = asyncio.create_task(cluster.add_server())
    await host_registry.lease_started.wait()

    await cluster.stop()
    assert "server" not in created_servers

    host_registry.finish_lease.set()
    await wait_for_add_task_done_or_server_created(add_task, created_servers)
    if "server" in created_servers:
        created_servers["server"].finish_install.set()

    with pytest.raises(RuntimeError, match="stopped while .* being added"):
        await add_task

    assert "server" not in created_servers
    assert not cluster.running
    assert not cluster.stopped
    assert not cluster.starting
    assert not cluster.leased_ips
    assert host_registry.released_hosts == [Host("127.0.0.1")]


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_method", ["stop", "stop_gracefully"])
async def test_cluster_stop_cancels_server_still_installing(stop_method: str) -> None:
    """Regression test for stopping a cluster while add_server() is still installing a server."""
    cluster, host_registry, created_servers = make_cluster()
    add_task, server = await start_adding_server(cluster, created_servers)

    await getattr(cluster, stop_method)()
    assert not cluster.starting

    server.finish_install.set()
    with pytest.raises(RuntimeError, match="stopped while it was being added"):
        await add_task

    assert not server.started
    assert not cluster.running
    assert not cluster.starting
    assert cluster.stopped[server.server_id] is server
    assert not cluster.leased_ips
    assert host_registry.released_hosts == [Host(server.ip_addr)]


@pytest.mark.asyncio
async def test_cluster_stop_cancels_server_still_installing_before_cluster_is_running() -> None:
    """Regression test for cleanup racing with initial cluster startup before is_running is set."""
    cluster, host_registry, created_servers = make_cluster()
    cluster.is_running = False
    add_task, server = await start_adding_server(cluster, created_servers)

    await cluster.stop()
    assert not cluster.starting

    server.finish_install.set()
    with pytest.raises(RuntimeError, match="stopped while it was being added"):
        await add_task

    assert not server.started
    assert not cluster.running
    assert not cluster.starting
    assert cluster.stopped[server.server_id] is server
    assert not cluster.leased_ips
    assert host_registry.released_hosts == [Host(server.ip_addr)]


@pytest.mark.asyncio
async def test_cluster_stop_after_start_check_still_prevents_server_from_running() -> None:
    """Regression test for stop landing after add_server() checks starting, but before start completes."""
    cluster, host_registry, created_servers = make_cluster()
    add_task, server = await start_adding_server(cluster, created_servers)
    server.pause_after_before_start = True

    server.finish_install.set()
    await server.before_start_checked.wait()
    await cluster.stop()
    assert not cluster.starting

    server.continue_after_before_start.set()
    with pytest.raises(RuntimeError, match="stopped while it was being added"):
        await add_task

    assert not server.started
    assert not cluster.running
    assert not cluster.starting
    assert cluster.stopped[server.server_id] is server
    assert not cluster.leased_ips
    assert host_registry.released_hosts == [Host(server.ip_addr)]


@pytest.mark.asyncio
async def test_cluster_stop_after_start_check_does_not_start_after_releasing_ip() -> None:
    """Regression test for cleanup releasing the IP before add_server() leaves the start path."""
    cluster, host_registry, created_servers = make_cluster()
    add_task, server = await start_adding_server(cluster, created_servers)
    server.pause_after_before_start = True

    server.finish_install.set()
    await server.before_start_checked.wait()
    await cluster.stop()
    await cluster.release_ips()

    server.continue_after_before_start.set()
    with pytest.raises(RuntimeError, match="stopped while it was being added"):
        await add_task

    assert not server.start_finished.is_set()
    assert not server.started
    assert not cluster.running
    assert not cluster.starting
    assert cluster.stopped[server.server_id] is server
    assert not cluster.leased_ips
    assert host_registry.released_hosts == [Host(server.ip_addr)]


@pytest.mark.asyncio
async def test_server_stop_cancels_server_still_installing() -> None:
    """Regression test for stopping a specific starting server before its process exists."""
    cluster, host_registry, created_servers = make_cluster()
    add_task, server = await start_adding_server(cluster, created_servers)

    await cluster.server_stop(server.server_id, gracefully=False)
    assert not cluster.starting
    assert cluster.stopped[server.server_id] is server

    server.finish_install.set()
    with pytest.raises(RuntimeError, match="stopped while it was being added"):
        await add_task

    assert not server.started
    assert not cluster.running
    assert not cluster.starting
    assert cluster.stopped[server.server_id] is server
    assert not cluster.leased_ips
    assert host_registry.released_hosts == [Host(server.ip_addr)]


@pytest.mark.asyncio
async def test_cancelled_initial_add_does_not_seed_next_server_with_released_ip() -> None:
    """Regression test for a cancelled first add leaving initial_seed set to its released IP."""
    cluster, host_registry, created_servers = make_cluster()
    add_task, first_server = await start_adding_server(cluster, created_servers)

    await cluster.server_stop(first_server.server_id, gracefully=False)
    first_server.finish_install.set()
    with pytest.raises(RuntimeError, match="stopped while it was being added"):
        await add_task

    second_add_task = asyncio.create_task(cluster.add_server())
    while "server2" not in created_servers:
        await asyncio.sleep(0)
    second_server = created_servers["server2"]
    second_server.finish_install.set()
    await second_add_task

    assert second_server.seeds == [second_server.ip_addr]
    assert cluster.initial_seed == second_server.ip_addr
    assert host_registry.released_hosts == [Host(first_server.ip_addr)]


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_action", ["server_stop", "cluster_stop"])
async def test_stop_cancels_starting_server_before_stop_finishes(stop_action: str) -> None:
    """Regression test for stop waiting on an in-progress start before it can remove starting."""
    cluster, host_registry, created_servers = make_cluster()
    add_task, server = await start_adding_server(cluster, created_servers)
    server.block_stop_until_released = True

    if stop_action == "server_stop":
        stop_task = asyncio.create_task(cluster.server_stop(server.server_id, gracefully=False))
    else:
        stop_task = asyncio.create_task(cluster.stop())
    await server.first_stop_started.wait()

    server.finish_install.set()
    second_stop_task = asyncio.create_task(server.second_stop_started.wait())
    start_finished_task = asyncio.create_task(server.start_finished.wait())
    _, pending = await asyncio.wait({second_stop_task, start_finished_task}, return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)

    if not server.second_stop_started.is_set():
        server.finish_stop.set()
        await asyncio.gather(add_task, stop_task, return_exceptions=True)
    assert server.second_stop_started.is_set()

    server.finish_stop.set()
    with pytest.raises(RuntimeError, match="stopped while it was being added"):
        await add_task
    await stop_task

    assert not server.started
    assert not cluster.running
    assert not cluster.starting
    assert cluster.stopped[server.server_id] is server
    assert not cluster.leased_ips
    assert host_registry.released_hosts == [Host(server.ip_addr)]


@pytest.mark.asyncio
async def test_cluster_uninstall_waits_for_cancelled_starting_server_to_finish_adding() -> None:
    """Regression test for uninstall racing with a stopped add_server() still inside install()."""
    cluster, host_registry, created_servers = make_cluster()
    add_task, server = await start_adding_server(cluster, created_servers)
    server.block_stop_until_released = True

    uninstall_task = asyncio.create_task(cluster.uninstall())
    await server.first_stop_started.wait()
    server.finish_stop.set()
    await server.stop_finished.wait()

    for _ in range(10):
        if server.uninstall_started.is_set():
            break
        await asyncio.sleep(0)
    assert not server.uninstall_started.is_set()

    server.finish_install.set()
    with pytest.raises(RuntimeError, match="stopped while it was being added"):
        await add_task
    await uninstall_task

    assert server.uninstall_started.is_set()
    assert not server.uninstalled_before_install_finished
    assert not cluster.running
    assert not cluster.starting
    assert not cluster.leased_ips
    assert host_registry.released_hosts == [Host(server.ip_addr)]
