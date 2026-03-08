# CQL Master Port

## Overview

The CQL master port is a unified listening port that auto-detects and handles
multiple client protocols on a single TCP endpoint. It eliminates the need for
clients to know the separate plain, TLS, shard-aware, and proxy-protocol ports
by multiplexing all of them onto one port.

When enabled, the server inspects the first byte(s) of each inbound connection
to determine the protocol layer in use, then routes the connection accordingly.

## Configuration

Enable the master port by setting `native_transport_master_port` in
`scylla.yaml`:

```yaml
# Example scylla.yaml configuration with master port enabled.
# The master port auto-detects plain CQL, TLS, proxy protocol v2,
# and shard selection on a single port.

native_transport_port: 9042
native_transport_port_ssl: 9142
native_shard_aware_transport_port: 19042
native_shard_aware_transport_port_ssl: 19142

# Unified master port -- set to a port not used by any of the above.
native_transport_master_port: 29042

client_encryption_options:
  enabled: true
  certificate: /etc/scylla/certs/scylla.crt
  keyfile: /etc/scylla/certs/scylla.key
```

A value of `0` (the default) disables the feature. When TLS is desired, the
standard `client_encryption_options` must also be configured; the master port
reuses those credentials.

**Important:** `native_transport_master_port` must be set to a value different
from all other CQL port settings: `native_transport_port`,
`native_transport_port_ssl`, `native_shard_aware_transport_port`,
`native_shard_aware_transport_port_ssl`, and all proxy-protocol port variants.
A collision with any of these will fail at startup with a clear config error.

## Protocol Detection

The server reads the first byte of each connection and uses it to identify the
protocol:

| First byte | Protocol                          |
|------------|-----------------------------------|
| `0x0D`     | Proxy Protocol v2 header follows  |
| `0xAA`     | Shard selection header follows    |
| `0x16`     | TLS ClientHello                   |
| `0x03`–`0x05` | Native CQL (protocol version)  |

Detection is layered: a proxy protocol v2 header may be followed by a shard
selection header, which may in turn indicate TLS.

### Known limitations of first-byte detection

- **`0x0D` is reserved for PP v2.** If a future protocol's first byte is
  `0x0D`, the master port will always attempt to parse it as a PP v2 header.
  This is an inherent trade-off of the single-byte dispatch approach.

- **Magic byte `0xAA` and CQL version space.** The shard selection magic
  `0xAA` occupies the same position as the CQL version byte. Current CQL
  versions are 3–5; version 170 (0xAA) is extremely unlikely but is an
  implicit assumption. If CQL ever approaches that version, the magic byte
  must be changed.

### Shard Selection Header (v1)

A 4-byte header that lets the client target a specific shard without relying on
the source port trick used by shard-aware drivers:

```
Byte 0:   Magic = 0xAA
Byte 1:   Flags (bit 0 = TLS follows; bits 1-7 reserved, must be 0)
Byte 2-3: Desired shard ID (uint16_t, big-endian)
```

The shard ID is taken modulo `smp::count`, so values larger than the shard
count wrap. After the header is consumed the stream contains either a TLS
ClientHello (if the TLS flag is set) or raw CQL frames.

**Versioning:** The magic byte `0xAA` identifies version 1 of the shard
selection header. There is no explicit version field in the header; future
versions must use a different magic byte or repurpose currently-reserved flag
bits to indicate extended header formats.

**Flag validation:** Connections that set any reserved flag bits (bits 1-7)
are rejected (fail-closed). This ensures clients detect misconfiguration
immediately and makes it safe to assign meaning to those bits in the future.

### Proxy Protocol v2

The master port recognizes the standard PROXY protocol v2 header (RFC). When
present, the real client address is extracted and the connection is attributed
to that address. After the PP v2 header is consumed, protocol detection
continues with the next byte (shard selection, TLS, or plain CQL).

The PP v2 address-data length is capped at 256 bytes (IPv6 with TLVs is ~36
bytes). This prevents unauthenticated clients from triggering large heap
allocations in the accept path.

### TLS

When a TLS ClientHello is detected (either via first-byte `0x16` or the TLS
flag in the shard selection header), the server wraps the connection with
`seastar::tls::wrap_server` using the configured credentials and recreates the
`_read_buf` / `_write_buf` streams from the new TLS-wrapped socket (via
`connection::rewrap_streams()`). If no TLS credentials are configured, the
connection is rejected with a warning.

## SUPPORTED Response

When the master port is enabled, the CQL `SUPPORTED` response includes the
key:

```
SCYLLA_MASTER_PORT: <port>
```

Clients can use this to discover the master port and switch to it for
subsequent connections.

## Architecture

### Seastar layer (`seastar/src/net/posix-stack.cc`)

`posix_server_socket_impl::accept_master_port()` — the core accept loop. It:

1. Accepts a raw TCP connection.
2. Reads the first byte to detect the protocol.
3. For PP v2: reads the full header and extracts the real client address.
4. For shard selection: reads the 4-byte header, extracts shard ID and TLS
   flag. Rejects connections with unknown flags set.
5. For legacy clients: saves the consumed byte as a prefix to be prepended to
   the data stream.
6. Routes the connection to the target shard via `smp::submit_to` or returns
   it directly if already on the correct shard. Failures during cross-shard
   routing are logged.

Invalid or incomplete headers cause the connection to be dropped (with
debug-level logging per connection and rate-limited warnings) and the loop
continues accepting the next connection.

### Generic server layer (`transport/generic_server.cc`)

`server::do_accepts()` handles both regular and master port listeners in a
single unified method via a `bool master_port` parameter. When
`master_port=true`:

- `is_tls` is derived from `accept_result::metadata` (set by Seastar's
  protocol detection) rather than being known at listen time.
- The server wraps the socket with TLS if TLS was detected and credentials
  are available.
- If TLS is detected but no credentials are configured, the connection is
  rejected.

**Design note:** The `bool master_port` parameter approach keeps the two code
paths (regular accept and master port accept) together, which is simpler than a
subclass/strategy pattern for the current level of complexity. If master port
adds significantly more logic, splitting into a strategy may be warranted.

### Controller layer (`transport/controller.cc`)

`controller::start_listening_on_tcp_sockets()` creates the master port
listener when `native_transport_master_port` is set to a non-zero value. It
validates the master port against all other configured CQL ports and passes TLS
credentials (if configured) so that TLS auto-detection can wrap connections.

## Timeouts

The protocol detection phase has a 5-second timeout
(`master_port_detect_timeout` in `seastar/src/net/posix-stack.cc`). If a client
connects but doesn't send any data within 5 seconds, the connection is dropped.
This prevents a single stalled client from blocking the accept loop.

**Note:** The timeout uses Seastar's `with_timeout`, which resolves the future
with `timed_out_error`. The underlying `read_exactly` I/O is cancelled via the
reactor's normal pollable_fd lifecycle — when the coroutine exits, the
`pollable_fd` is destroyed, which deregisters it from epoll.

## Error Handling

- **Truncated headers**: connection is dropped, debug log emitted.
- **Invalid PP v2 signature**: connection is dropped, debug log emitted.
- **Unsupported address family** (e.g., AF_UNIX in PP v2): connection dropped.
- **PP v2 address-data too large** (> 256 bytes): connection dropped.
- **Unknown shard selection flags**: connection rejected (fail-closed).
- **TLS without credentials**: connection rejected with a rate-limited warning.
- **TLS wrap failure**: connection dropped, debug log emitted.
- **Protocol detection timeout**: connection dropped after 5 seconds of
  inactivity during the detection phase.
- **Cross-shard routing failure**: warning logged (e.g., OOM on target shard).
- **Partial reads**: all reads loop until the expected byte count is received
  or EOF, preventing silent data corruption from `read_some` returning fewer
  bytes than requested.

## Performance Considerations

All connections initially arrive on shard 0's accept loop for protocol
detection, then get forwarded to target shards. With very high connection rates
this could become a bottleneck. The existing `connection_distribution` LBA for
regular ports distributes accept load via `SO_REUSEPORT`, but the master port
cannot use `SO_REUSEPORT` because protocol detection must happen before shard
routing.

**Why `SO_REUSEPORT` is incompatible:** With `SO_REUSEPORT`, the kernel
distributes incoming connections across multiple sockets (one per shard), so
each shard's accept loop receives connections directly. The master port requires
reading the first bytes of each connection to determine the protocol and target
shard *before* routing. If `SO_REUSEPORT` were used, a connection could land on
any shard, but the shard selection header might specify a different shard — the
connection would need to be re-routed anyway. Worse, the protocol detection
read must happen on the accepting shard, and the connection must be forwarded
atomically with the consumed prefix bytes. A single listener on shard 0 avoids
this complexity at the cost of shard 0 being the bottleneck for new connections.

If this becomes a bottleneck in practice, a future optimization could perform
protocol detection on the target shard instead.

## Load Balancer Configuration

When placing an L4 load balancer in front of ScyllaDB nodes with the master
port enabled:

- **PP v2 header (recommended):** Configure the load balancer to prepend a
  PROXY protocol v2 header with the real client address. The master port will
  extract the client address and attribute the connection correctly. This is
  the standard approach for HAProxy, nginx stream, and AWS NLB.

- **Shard selection:** The load balancer itself typically does *not* send shard
  selection headers. Shard selection is a client-side concern — the ScyllaDB
  driver discovers shard count via the `SUPPORTED` response and sends the
  shard selection header on subsequent connections. The load balancer should
  pass through these bytes transparently.

- **TLS passthrough vs. termination:** If the load balancer terminates TLS,
  downstream connections to the master port will appear as plain CQL. If TLS
  passthrough is used, the master port detects the TLS ClientHello and performs
  the handshake. In passthrough mode, PP v2 headers must be sent *before* the
  TLS data (i.e., as a TCP prefix, not inside the TLS stream).

- **Health checks:** Use a TCP connect check to the master port. Sending a CQL
  `OPTIONS` frame and expecting a `SUPPORTED` response is also valid for L7
  health checks.

## Versioning and Version Discovery

The magic byte `0xAA` identifies version 1 of the shard selection header. There
is no in-band version negotiation for the header format itself. Clients discover
master port support and the port number via the `SCYLLA_MASTER_PORT` key in the
CQL `SUPPORTED` response.

**Future versions:** If a v2 header format is needed, there are two options:

1. **New magic byte:** Use a different first byte (e.g., `0xAB`) to signal v2.
   The server can dispatch on the first byte as it does today. Clients that see
   `SCYLLA_MASTER_PORT` in `SUPPORTED` know the server supports at least v1.

2. **Reserved flag bits:** The v1 header reserves bits 1-7 of the flags byte.
   A future version could set one of these bits to signal extended header
   fields that follow the initial 4 bytes.

In either case, the server should advertise the supported header version in the
`SUPPORTED` response (e.g., `SCYLLA_MASTER_PORT_VERSION: 1,2`) so clients can
select the appropriate format. Until then, clients should assume v1.

## Testing

### Boost unit tests (`test/boost/master_port_test.cc`)

These test the Seastar-level socket accept behavior: header format validation,
protocol detection, and the accept loop's handling of malformed input. They
require `-c1` (single shard) because they rely on deterministic accept ordering.

**Note:** These tests do not exercise full CQL server integration — they verify
that the Seastar layer correctly parses headers and returns the right
`accept_result` metadata.

### Python integration tests (`test/cqlpy/test_master_port.py`)

These test full CQL protocol interaction over the master port. They require a
running ScyllaDB cluster with the master port enabled.

**Running the tests:**
```bash
# Set up a ScyllaDB instance with master port enabled, then:
export MASTER_PORT=29042
pytest test/cqlpy/test_master_port.py --host=127.0.0.1
```

The `MASTER_PORT` environment variable must be set explicitly. If not set, tests
are skipped to avoid false positives from accidentally testing against the
regular CQL port.
