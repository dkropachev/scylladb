# Copyright 2026-present ScyllaDB
#
# SPDX-License-Identifier: LicenseRef-ScyllaDB-Source-Available-1.1

from cassandra.protocol import InvalidRequest
import pytest

from .util import unique_name


@pytest.fixture(params=["score_ascending", "score_descending"])
def score_table(cql, test_keyspace, request):
    table = f"{test_keyspace}.{unique_name()}"
    mode = request.param
    cql.execute(f"CREATE TABLE {table} (k int PRIMARY KEY, x int, y int) "
                f"WITH reconciliation_mode = '{mode}'")
    yield table, mode
    cql.execute(f"DROP TABLE {table}")


def test_score_orders_independent_cells(cql, score_table):
    table, mode = score_table
    best = 1 if mode == "score_ascending" else 3
    cql.execute(f"INSERT INTO {table} (k, x, y) VALUES (1, 10, 20) USING SCORE 2")
    cql.execute(f"UPDATE {table} USING SCORE {best} SET x = 11 WHERE k = 1")
    cql.execute(f"UPDATE {table} USING SCORE {3 if best == 1 else 1} SET x = 12 WHERE k = 1")
    assert cql.execute(f"SELECT x, y FROM {table} WHERE k = 1").one() == (11, 20)


def test_score_required_and_timestamp_rejected(cql, score_table):
    table, _ = score_table
    with pytest.raises(InvalidRequest, match="USING SCORE"):
        cql.execute(f"INSERT INTO {table} (k, x) VALUES (1, 10)")
    with pytest.raises(InvalidRequest, match="USING SCORE"):
        cql.execute(f"INSERT INTO {table} (k, x) VALUES (1, 10) USING TIMESTAMP 1")
    with pytest.raises(InvalidRequest, match="conditional|Conditional"):
        cql.execute(f"INSERT INTO {table} (k, x) VALUES (1, 10) IF NOT EXISTS USING SCORE 1")


def test_bound_score(cql, score_table):
    table, mode = score_table
    insert = cql.prepare(f"INSERT INTO {table} (k, x) VALUES (?, ?) USING SCORE ?")
    cql.execute(insert, (1, 10, 2))
    cql.execute(insert, (1, 11, 1 if mode == "score_ascending" else 3))
    assert cql.execute(f"SELECT x FROM {table} WHERE k = 1").one()[0] == 11


def test_reserved_score_rejected(cql, score_table):
    table, mode = score_table
    reserved = 2**63 - 1 if mode == "score_ascending" else -(2**63)
    with pytest.raises(InvalidRequest, match="reserved missing timestamp"):
        cql.execute(f"INSERT INTO {table} (k, x) VALUES (1, 10) USING SCORE {reserved}")


def test_score_delete(cql, score_table):
    table, mode = score_table
    winning = 1 if mode == "score_ascending" else 3
    losing = 3 if mode == "score_ascending" else 1
    cql.execute(f"INSERT INTO {table} (k, x) VALUES (1, 10) USING SCORE 2")
    cql.execute(f"DELETE x FROM {table} USING SCORE {losing} WHERE k = 1")
    assert cql.execute(f"SELECT x FROM {table} WHERE k = 1").one()[0] == 10
    cql.execute(f"DELETE x FROM {table} USING SCORE {winning} WHERE k = 1")
    assert cql.execute(f"SELECT x FROM {table} WHERE k = 1").one()[0] is None


def test_mode_is_immutable(cql, score_table):
    table, _ = score_table
    with pytest.raises(InvalidRequest, match="cannot be changed"):
        cql.execute(f"ALTER TABLE {table} WITH reconciliation_mode = 'timestamp'")
