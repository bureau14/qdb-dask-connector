import math
import pytest
import quasardb
import quasardb.pandas as qdbpd
import qdb_dask_connector as qdbdsk
import logging
import conftest
from utils import *

logger = logging.getLogger("test-dask-data-integrity")

### Query tests, we care about results of dask query matching those of pandas
# when using default index, it has to be reset to match pandas DataFrame.
# we neeed to check that each query is split into multiple dask partitions
#
# index for a Dask DataFrame will not be monotonically increasing from 0.
# Instead, it will restart at 0 for each partition (e.g. index1 = [0, ..., 10], index2 = [0, ...]).
# This is due to the inability to statically know the full length of the index.
# https://docs.dask.org/en/stable/generated/dask.dataframe.DataFrame.reset_index.html


@pytest.mark.parametrize("frequency", ["h"], ids=["frequency=H"], indirect=True)
@pytest.mark.parametrize(
    "range_slice",
    [1, 0.5, 0.25],
    ids=["range_slice=1", "range_slice=0.5", "range_slice=0.25"],
)
@pytest.mark.parametrize(
    "query_options",
    [{"index": None}, {"index": "$timestamp"}],
    ids=["index=None", "index=$timestamp"],
)
def test_dask_df_select_star_equals_pandas_df(
    df_with_table, qdbd_connection, qdbd_settings, query_options, range_slice
):
    _, _, df, _ = df_with_table
    query_range = get_subrange(df, range_slice)
    _, _, query = prepare_query_test(df_with_table, qdbd_connection, "*", query_range)

    pandas_df = qdbpd.query(qdbd_connection, query, **query_options)
    dask_df = qdbdsk.query(
        query, qdbd_settings.get("uri").get("insecure"), **query_options
    )

    assert dask_df.npartitions > 1, "Dask DataFrame should have multiple partitions"
    dask_df = dask_df.compute()

    if query_options.get("index") is None:
        dask_df = dask_df.reset_index(drop=True)

    assert_df_equal(pandas_df, dask_df)


@pytest.mark.parametrize("frequency", ["h"], ids=["frequency=H"], indirect=True)
@pytest.mark.parametrize(
    "range_slice",
    [1, 0.5, 0.25],
    ids=["range_slice=1", "range_slice=0.5", "range_slice=0.25"],
)
@pytest.mark.parametrize(
    "use_alias", [False, True], ids=["use_alias=False", "use_alias=True"]
)
def test_dask_df_select_columns_equals_pandas_df(
    df_with_table, qdbd_connection, qdbd_settings, use_alias, range_slice
):
    _, _, df, _ = df_with_table

    columns = ", ".join(
        [f"{col} as {col}_alias" if use_alias else f"{col}" for col in df.columns]
    )

    query_range = get_subrange(df, range_slice)
    _, _, query = prepare_query_test(
        df_with_table, qdbd_connection, columns, query_range
    )
    pandas_df = qdbpd.query(qdbd_connection, query)
    dask_df = qdbdsk.query(query, qdbd_settings.get("uri").get("insecure"))

    assert dask_df.npartitions > 1, "Dask DataFrame should have multiple partitions"
    dask_df = dask_df.compute()

    dask_df = dask_df.reset_index(drop=True)

    assert_df_equal(pandas_df, dask_df)


@pytest.mark.parametrize("frequency", ["h"], ids=["frequency=H"], indirect=True)
@pytest.mark.parametrize("group_by", ["1h", "1d"])
def test_dask_df_select_agg_group_by_time_equals_pandas_df(
    df_with_table, qdbd_connection, qdbd_settings, group_by
):
    _, _, df, _ = df_with_table
    columns = ", ".join([f"count({col})" for col in df.columns])
    df, _, query = prepare_query_test(
        df_with_table,
        qdbd_connection,
        columns=f"$timestamp, {columns}",
        group_by=group_by,
    )

    pandas_df = qdbpd.query(qdbd_connection, query)
    dask_df = qdbdsk.query(query, qdbd_settings.get("uri").get("insecure"))

    assert dask_df.npartitions > 1, "Dask DataFrame should have multiple partitions"
    dask_df = dask_df.compute()

    dask_df = dask_df.reset_index(drop=True)

    assert_df_equal(pandas_df, dask_df)


@pytest.mark.parametrize(
    "query",
    [
        "INSERT INTO test ($timestamp, x) VALUES (now(), 2)",
        "DROP TABLE test",
        "DELETE FROM test",
        "CREATE TABLE test (x INT64)",
        "SHOW TABLE test",
        "ALTER TABLE test ADD COLUMN y INT64",
        "SHOW DISK USAGE ON test",
    ],
)
def test_dask_query_exception_on_non_select_query(qdbd_settings, query):
    """
    Tests that a non-select query raises an exception
    """
    with pytest.raises(NotImplementedError):
        qdbdsk.query(query, qdbd_settings.get("uri").get("insecure"))
