import math
import pytest
import quasardb
import quasardb.pandas as qdbpd
import quasardb_dask as qdbdsk
import logging
import conftest
from utils import *

logger = logging.getLogger("test-dask-data-equals-pandas")

### Query tests, we care about results of dask query matching those of pandas


@pytest.mark.parametrize(
    "range_slice",
    [1, 0.5, 0.25],
    ids=["range_slice=1", "range_slice=0.5", "range_slice=0.25"],
)
def test_dask_df_select_star_equals_pandas_df(
    df_with_table, qdbd_connection, qdbd_settings, range_slice
):
    _, _, df, _ = df_with_table
    query_range = get_subrange(df, range_slice)
    _, _, query = prepare_query_test(df_with_table, qdbd_connection, "*", query_range)

    pandas_df = qdbpd.query(qdbd_connection, query, index="$timestamp")
    dask_df = qdbdsk.query(query, cluster_uri=qdbd_settings.get("uri").get("insecure"))

    dask_df = dask_df.compute()

    assert_df_equal(pandas_df, dask_df)


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

    columns = "$timestamp, " + ", ".join(
        [
            f"{col} as {col}_alias" if use_alias else f"{col}"
            for col in df.columns
            if col != "$timestamp"
        ]
    )

    query_range = get_subrange(df, range_slice)
    _, _, query = prepare_query_test(
        df_with_table, qdbd_connection, columns, query_range
    )
    pandas_df = qdbpd.query(qdbd_connection, query, index="$timestamp")
    dask_df = qdbdsk.query(query, cluster_uri=qdbd_settings.get("uri").get("insecure"))

    dask_df = dask_df.compute()

    assert_df_equal(pandas_df, dask_df)


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

    pandas_df = qdbpd.query(qdbd_connection, query, index="$timestamp")
    dask_df = qdbdsk.query(query, cluster_uri=qdbd_settings.get("uri").get("insecure"))

    dask_df = dask_df.compute()

    assert_df_equal(pandas_df, dask_df)
