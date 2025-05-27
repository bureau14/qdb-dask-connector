import math
import pytest
import logging
import conftest
from utils import *
import dask.dataframe as dd
import quasardb_dask as qdbdsk
from dask.distributed import LocalCluster, Client

logger = logging.getLogger("test-dask-integration")


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
    with pytest.raises(NotImplementedError):
        qdbdsk.query(query, cluster_uri=qdbd_settings.get("uri").get("insecure"))


@pytest.mark.parametrize("row_count", [224], ids=["row_count=224"], indirect=True)
def test_dask_query_meta_set(df_with_table, qdbd_connection, qdbd_settings):
    """
    tests that the meta is set correctly in the dask DataFrame
    """
    _, _, query = prepare_query_test(df_with_table, qdbd_connection)
    df = qdbpd.query(qdbd_connection, query)
    ddf = qdbdsk.query(query, cluster_uri=qdbd_settings.get("uri").get("insecure"))

    dask_meta = ddf._meta_nonempty
    pandas_cols = df.columns

    assert dask_meta.columns.names == pandas_cols.names, "column names do not match"

    for col_name in pandas_cols:
        # treat string[pyarrow] as object
        if dask_meta[col_name].dtype == "string[pyarrow]":
            dask_meta[col_name] = dask_meta[col_name].astype("object")

        assert (
            dask_meta[col_name].dtype == df[col_name].dtype
        ), f"dtype of column {col_name} does not match"


@conftest.override_cdtypes([np.dtype("float64")])
@pytest.mark.parametrize("row_count", [224], ids=["row_count=224"], indirect=True)
@pytest.mark.parametrize("sparsify", [100], ids=["sparsify=none"], indirect=True)
def test_dask_query_lazy_evaluation(df_with_table, qdbd_connection, qdbd_settings):
    """
    tests that the function is lazy and does not return a Dataframe immediately.
    """

    _, _, query = prepare_query_test(df_with_table, qdbd_connection)

    ddf = qdbdsk.query(query, cluster_uri=qdbd_settings.get("uri").get("insecure"))

    assert isinstance(ddf, dd.DataFrame)
    assert ddf.__dask_graph__() is not None, "Dask DataFrame should have a dask graph"
    result = ddf.compute()
    assert isinstance(result, pd.DataFrame)


@conftest.override_cdtypes([np.dtype("float64")])
@pytest.mark.parametrize("row_count", [224], ids=["row_count=224"], indirect=True)
@pytest.mark.parametrize("sparsify", [100], ids=["sparsify=none"], indirect=True)
def test_dask_query_parallelized(df_with_table, qdbd_connection, qdbd_settings):
    """
    tests that query is split into multiple partitions
    """
    _, _, df, table = df_with_table
    shard_size = table.get_shard_size()
    start, end = df.index[0], df.index[-1]

    _, _, query = prepare_query_test(df_with_table, qdbd_connection)

    ddf = qdbdsk.query(query, cluster_uri=qdbd_settings.get("uri").get("insecure"))

    # value of npartitions determines number of delayed tasks
    # delayed tasks can be executed in parallel on dask cluster

    # currently tasks are created for each shard
    expected_number_of_partitions = math.ceil(
        (end - start).total_seconds() / shard_size.total_seconds()
    )

    assert ddf.npartitions == expected_number_of_partitions
    assert ddf.npartitions > 1, "Dask DataFrame should have multiple partitions"


@conftest.override_cdtypes([np.dtype("float64")])
@pytest.mark.parametrize("row_count", [224], ids=["row_count=224"], indirect=True)
@pytest.mark.parametrize("sparsify", [100], ids=["sparsify=none"], indirect=True)
def test_dask_compute_on_local_cluster(df_with_table, qdbd_connection, qdbd_settings):
    """
    tests that dask integration can be used with dask cluster
    """
    _, _, query = prepare_query_test(df_with_table, qdbd_connection)

    with LocalCluster(n_workers=2) as cluster:
        with Client(cluster):
            ddf = qdbdsk.query(
                query, cluster_uri=qdbd_settings.get("uri").get("insecure")
            )
            res = ddf.compute()
            res.head()


@conftest.override_cdtypes([np.dtype("float64"), np.dtype("int64")])
def test_dask_can_do_math_on_dask_dataframe(
    df_with_table, qdbd_connection, qdbd_settings
):
    """
    tests that dask.dataframe operations are supported for dask integration
    """
    df, _, query = prepare_query_test(df_with_table, qdbd_connection)
    ddf = qdbdsk.query(query, cluster_uri=qdbd_settings.get("uri").get("insecure"))

    assert isinstance(ddf, dd.DataFrame)

    number_column = df.columns[-1]

    ddf[f"sum_{number_column}"] = ddf[number_column].sum()
    ddf[f"mean_{number_column}"] = ddf[number_column].mean()
    ddf[f"std_{number_column}"] = ddf[number_column].std()

    result = ddf.compute()

    assert f"sum_{number_column}" in result.columns
    assert f"mean_{number_column}" in result.columns
    assert f"std_{number_column}" in result.columns
