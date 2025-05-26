import math
import pytest
import logging
import conftest
from utils import *
import dask.dataframe as dd
import qdb_dask_connector as qdbdsk
from dask.distributed import LocalCluster, Client

logger = logging.getLogger("test-dask-integration")


@conftest.override_cdtypes([np.dtype("float64")])
@pytest.mark.parametrize("row_count", [224], ids=["row_count=224"], indirect=True)
@pytest.mark.parametrize("sparsify", [100], ids=["sparsify=none"], indirect=True)
def test_dask_query_meta_set(df_with_table, qdbd_connection, qdbd_settings):
    """
    tests that the columns are set correctly in the dask DataFrame
    """
    _, _, query = prepare_query_test(df_with_table, qdbd_connection)
    df = qdbpd.query(qdbd_connection, query)
    ddf = qdbdsk.query(query, qdbd_settings.get("uri").get("insecure"))

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

    ddf = qdbdsk.query(query, qdbd_settings.get("uri").get("insecure"))

    assert isinstance(ddf, dd.DataFrame)
    result = ddf.compute()
    assert isinstance(result, pd.DataFrame)


@conftest.override_cdtypes([np.dtype("float64")])
@pytest.mark.parametrize("row_count", [224], ids=["row_count=224"], indirect=True)
@pytest.mark.parametrize("sparsify", [100], ids=["sparsify=none"], indirect=True)
@pytest.mark.parametrize("frequency", ["h"], ids=["frequency=H"], indirect=True)
def test_dask_query_parallelized(df_with_table, qdbd_connection, qdbd_settings):
    _, _, df, table = df_with_table
    shard_size = table.get_shard_size()
    start, end = df.index[0], df.index[-1]

    _, _, query = prepare_query_test(df_with_table, qdbd_connection)

    ddf = qdbdsk.query(query, qdbd_settings.get("uri").get("insecure"))

    # value of npartitions determines number of delayed tasks
    # delayed tasks can be executed in parallel
    # currently tasks are created for each shard
    expected_number_of_partitions = math.ceil(
        (end - start).total_seconds() / shard_size.total_seconds()
    )
    assert ddf.npartitions == expected_number_of_partitions


@conftest.override_cdtypes([np.dtype("float64")])
@pytest.mark.parametrize("row_count", [224], ids=["row_count=224"], indirect=True)
@pytest.mark.parametrize("sparsify", [100], ids=["sparsify=none"], indirect=True)
def test_dask_compute_on_local_cluster(df_with_table, qdbd_connection, qdbd_settings):
    _, _, query = prepare_query_test(df_with_table, qdbd_connection)

    with LocalCluster(n_workers=2) as cluster:
        with Client(cluster):
            ddf = qdbdsk.query(query, qdbd_settings.get("uri").get("insecure"))
            res = ddf.compute()
            res.head()


# we will need some tests for dask the math operations
