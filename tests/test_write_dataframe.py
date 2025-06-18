import pytest
import quasardb_dask as qdbdask
import logging
import dask
from utils import assert_df_equal
import quasardb.pandas as qdbpd
from dask.delayed import Delayed


logger = logging.getLogger("test-write-dataframe")


def test_write_dataframe_is_lazy(df_with_table, qdbd_settings):
    _, _, df, table = df_with_table
    table_name = table.get_name()

    write_task = qdbdask.write_dataframe(
        df, cluster_uri=qdbd_settings.get("uri").get("insecure"), table_name=table_name
    )

    assert (
        type(write_task) == Delayed
    ), "write_dataframe should return a dask delayed object"


def test_write_dataframe_from_pandas_df(df_with_table, qdbd_connection, qdbd_settings):
    """
    Verify that the written dataframe matches the original pandas dataframe.
    """
    _, _, df, table = df_with_table
    table_name = table.get_name()

    qdbdask.write_dataframe(
        df, cluster_uri=qdbd_settings.get("uri").get("insecure"), table_name=table_name
    ).compute()

    # Verify that the written dataframe matches the original
    written_df = qdbpd.read_dataframe(qdbd_connection, table)

    assert_df_equal(df, written_df)


def test_write_dataframe_from_dask_df(df_with_table, qdbd_connection, qdbd_settings):
    _, _, df, table = df_with_table
    pass
