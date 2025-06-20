import quasardb_dask as qdbdask
import logging
from utils import assert_df_equal
import quasardb.pandas as qdbpd
from dask.delayed import Delayed
import dask.dataframe as dd
import pandas as pd

logger = logging.getLogger("test-write-dataframe")


def test_write_dataframe_is_lazy(df_with_table, qdbd_settings):
    """
    Checks that the write_dataframe function returns a dask delayed object
    """
    _, _, df, table = df_with_table
    table_name = table.get_name()

    write_task = qdbdask.write_dataframe(
        df,
        cluster_uri=qdbd_settings.get("uri").get("insecure"),
        table=table_name,
        create=False,
    )

    assert write_task is not True, "write_dataframe shouldn't return True immediately"
    assert isinstance(
        write_task, Delayed
    ), "write_dataframe should return a dask delayed object"


def test_write_pandas_dataframe(df_with_table, qdbd_connection, qdbd_settings):
    """
    Tests that a pandas dataframe can be written to a QuasarDB table
    """
    _, _, df, table = df_with_table
    table_name = table.get_name()

    write_task = qdbdask.write_dataframe(
        df, cluster_uri=qdbd_settings.get("uri").get("insecure"), table=table_name
    ).compute()
    assert write_task is True, "write_dataframe should return True on success"

    # Verify that the written dataframe matches the original
    written_df = qdbpd.read_dataframe(qdbd_connection, table)

    assert_df_equal(df, written_df)


def test_write_dask_dataframe(df_with_table_inserted, qdbd_connection, qdbd_settings):
    """
    Tests that a dask delayed dataframe can be written to a QuasarDB table
    """
    df, table = df_with_table_inserted

    # Query back using dask connector
    original_table_name = table.get_name()
    ddf = qdbdask.query(
        f'SELECT * FROM "{original_table_name}"',
        cluster_uri=qdbd_settings.get("uri").get("insecure"),
    )

    assert isinstance(ddf, dd.DataFrame)

    # Write delayed dataframe to a new table
    new_table_name = f"{original_table_name}_COPY"
    write_task = qdbdask.write_dataframe(
        ddf,
        table=new_table_name,
        cluster_uri=qdbd_settings.get("uri").get("insecure"),
    ).compute()
    assert write_task is True, "write_dataframe should return True on success"

    # Read back from new table and compare with source
    written_df = qdbpd.read_dataframe(qdbd_connection, new_table_name)

    assert_df_equal(df, written_df)


def test_write_dataframe_without_set_index(qdbd_connection, qdbd_settings, table_name):
    """
    Tests that if no index is set on the dataframe, but $timestamp is present,
    it is still written correctly to the QuasarDB table.
    """
    ts = pd.date_range("2023-01-01", periods=10, freq="min")
    df = pd.DataFrame({"$timestamp": ts, "x": range(10)})

    df_with_index = df.set_index("$timestamp")

    # Write dataframe without set index
    write_task = qdbdask.write_dataframe(
        df,
        cluster_uri=qdbd_settings.get("uri").get("insecure"),
        table=table_name,
    ).compute()
    assert write_task is True, "write_dataframe should return True on success"

    # Verify that the written dataframe matches the original
    written_df = qdbpd.read_dataframe(qdbd_connection, table_name)
    assert_df_equal(df_with_index, written_df)
