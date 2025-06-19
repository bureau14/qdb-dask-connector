import quasardb_dask as qdbdask
import logging
from utils import assert_df_equal
import quasardb.pandas as qdbpd
from dask.delayed import Delayed

logger = logging.getLogger("test-write-dataframe")


def test_write_dataframe_is_lazy(df_with_table, qdbd_connection, qdbd_settings):
    _, _, df, table = df_with_table
    table_name = table.get_name()

    write_task = qdbdask.write_dataframe(
        df,
        cluster_uri=qdbd_settings.get("uri").get("insecure"),
        table=table_name,
        create=False,
    )

    assert (
        type(write_task) == Delayed
    ), "write_dataframe should return a dask delayed object"


def test_write_pandas_dataframe(df_with_table, qdbd_connection, qdbd_settings):
    _, _, df, table = df_with_table
    table_name = table.get_name()

    qdbdask.write_dataframe(
        df, cluster_uri=qdbd_settings.get("uri").get("insecure"), table=table_name
    ).compute()

    # Verify that the written dataframe matches the original
    written_df = qdbpd.read_dataframe(qdbd_connection, table)

    assert_df_equal(df, written_df)


def test_write_dask_dataframe(df_with_table_inserted, qdbd_connection, qdbd_settings):
    df, table = df_with_table_inserted

    # Query back using dask connector
    original_table_name = table.get_name()
    ddf = qdbdask.query(
        f'SELECT * FROM "{original_table_name}"',
        cluster_uri=qdbd_settings.get("uri").get("insecure"),
    )

    # Write delayed dataframe to a new table
    new_table_name = f"{original_table_name}_COPY"
    qdbdask.write_dataframe(
        ddf,
        table=new_table_name,
        cluster_uri=qdbd_settings.get("uri").get("insecure"),
    ).compute()

    # Read back from new table and compare with source
    written_df = qdbpd.read_dataframe(qdbd_connection, new_table_name)

    assert_df_equal(df, written_df)
