import logging
import pandas as pd
import numpy as np
from uuid import uuid4
from dask.delayed import delayed
import dask.dataframe as dd

logger = logging.getLogger("quasardb_dask")

try:
    import quasardb
    import quasardb.pandas as qdbpd
except ImportError as err:
    pass


def read_dataframe(
    query: str, meta: pd.DataFrame, conn_kwargs: dict, query_kwargs: dict
):
    """
    Creates connection and queries cluster with passed query, server side of Dask.
    """
    logger.debug('Querying QuasarDB with query: "%s"', query)
    with quasardb.Cluster(**conn_kwargs) as conn:
        df = qdbpd.query(conn, query, **query_kwargs)

    if len(df) == 0:
        return meta
    else:
        return df


def _create_table_from_meta(conn, table_name: str, meta: pd.DataFrame):
    logger.debug("Creating temporary table %s from meta", table_name)
    _dtype_to_column_type = {
        np.dtype("int64"): quasardb.ColumnType.Int64,
        np.dtype("int32"): quasardb.ColumnType.Int64,
        np.dtype("int16"): quasardb.ColumnType.Int64,
        np.dtype("float64"): quasardb.ColumnType.Double,
        np.dtype("float32"): quasardb.ColumnType.Double,
        np.dtype("float16"): quasardb.ColumnType.Double,
        np.dtype("unicode"): quasardb.ColumnType.String,
        np.dtype("bytes"): quasardb.ColumnType.Blob,
        np.dtype("datetime64[ns]"): quasardb.ColumnType.Timestamp,
        np.dtype("datetime64[ms]"): quasardb.ColumnType.Timestamp,
        np.dtype("datetime64[s]"): quasardb.ColumnType.Timestamp,
    }

    table_config = []

    for column_name, dtype in zip(meta.columns, meta.dtypes):
        if column_name.startswith("$"):
            continue  # skip internal columns
        column_type = _dtype_to_column_type[dtype]
        table_config.append(quasardb.ColumnInfo(column_type, column_name))

    table = conn.table(table_name)

    # TODO: we need some way to estimate shard size for this
    table.create(table_config)


@delayed
def writeback_query_to_cluster(
    query: str, meta: pd.DataFrame, conn_kwargs: dict, query_kwargs: dict
) -> str:
    """
    Creates connection and queries cluster with passed query.
    Writes the result to a temporary table and returns a query to read from that table.

    With complex interpolation queries spliting the query into multiple parts would result in loss of accuracy,
    so we write the result to a temporary table and return a simple `SELECT *` query, which can be easily split.
    """
    with quasardb.Cluster(**conn_kwargs) as conn:
        logger.debug('Querying QuasarDB with query: "%s"', query)
        df = qdbpd.query(conn, query, **query_kwargs)
        df.set_index("$timestamp", inplace=True)

        table_name = (
            f"tmptable"  # TODO: $qdb prefix is reserved, add support for / in queries
        )
        _create_table_from_meta(conn, table_name, meta)

        logger.debug("Writing dataframe to temporary table %s", table_name)
        qdbpd.write_dataframe(df, conn, table_name, fast=True)

    writeback_query = f'SELECT * FROM "{table_name}"'
    return writeback_query
