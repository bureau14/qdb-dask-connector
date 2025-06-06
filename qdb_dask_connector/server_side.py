import logging
import pandas as pd
import numpy as np
from dask.delayed import delayed

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

    # TODO: we might need some way to estimate shard size for this
    table.create(table_config)


def read_write_back_query_to_cluster(
    query: str,
    table_name: str,
    meta: pd.DataFrame,
    conn_kwargs: dict,
    query_kwargs: dict,
):
    """
    Creates connection and queries cluster with passed query.
    Writes the result to a temporary table.
    """
    with quasardb.Cluster(**conn_kwargs) as conn:
        logger.debug('Querying QuasarDB with query: "%s"', query)
        df = qdbpd.query(conn, query, **query_kwargs)
        # we need some index to set $timestamp for new table
        # if the query has an index use it, otherwise default to $timestamp
        # this might create some issues, needs to be tested
        if query_kwargs["index"]:
            df.set_index(query_kwargs["index"], inplace=True)
        else:
            df.set_index("$timestamp", inplace=True)

        _create_table_from_meta(conn, table_name, meta)

        logger.debug("Writing dataframe to temporary table %s", table_name)
        qdbpd.write_dataframe(df, conn, table_name, fast=True)


def prepare_persist_query(
    query: str,
    new_table_name: str,
    meta: pd.DataFrame,
    conn_kwargs: dict,
    query_kwargs: dict,
) -> str:
    read_write_back_query_to_cluster(query, new_table_name, meta, conn_kwargs, query_kwargs)
    return f'SELECT * FROM "{new_table_name}"'


def cleanup_persisted_table(
    table_name: str, conn_kwargs: dict
):
    """
    Cleans up the temporary table created for persisting the query.
    """
    with quasardb.Cluster(**conn_kwargs) as conn:
        logger.debug("Dropping temporary table %s", table_name)
        conn.table(table_name).remove()