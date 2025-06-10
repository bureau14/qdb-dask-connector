import logging
import pandas as pd
import numpy as np

import pendulum  # easy date/time interaction
from ksuid import Ksuid  # k-sortable identifiers for temporary tables

logger = logging.getLogger("quasardb_dask")

try:
    import quasardb
    import quasardb.pandas as qdbpd
except ImportError as err:
    pass


def _restore_empty_float_columns(df: pd.DataFrame, meta: pd.DataFrame) -> None:
    """
    Fix dtype mismatches caused by pandas’ default behaviour when a column has no
    data for the current split:

    • If a Series is promoted to float64 *only* because it contains only NaN
      values, cast it back to the dtype found in *meta* so that downstream
      dtypes stay consistent.

    The function mutates *df* in-place and silently ignores columns that do not
    satisfy the three bullet-point criteria above.
    """
    for col in df.columns.intersection(meta.columns):
        if (
            df[col].dtype == np.float64
            and meta[col].dtype != np.float64
            and df[col].isna().all()
        ):
            # rebuild the Series with the correct dtype while keeping index length
            df[col] = pd.Series(
                [pd.NA] * len(df), index=df.index, dtype=meta[col].dtype
            )


def split_query(conn, query: str) -> list[str]:
    """
    Attempts to decompose *query* into a set of non-overlapping smaller queries
    that can be executed independently.

    If this fails (e.g. the query uses window functions, contains an ASOF JOIN,
    or any other construct we can’t analyse yet), we return the original
    query as a single‐element list; the caller will then fall back to
    `materialize_to_temp`, which handles non-splittable workloads.

    Returns
    -------
    list[str]
        Either the list of split queries or ``[query]`` when splitting is
        impossible.
    """
    try:
        # Temporary Python implementation until SC-16768/add-function-that-splits-query-to-the-c-api
        # provides native query-splitting in the QuasarDB C API.
        #
        # 1. Identify table and its shard size
        table_name = _extract_table_name_from_query(query)
        shard_size = conn.table(table_name.replace('"', "")).get_shard_size()

        # 2. Work out the time range we need to cover
        start, end = _extract_range_from_query(conn, query, table_name)

        # 3. Ask QuasarDB to build evenly-sized sub-ranges
        ranges_to_query = conn.split_query_range(start, end, shard_size)

        # 4. Generate one sub-query per range
        return [_create_subrange_query(query, rng) for rng in ranges_to_query]

    except Exception as err:  # noqa: BLE001 – broad on purpose; see note below
        # Most likely the query cannot be split with the limited heuristics
        # we have today. Fallback: execute the full query once via
        # `materialize_to_temp`, which will write the result to a temporary
        # table and split from there.
        logger.debug(
            "Query splitting failed, falling back to materialize_to_temp: %s", err
        )
        return [query]


def read_dataframe(
    query: str, meta: pd.DataFrame, conn_kwargs: dict, query_kwargs: dict
) -> pd.DataFrame:
    """
    Creates connection and queries cluster with passed query, server side of Dask.
    """
    with quasardb.Cluster(**conn_kwargs) as conn:
        df = qdbpd.query(conn, query, **query_kwargs)

    _restore_empty_float_columns(df, meta)

    if len(df) == 0:
        return meta
    else:
        return df


def _create_table_from_meta(
    conn,
    *,
    table_name: str,
    meta: pd.DataFrame,
    shard_size: pendulum.Duration = pendulum.duration(days=1),
    ttl: pendulum.Duration = pendulum.duration(days=7),
):
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

    table.create(table_config, shard_size=shard_size, ttl=ttl)


def materialize_to_temp(
    query: str, meta: pd.DataFrame, conn_kwargs: dict, query_kwargs: dict
) -> str:
    """
    Takes query, executes it, writes it into a temporary table, query that selects all data from temporary table.

    This is a placeholder for when quasardb implements actual "INSERT INTO <table> SELECT ..." logic. See shortcut tickets:
     * sc-7166/insert-into-table-select
     * sc-16833/create-table-as-select
    """

    # a lot of rocksdb compaction pressure.

    with quasardb.Cluster(**conn_kwargs) as conn:
        df = qdbpd.query(conn, query, **query_kwargs)

        print("executing query: {}".format(query))

        logger.debug("got dataframe: %s", df.head)

        # We use Ksuid() so that the tables we create are sorted by time. This is highly effective for rocksdb,
        # as this pretty much guarantees that newly created tables don't overlap with old tables, and as such reduce
        table_name = "qdb/dask/temp/{}".format(Ksuid())

        # Create the table *after* we fetched the data, because now we at least know that the query succeeded
        # and don't garbage leave empty tables around.
        #
        # We use a TTL of 1 day so that data is cleared up quickly
        _create_table_from_meta(
            conn, table_name=table_name, meta=meta, ttl=pendulum.duration(days=1)
        )

        qdbpd.write_dataframe(
            df, conn, table_name, push_mode=quasardb.WriterPushMode.Fast
        )

        query_ = 'SELECT * FROM "{}"'.format(table_name)

        return split_query(conn, query_)


def get_meta(query: str, conn_kwargs: dict, query_kwargs: dict) -> pd.DataFrame:
    """
    Returns empty dataframe with the expected schema of the query result.
    """

    ## TODO: fix
    #
    # Waiting for real function to be implemented

    # np_res = conn.validate_query(query)
    # col_dtypes = {}
    # for id, column in enumerate(np_res):
    #     col_dtypes[column[0]] = pd.Series(dtype=column[1].dtype)

    # df = pd.DataFrame(col_dtypes)
    # if query_kwargs["index"]:
    #     df.set_index(query_kwargs["index"], inplace=True)

    ##
    # Hard-coded to execute the *actual* query right now, drop all the data
    # and just return the schema.
    #
    # This is an uber-hack, should be removed
    with quasardb.Cluster(**conn_kwargs) as conn:
        df = qdbpd.query(conn, query, **query_kwargs)

        # Return dataframe with all rows removed
        return df[0:0]

    # return pd.DataFrame(
    #     {
    #         "$timestamp": pd.Series(dtype="datetime64[ns]"),
    #         "$table": pd.Series(dtype="unicode"),
    #         "facility_code": pd.Series(dtype="unicode"),
    #         "pointid": pd.Series(dtype="int64"),
    #         "tagname": pd.Series(dtype="unicode"),
    #         "unique_tagname": pd.Series(dtype="unicode"),
    #         "numericvalue": pd.Series(dtype="float64"),
    #         "stringvalue": pd.Series(dtype="unicode"),
    #         "batch_id": pd.Series(dtype="unicode"),
    #     }
    # )
