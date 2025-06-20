import logging
import pandas as pd
import numpy as np
import pendulum  # easy date/time interaction
from ksuid import Ksuid  # k-sortable identifiers for temporary tables
from typing import List, Tuple, Union

logger = logging.getLogger("quasardb_dask")

try:
    import quasardb
    import quasardb.pandas as qdbpd
except ImportError as err:
    pass


def _empty_like(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return an *empty* DataFrame that preserves
    • column order,
    • dtypes,
    • index name and dtype.

    Works even when *df* itself is all-NA.
    """
    # 1. one empty Series per column, dtype taken verbatim
    cols = {c: pd.Series(dtype=df.dtypes[c]) for c in df.columns}

    # 2. Create a np.datetime64 index spec -- for now we only support
    #    dataframes with a $timestamp np.datetime64[ns] index.
    idx = pd.Index([], name="$timestamp", dtype=np.dtype("datetime64[ns]"))

    # 3. Create a new, empty dataframe
    return pd.DataFrame(cols, index=idx)


def _restore_empty_float_columns(df: pd.DataFrame, meta: pd.DataFrame) -> None:
    """
    Down-cast columns that Pandas temporarily promoted to ``float64`` solely
    because the *current partition* consists of all-NaN values.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame to fix (mutated **in-place**).
    meta : pandas.DataFrame
        Authoritative schema whose dtypes must be preserved.

    Decision rationale:
    • Guarantees dtype stability across partitions so that later concatenation
      or group-by steps do not raise “dtype mismatch” errors.

    Key assumptions:
    • A column is safe to cast back when **all** its values are ``NaN``.
    • *meta* contains the desired dtype for every column present in *df*.

    Performance trade-offs:
    • Operates column-wise with vectorised checks; overhead is negligible
      compared with network I/O.
    """
    for col in df.columns.intersection(meta.columns):
        if (
            df[col].dtype == np.float64
            and meta[col].dtype not in (np.float64, np.float32, np.float16)
            and df[col].isna().all()
        ):
            df[col] = pd.Series(
                [pd.NA] * len(df), index=df.index, dtype=meta[col].dtype
            )


def _coerce_timestamp_index(df: pd.DataFrame, name: str = "$timestamp") -> None:
    """
    Normalise *df* so its index is a ``DatetimeIndex[ns]`` named ``$timestamp``
    and expose the same data as a column of that name when absent.

    Parameters
    ----------
    df : pandas.DataFrame
        Frame to normalise (mutated **in-place**).
    name : str, default ``"$timestamp"``
        Desired index / column label.

    Decision rationale:
    • QuasarDB treats ``$timestamp`` as the canonical time axis; normalising
      early prevents subtle alignment bugs during Dask shuffles and merges.

    Key assumptions:
    • Either the current index or the column *name* can be coerced to
      ``datetime64[ns]``.

    Performance trade-offs:
    • Uses cheap dtype checks and vectorised ``pd.to_datetime``; the cost is
      dominated by upstream network latency.
    """
    # 1. Normalise to DatetimeIndex[ns] -----------------------------
    if df.index.dtype != np.dtype("datetime64[ns]"):
        if name in df.columns:
            # convert column if needed, then promote to index
            if df[name].dtype != np.dtype("datetime64[ns]"):
                df[name] = pd.to_datetime(df[name]).astype("datetime64[ns]")
            df.set_index(name, inplace=True)
        else:
            # last-resort: try to coerce the existing index
            df.index = pd.to_datetime(df.index).astype("datetime64[ns]")

    # 2. Enforce index name ----------------------------------------
    df.index.name = name


def create_partition_tasks(
    query: str,
    meta: pd.DataFrame,
    conn_kwargs: dict,
    query_kwargs: dict,
    npartitions: int,
) -> list[tuple[str, str] | tuple[str, dict]]:
    """
    Decompose *query* into smaller, independent tasks that can be executed as
    separate Dask partitions.

    Parameters
    ----------
    query : str
        User-supplied SELECT statement.
    meta : pandas.DataFrame
        Empty frame describing the expected schema.
    conn_kwargs, query_kwargs : dict
        Forwarded to the QuasarDB Python API.
    npartitions : int
        Desired number of partitions (≥ 1).

    Returns
    -------
    list[tuple[str, str] | tuple[str, dict]]
        Each element is either
        • ("query",   <sql>)   – run `<sql>` directly, or
        • ("reader",  <dict>) – call ``qdbpd.read_dataframe`` with *dict*.

    Decision rationale:
    • Enables parallel execution by turning one large query into many smaller,
      non-overlapping jobs.
    • Falls back to materialising the result into a temporary table when the
      query cannot be split safely (ASOF JOIN, window functions, …).

    Performance trade-offs:
    • Splitting avoids repeated large transfers; materialisation trades a single
      write & short-lived storage for simplified downstream reads.
    """

    if npartitions == 1:
        # Don't even bother creating multiple partition tasks, just return the query as-is.
        logger.info("single partition input query, avoiding partitioning")
        return [("query", query)]

    ##
    # TODO: use QuasarDB Python API to parse the query and create multiple
    #       partition tasks. Needs implementation in QuasarDB C API.
    #
    #       See shortcut ticket: sc-16768/add-function-that-splits-query-to-the-c-api

    tasks = [("query", query)]

    # If only 1 partition task was made by the C API, it means the query is not splittable (e.g.
    # an ASOF JOIN, window function, etc).
    #
    # In this case, we "materialize" the query's result into a temporary table.

    if len(tasks) == 1:
        task_type, query = tasks[0]
        assert task_type == "query"

        tasks = materialize_to_temp(
            query, meta, conn_kwargs, query_kwargs, npartitions=npartitions
        )

    logger.debug("created %d partition tasks from input query", len(tasks))

    # TODO: implement
    return tasks


def get_meta(query: str, conn_kwargs: dict, query_kwargs: dict) -> pd.DataFrame:
    """
    Returns empty dataframe with the expected schema of the query result.
    """

    ## TODO: fix
    #
    # Waiting for real function to be implemented

    ##
    # Hard-coded to execute the *actual* query right now, drop all the data
    # and just return the schema.
    #
    # This is an uber-hack, should be removed
    with quasardb.Cluster(**conn_kwargs) as conn:
        df = qdbpd.query(conn, query, **query_kwargs)
        return _empty_like(df)


def run_partition_task(
    task: tuple[str, str] | tuple[str, dict],
    meta: pd.DataFrame,
    conn_kwargs: dict,
    query_kwargs: dict,
) -> pd.DataFrame:
    """
    Executes a single Dask partition task on a worker.

    Parameters
    ----------
    task : ("query", str) | ("reader", dict)
        Kind discriminator plus its payload.
    meta : pandas.DataFrame
        Schema holder used to enforce dtype consistency.
    conn_kwargs, query_kwargs : dict
        Forwarded to quasardb.Cluster / quasardb.pandas.query.

    Returns
    -------
    pandas.DataFrame

    Decision rationale:
    • Keeps QuasarDB traffic on the worker, avoiding client-side deps.

    Performance trade-offs:
    • Returning *meta* for empty partitions avoids needless serialisation.
    """
    task_type, payload = task
    with quasardb.Cluster(**conn_kwargs) as conn:
        if task_type == "query":
            df = _execute_query(conn, payload, query_kwargs)
        elif task_type == "reader":
            df = _execute_reader(conn, payload)
        else:
            raise ValueError(f"run_partition_task: unknown task {task_type!r}")

    if df.empty:  # keep schema when partition has no rows
        return meta

    _restore_empty_float_columns(df, meta)

    return df


def _execute_query(conn, query: str, query_kwargs: dict) -> pd.DataFrame:
    """
    Executes a SELECT query on the cluster and returns the result as a DataFrame.

    Parameters
    ----------
    conn : quasardb.Cluster
        Active cluster connection.
    query : str
        SQL SELECT statement.
    query_kwargs : dict
        Forwarded to quasardb.pandas.query.

    Returns
    -------
    pandas.DataFrame

    Notes
    -----
    Used for partitioned Dask execution to keep computation on the worker.
    """
    df = qdbpd.query(conn, query, **query_kwargs)
    _coerce_timestamp_index(df)
    return df


def _execute_reader(conn, args: dict) -> pd.DataFrame:
    """
    Reads a DataFrame from the cluster using the provided arguments.

    Parameters
    ----------
    conn : quasardb.Cluster
        Active cluster connection.
    args : dict
        Must contain "table", "ranges", "column_names".

    Returns
    -------
    pandas.DataFrame

    Raises
    ------
    KeyError
        If any required key is missing.

    Notes
    -----
    Used for reading partitioned data from temporary tables.
    """
    required = {"table", "ranges", "column_names"}
    if missing := required - args.keys():
        raise KeyError(f"_execute_reader: missing {', '.join(missing)}")

    df = qdbpd.read_dataframe(conn, **args)
    _coerce_timestamp_index(df)
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
        np.dtype("O"): quasardb.ColumnType.String,  # Objects are strings
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


def _df_to_ranges(
    df: pd.DataFrame,
    *,
    npartitions: int,
) -> list[tuple[np.datetime64, np.datetime64]]:
    """
    Split a DataFrame’s DatetimeIndex into *npartitions* consecutive,
    non-overlapping, half-open intervals.

    Parameters
    ----------
    df : pandas.DataFrame
        Non-empty frame whose index is ``DatetimeIndex[ns]``.
    npartitions : int
        Number of desired splits (≥ 1).

    Returns
    -------
    list[tuple[numpy.datetime64, numpy.datetime64]]
        ``[(start0, end0), (start1, end1), …]`` where each *end* timestamp is
        exclusive.

    Raises
    ------
    AssertionError
        If *df* is empty or its index is not ``datetime64[ns]``.
        If *npartitions* is ``None`` or < 2.

    Decision rationale:
    • Keeps implementation simple and predictable; no shard alignment here.

    Performance trade-offs:
    • Uses vectorised `pd.date_range`, negligible cost for non-massive inputs.

    """
    assert not df.empty, "_df_to_ranges: dataframe must be non-empty"
    assert df.index.dtype == np.dtype("datetime64[ns]"), (
        "_df_to_ranges: index must be datetime64[ns], " f"got {df.index.dtype}"
    )
    assert npartitions is not None, "_df_to_ranges: npartitions must be an int > 1"
    assert npartitions > 1, "_df_to_ranges: npartitions must be > 1"

    first = pd.Timestamp(df.index.min())
    last_exclusive = pd.Timestamp(df.index.max()) + pd.Timedelta("1ns")

    # We create an **evenly-spaced grid** of npartitions + 1 timestamps that
    # spans the whole interval `[first, last_exclusive)`.
    #   boundaries[0]           == first
    #   boundaries[-1]          == last_exclusive
    # Consecutive pairs of these timestamps form the `npartitions` *half-open*
    # intervals we want:  [s, e) with `s` included and `e` excluded.
    boundaries = pd.date_range(start=first, end=last_exclusive, periods=npartitions + 1)
    # Convert every `[start, end)` pair to a tuple of numpy.datetime64[ns]
    # values – this keeps the downstream serialisation compact and avoids
    # dtype surprises when the ranges are fed into the QuasarDB reader.
    return [
        (s.to_datetime64(), e.to_datetime64())
        for s, e in zip(boundaries[:-1], boundaries[1:])
    ]


def materialize_to_temp(
    query: str,
    meta: pd.DataFrame,
    conn_kwargs: dict,
    query_kwargs: dict,
    npartitions: int,
) -> list[tuple]:
    """
    Materialises *query* into a fresh temporary QuasarDB table and returns a
    rewritten SELECT statement that reads the data back.

    Parameters
    ----------
    query : str
        User-supplied SELECT statement.
    meta : pandas.DataFrame
        Empty DataFrame describing the expected schema; reused for table
        creation and column projection.
    npartitions : int
        Number of distinct partitions to return
    conn_kwargs : dict
        Arguments forwarded to ``quasardb.Cluster``.
    query_kwargs : dict
        Arguments passed verbatim to ``quasardb.pandas.query``.

    Returns
    -------
    list[tuple]
        Single-element list containing new tasks. Tasks are defined as a tuple
        of (task_type, task_info).

    Raises
    ------
    ValueError
        If the result lacks a ``$timestamp`` column.
    TypeError
        If ``$timestamp`` is not dtype ``datetime64[ns]``.

    Decision rationale:
    • Certain queries (window functions, ASOF joins …) cannot be split for
      Dask; persisting once on the server avoids repeated large transfers.
    • Ksuid-based table names are time-sortable, minimising RocksDB compaction.

    Key assumptions:
    • *query* is a pure, side-effect-free SELECT.
    • Result set includes a ``$timestamp`` column with dtype ``datetime64[ns]``.

    Performance trade-offs:
    • Adds a single write and short-lived storage footprint (TTL one day),
      but downstream tasks read locally from the cluster, improving throughput.

    """

    with quasardb.Cluster(**conn_kwargs) as conn:

        # Work with a copy so caller’s dict stays untouched and ensure a default.
        local_qkwargs = query_kwargs.copy()
        local_qkwargs.setdefault("index", "$timestamp")

        df = qdbpd.query(conn, query, **local_qkwargs)
        _coerce_timestamp_index(df)

        # --- validation ----------------------------------------------------
        # We require an index of dtype datetime64[ns]; anything
        # else will break downstream assumptions about indexing and typing.
        if df.index.dtype != np.dtype("datetime64[ns]"):
            raise TypeError(
                "materialize_to_temp: index must be dtype `datetime64[ns]`, "
                f"got {df.index.dtype}"
            )
        # -------------------------------------------------------------------

        # We use Ksuid() so that the tables we create are sorted by time. This is highly effective for rocksdb,
        # as this pretty much guarantees that newly created tables don't overlap with old tables, and as such reduce
        # a lot of rocksdb compaction pressure.
        table_name = "qdb/dask/temp/{}".format(Ksuid())

        # Create the table *after* we fetched the data, because now we at least know that the query succeeded
        # and don't garbage leave empty tables around.
        #
        # We use a TTL of 1 day so that data is cleared up quickly
        shard_size = pendulum.duration(days=1)
        ttl = pendulum.duration(days=1)

        _create_table_from_meta(
            conn, table_name=table_name, meta=meta, shard_size=shard_size, ttl=ttl
        )

        qdbpd.write_dataframe(
            df, conn, table_name, push_mode=quasardb.WriterPushMode.Fast
        )

        # Build the result tasks: we'll use the bulk reader implementation in this case as
        # it's more efficient and we want to return an entire dataframe's results.
        #
        # Start by building (start, end) ranges for the reader task.
        # derive a split count that roughly follows shard_size semantics
        ranges = _df_to_ranges(df, npartitions=npartitions)

        # ------------------------------------------------------------------
        # Select only the *user-visible* columns.
        #
        # QuasarDB prefixes internal bookkeeping fields with “$” (e.g. `$table`,
        # `$timestamp`).  Those are either injected automatically or not needed
        # in the user’s result, so requesting them again would just waste I/O.
        # ------------------------------------------------------------------
        column_names: list[str] = [c for c in meta.columns if not c.startswith("$")]

        # Build one reader task per (start, end) interval.We provide “ranges”
        # which is what `qdbpd.read_dataframe` expects, but just put a single
        # range in there.
        return [
            (
                "reader",
                {
                    "table": table_name,
                    "ranges": [rng],
                    "column_names": column_names,
                },
            )
            for rng in ranges
        ]


def write_df(
    df: pd.DataFrame,
    # conn options
    conn_kwargs: dict,
    # write options
    table: str,
    create: bool,
    push_mode: quasardb.WriterPushMode,
    shard_size: pendulum.Duration,
    deduplicate: bool,
    infer_types: bool,
    deduplication_mode: str,
) -> bool:
    """
    Creates a connection and writes dataframe to a QuasarDB table.

    Returns True on success.
    """
    _coerce_timestamp_index(df)
    # drop internal $table column before writing
    # this addresses the case where the user passed a result of a "SELECT * FROM table" query,
    # which would have a $table column in it
    # without this we would get an error for reserved alias
    if "$table" in df.columns:
        logger.warning("Dropping internal $table column before writing!")
        df.drop(columns=["$table"], inplace=True, errors="ignore")

    with quasardb.Cluster(**conn_kwargs) as conn:
        qdbpd.write_dataframe(
            df,
            conn,
            table=table,
            create=create,
            push_mode=push_mode,
            shard_size=shard_size,
            deduplicate=deduplicate,
            deduplication_mode=deduplication_mode,
            infer_types=infer_types,
        )

    return True
