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

    ###
    # XXX(leon): for GeorgiaPacific, we need to temporarily deploy this connector with
    #            a very old version.
    #
    #            in this situation, we don't bother with stuff like materialized views and
    #            whatnot, just return the query and make it a dask persisted dataframe

    # Don't even bother creating multiple partition tasks, just return the query as-is.
    logger.info("single partition input query, avoiding partitioning")
    return [("query", query)]


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
