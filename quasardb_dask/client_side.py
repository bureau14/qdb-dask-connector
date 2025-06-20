import re
import logging
import pandas as pd

import dask.dataframe as dd
from dask.distributed import get_client, wait
from dask.delayed import delayed, Delayed

logger = logging.getLogger("quasardb_dask")

from .server_side import *


# REGEX patterns used to parse the query
table_pattern = re.compile(r"(?i)\bFROM\b\s+([^\s]+)")
range_pattern = re.compile(r"(?i)\bIN\s+RANGE\s*\(([^,]+),\s*([^,]+)\)")

_SELECT_RE = re.compile(r"(?i)^\s*SELECT\b")


def _ensure_select_query(query: str) -> None:
    """
    Raises
    ------
    ValueError
       If *query* does not start with the SQL keyword SELECT (case-insensitive).
    """
    if _SELECT_RE.match(query) is None:
        raise ValueError("Only SELECT statements are supported; got: %r" % query)


def query(
    query: str,
    *,
    cluster_uri: str,
    # python api options
    user_name: str = "",
    user_private_key: str = "",
    cluster_public_key: str = "",
    timeout: pendulum.Duration = pendulum.Duration(seconds=60),
    enable_encryption: bool = False,
    client_max_parallelism: int = 0,
    # temporary hack until we get the `qdb_query_validate` function implemented,
    # this allows the user to specify the actual metadata dataframe so that we don't
    # have to run the entire query to figure out what the type of the result will be.
    meta: pd.DataFrame = None,
    npartitions: int = 1,  # The number of dask partitions to return, which enables parallelization of input queries
):
    """
    Base parameters
    ----------
    query : str
        Query to execute. Must be a valid SELECT statement.

    Connection parameters
    -------------------
    user_name : str
        The username to authenticate with the QuasarDB cluster.
    user_private_key : str
        The private key of the user to authenticate with the QuasarDB cluster.
    cluster_public_key : str
        The public key of the QuasarDB cluster to authenticate with.
    cluster_uri : str
        The URI of the QuasarDB cluster to connect to, e.g. `qdb://127.0.0.1:2836`.
    timeout : pendulum.Duration
        The timeout for the connection to the QuasarDB cluster. Defaults to 60 seconds.
    enable_encryption : bool
        If true, enables encryption for the connection to the QuasarDB cluster. Defaults to False. Requires encryption to be enabled on the QuasarDB cluster. Requires encryption to be enabled on the QuasarDB cluster.
    client_max_parallelism : int
        The maximum number of parallel operations that the client can perform. Defaults to 0.

    Dask parameters
    -------------------
    meta : pandas.DataFrame
        Metadata DataFrame that describes the schema of the result. If not provided, the function will attempt to infer the schema by executing a metadata query.

    npartitions : int
        The number of partitions to split the query into. Defaults to 1.
    """
    _ensure_select_query(query)

    conn_kwargs = {
        "uri": cluster_uri,
        "user_name": user_name,
        "user_private_key": user_private_key,
        "cluster_public_key": cluster_public_key,
        "timeout": timeout,
        "enable_encryption": enable_encryption,
        "client_max_parallelism": client_max_parallelism,
    }

    # Hard-code the canonical index for every query.
    query_kwargs: dict[str, str] = {"index": "$timestamp"}

    ##
    # Coding style for this function:
    #
    # Every `delayed` variable and is not able to be evaluated client-side
    # carries the `_dly` suffix.

    # ------------------------------------------------------------------
    # 1. Retrieve metadata based on the query being executed, i.e. an
    #    empty dataframe that returns the schema of the result.
    #
    #    It's a "pure" function (no server interaction required), but as
    #    this requires the quasardb python API, we execute it on the Dask
    #    workers instead of client-side so that we have no client-side
    #    quasardb python api dependencies. This reduces friction for users.
    # ------------------------------------------------------------------
    meta_dly = delayed(get_meta)(query, conn_kwargs, query_kwargs)

    # But we immediately compute it -- it's very cheap to do, and in the
    # call to `dd.from_delayed`, `meta` is not allowed to be a delayed.
    if meta is None:
        meta = meta_dly.compute()

    logger.info("using meta: %s", meta.dtypes)

    # ------------------------------------------------------------------
    # 1. Take a single large query and split it into smaller tasks
    # ------------------------------------------------------------------
    tasks_dly = delayed(create_partition_tasks)(
        query, meta, conn_kwargs, query_kwargs, npartitions=npartitions
    )

    # ------------------------------------------------------------------
    # 2. Build the delayed read partitions (one per split)
    # ------------------------------------------------------------------
    #
    # These are each "partial" dataframe results.
    #
    # We run it inside a callback so that it's evaluated on a Dask worker, as
    # `split_queries` is a `delayed` and we cannot evaluate it in the client.
    #
    # Returns an array of delayeds
    def _build_df_partitions(xs):
        # Executed on worker
        return [
            delayed(run_partition_task)(x, meta, conn_kwargs, query_kwargs) for x in xs
        ]

    df_partitions_dly = delayed(_build_df_partitions)(tasks_dly)

    # But now we need to "unwrap" the result itself, do that it goes from
    #
    # delayed[list[delayed[df]]]
    #
    # to
    #
    # list[delayed[df]]
    df_partitions = df_partitions_dly.compute()

    # ------------------------------------------------------------------
    # 3. Persist those tasks
    # ------------------------------------------------------------------
    try:
        client = get_client()  # use existing distributed cluster
    except ValueError:  # no distributed scheduler
        logger.warning("No distributed scheduler, falling back to lazy execution")
        persisted_parts = df_partitions  # fall back to lazy execution
    else:
        # Persist each individual query result, which avoids a lot of pitfalls
        # if the user may be using the returned delayed dataframe in incorrect
        # ways and accidentally trigger the same query multiple times.
        #
        # This persists each individual delayed part independently, rather than
        # the list as a whole, meaning there are no large dataframes being
        # cached and the total dataset size can exceed memory.
        logger.info("Using distributed scheduler, persisting individual parts")
        persisted_parts = client.persist(df_partitions, optimize_graph=True)

    # ------------------------------------------------------------------
    # 3. Wrap them in a Dask-dataframe shell
    # ------------------------------------------------------------------
    #
    # persisted_parts is a list of Futures (or Delayed objects on fallback),
    # this code builds a single distributed dataframe out of those.
    return dd.from_delayed(persisted_parts, meta=meta, divisions=None)


def write_dataframe(
    df: dd.DataFrame | pd.DataFrame,
    table: str,
    *,
    cluster_uri: str,
    # python api options
    user_name: str = "",
    user_private_key: str = "",
    cluster_public_key: str = "",
    timeout: pendulum.Duration = pendulum.Duration(seconds=60),
    enable_encryption: bool = False,
    client_max_parallelism: int = 0,
    # write dataframe options
    create: bool = True,
    push_mode=quasardb.WriterPushMode.Fast,
    shard_size: pendulum.Duration = pendulum.Duration(days=1),
    deduplicate: bool = False,
    deduplication_mode="drop",
    infer_types=True,
) -> Delayed:
    """
    Writes DataFrame to a QuasarDB table.

    Accepts either a Dask DataFrame or a Pandas DataFrame and writes it to the specified QuasarDB table.

    Returns a delayed object that **must** be computed to perform the write operation.
    On successful completion, delayed object returns True.

    Base parameters
    ----------
    df : pandas.DataFrame | dask.DataFrame
        DataFrame to write to the QuasarDB table.

    Connection parameters
    -------------------
    user_name : str
        The username to authenticate with the QuasarDB cluster.
    user_private_key : str
        The private key of the user to authenticate with the QuasarDB cluster.
    cluster_public_key : str
        The public key of the QuasarDB cluster to authenticate with.
    cluster_uri : str
        The URI of the QuasarDB cluster to connect to, e.g. `qdb://127.0.0.1:2836`.
    timeout : pendulum.Duration
        The timeout for the connection to the QuasarDB cluster. Defaults to 60 seconds.
    enable_encryption : bool
        If true, enables encryption for the connection to the QuasarDB cluster. Defaults to False
    client_max_parallelism : int
        The maximum number of parallel operations that the client can perform.

    Writer parameters
    -------------------
    table : str
        The name of the QuasarDB table to write the data to.
    push_mode : quasardb.WriterPushMode
        The mode used for inserting data. Defaults to `Fast`.

        Available options:

        * Truncate: Truncates (also referred to as upserts) the data in-place. Will detect time range to truncate from the time range inside the dataframe.

        * Async: Uses asynchronous insertion API where commits are buffered server-side and acknowledged before they are written to disk.

        * Fast: Enables “fast push” for small batch inserts.

        * Transactional: Ensures full transactional consistency.
    create : bool
        If true, automatically creates the table if it did not yet exist.

    shard_size : datetime.timedelta
        If create is True and a table is to be created, uses this value as the shard size. If not, uses QuasarDB’s default shard size (1d).
    deduplicate : bool
        When deduplicate is enabled, specifies how deduplication should be performed. When 'drop', it drops any new data when a duplicate was previously stored. When 'upsert', it replaces the old data with the new data.
    deduplication_mode : str
        When deduplicate is enabled, specifies how deduplication should be performed. When 'drop', it drops any new data when a duplicate was previously stored. When 'upsert', it replaces the old data with the new data.
    infer_types : bool
        If true, will attempt to convert types from Python to QuasarDB native types if the provided dataframe has incompatible types.

        **Important**: as conversions are expensive and often the majority of time spent when inserting data into QuasarDB, we strongly recommend setting this to False for performance-sensitive code.
    """

    conn_kwargs = {
        "uri": cluster_uri,
        "user_name": user_name,
        "user_private_key": user_private_key,
        "cluster_public_key": cluster_public_key,
        "timeout": timeout,
        "enable_encryption": enable_encryption,
        "client_max_parallelism": client_max_parallelism,
    }

    return delayed(write_df)(
        df,
        conn_kwargs,
        table,
        create,
        push_mode,
        shard_size,
        deduplicate,
        infer_types,
        deduplication_mode,
    )
