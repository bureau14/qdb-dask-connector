import logging
import datetime
import re
import pandas as pd

logger = logging.getLogger("qdb-dask-connector")


class DaskRequired(ImportError):
    """
    Exception raised when trying to use QuasarDB dask integration, but
    required packages are not installed.
    """

    pass


try:
    import dask.dataframe as dd
    from dask.delayed import delayed
except ImportError as err:
    logger.exception(err)
    raise DaskRequired(
        "The dask library is required to use QuasarDB dask integration."
    ) from err


class DateParserRequired(ImportError):
    """
    Exception raised when trying to use QuasarDB dask integration, but
    required packages are not installed.
    """

    pass


try:
    import dateparser
except ImportError as err:
    logger.exception(err)
    raise DaskRequired(
        "Dateparser library is required to use QuasarDB dask integration."
    ) from err

### Dask integration can be used with either the QuasarDB Python client or the REST API.
# The default is to use the Python client if available, if not, it will fall back to the REST API.

CLIENT_MODE = "PYTHON_API"
try:
    import quasardb
    import quasardb.pandas as qdbpd
except ImportError as err:
    logger.warning(
        "quasardb or quasardb.pandas not found, will utilize QuasarDB REST API."
    )
    CLIENT_MODE = "REST_API"

# REGEX patterns used to parse the query
general_select_pattern = re.compile(r"(?i)^\s*SELECT\b")
table_pattern = re.compile(r"(?i)\bFROM\s+([`\"\[]?\w+[`\"\]]?)")
range_pattern = re.compile(r"(?i)\bIN\s+RANGE\s*\(([^,]+),\s*([^,]+)\)")


def _read_dataframe(
    query: str, meta: pd.DataFrame, conn_kwargs: dict, query_kwargs: dict
):
    """
    Creates connection and queries cluster with passed query.
    """
    logger.debug('Querying QuasarDB with query: "%s"', query)
    with quasardb.Cluster(**conn_kwargs) as conn:
        df = qdbpd.query(conn, query, **query_kwargs)

    if len(df) == 0:
        return meta
    else:
        return df


def _extract_table_name_from_query(query: str) -> str:
    # XXX:igor for now this works for queries using one table
    # tags and multiple tables are not supported yet

    logger.debug('Extracting table name from query: "%s"', query)
    match = re.search(table_pattern, query)
    if match:
        table_name = match.group(1)
        logger.debug('Extracted table name: "%s"', table_name)
        return table_name
    else:
        raise ValueError("Could not extract table name from query. ")


def _extract_range_from_query(conn, query: str, table_name: str) -> tuple:
    """
    Extracts the range from the query, parses it to datetime and returns.
    If no range is found in the query, it queries the table for the first and last timestamp.
    """
    logger.debug('Extracting query range from: "%s"', query)
    match = re.search(range_pattern, query)
    # first we check try to extract "in range (start, end)" from query
    # if we can't do it we will query first() and last() from the table
    query_range = tuple()
    if match:
        start_str = match.group(1)
        end_str = match.group(2)
        logger.debug("Extracted strings: (%s, %s)", start_str, end_str)
        parser_settings = {
            "PREFER_DAY_OF_MONTH": "first",
            "PREFER_MONTH_OF_YEAR": "first",
        }
        start_date = dateparser.parse(start_str, settings=parser_settings)
        end_date = dateparser.parse(end_str, settings=parser_settings)
        query_range = (start_date, end_date)
        logger.debug("Parsed datetime: %s", query_range)
    else:
        logger.debug(
            "No range found in query, querying table for first and last timestamp"
        )
        range_query = f"SELECT first($timestamp), last($timestamp) FROM {table_name}"
        df = qdbpd.query(conn, range_query)
        if not df.empty:
            df.loc[0, "last($timestamp)"] += datetime.timedelta(microseconds=1)
            query_range += tuple(df.iloc[0])
            logger.debug("Extracted range from table: %s", query_range)

    return query_range


def _create_subrange_query(
    query: str, query_range: tuple[datetime.datetime, datetime.datetime]
) -> str:
    """
    Adds range to base query.
    IF range is found in the query, it will be replaced with the new range.
    IF no range is found, it will be added after the "FROM {table}" clause.
    """
    new_query = query
    range_match = re.search(range_pattern, query)
    start_str = query_range[0].strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    end_str = query_range[1].strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    if range_match:
        if len(range_match.groups()) == 2:
            new_query = re.sub(
                range_pattern,
                f"IN RANGE ({start_str}, {end_str})",
                query,
            )
        logger.debug("Created subquery: %s", new_query)
        return new_query

    table_match = re.search(table_pattern, query)
    new_query = re.sub(
        table_pattern,
        f"FROM {table_match.group(1)} IN RANGE ({start_str}, {end_str})",
        query,
    )
    logger.debug("Created subquery: %s", new_query)
    return new_query


def _get_subqueries(conn, query: str, table_name: str) -> list[str]:
    # this will be moved to c++ functions in the future
    shard_size = conn.table(table_name.replace('"', "")).get_shard_size()
    start, end = _extract_range_from_query(conn, query, table_name)
    ranges_to_query = conn.split_query_range(start, end, shard_size)

    subqueries = []
    for rng in ranges_to_query:
        subqueries.append(_create_subrange_query(query, rng))
    return subqueries


def _get_meta(conn, query: str, query_kwargs: dict) -> pd.DataFrame:
    """
    Returns empty dataframe with the expected schema of the query result.
    """
    np_res = conn.validate_query(query)
    col_dtypes = {}
    for id, column in enumerate(np_res):
        col_dtypes[column[0]] = pd.Series(dtype=column[1].dtype)

    df = pd.DataFrame(col_dtypes)
    if query_kwargs["index"]:
        df.set_index(query_kwargs["index"], inplace=True)
    return df


def _get_tasks_from_rest_api(
    query: str, rest_api_uri: str
) -> tuple[list[str], pd.DataFrame]:
    return ["select * from test"], pd.DataFrame()


def _get_subqueries_from_python_client(
    query: str, conn_kwargs: dict, query_kwargs: dict
) -> tuple[list[str], pd.DataFrame]:
    table_name = _extract_table_name_from_query(query)
    with quasardb.Cluster(**conn_kwargs) as conn:
        meta = _get_meta(conn, query, query_kwargs)
        return _get_subqueries(conn, query, table_name), meta


def query(
    query: str,
    cluster_uri: str,
    client_mode: str = None,
    # rest api options
    rest_api_uri: str = "",
    *,
    # python api options
    user_name: str = "",
    user_private_key: str = "",
    cluster_public_key: str = "",
    user_security_file: str = "",
    cluster_public_key_file: str = "",
    timeout: datetime.timedelta = datetime.timedelta(seconds=60),
    do_version_check: bool = False,
    enable_encryption: bool = False,
    client_max_parallelism: int = 0,
    # query options
    index=None,
    blobs: bool = False,
    numpy: bool = False,
):
    if not re.match(general_select_pattern, query):
        raise NotImplementedError(
            "Only SELECT queries are supported. Please refer to the documentation for more information."
        )

    global CLIENT_MODE
    if client_mode:
        if client_mode not in ["PYTHON_API", "REST_API"]:
            raise ValueError("client_mode must be either 'PYTHON_API' or 'REST_API'.")
        CLIENT_MODE = client_mode

    conn_kwargs = {
        "uri": cluster_uri,
        "user_name": user_name,
        "user_private_key": user_private_key,
        "cluster_public_key": cluster_public_key,
        "user_security_file": user_security_file,
        "cluster_public_key_file": cluster_public_key_file,
        "timeout": timeout,
        "do_version_check": do_version_check,
        "enable_encryption": enable_encryption,
        "client_max_parallelism": client_max_parallelism,
    }

    query_kwargs = {
        "index": index,
        "blobs": blobs,
        "numpy": numpy,
    }

    if CLIENT_MODE == "PYTHON_API":
        subqueries, meta = _get_subqueries_from_python_client(
            query, conn_kwargs, query_kwargs
        )
    elif CLIENT_MODE == "REST_API":
        if not rest_api_uri:
            raise ValueError(
                "REST API URI must be provided when using REST API client mode."
            )
        conn_kwargs["uri"] = rest_api_uri
        subqueries, meta = _get_tasks_from_rest_api(query, rest_api_uri)

    if len(subqueries) == 0:
        logging.warning("No subqueries, returning empty dataframe")
        return meta

    parts = []
    for subquery in subqueries:
        parts.append(
            delayed(_read_dataframe)(subquery, meta, conn_kwargs, query_kwargs)
        )
    logger.debug("Assembled %d subqueries", len(parts))

    return dd.from_delayed(parts, meta=meta)
