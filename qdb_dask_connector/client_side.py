import re
import sys
import logging
import datetime
import dateparser
import pandas as pd

logger = logging.getLogger("quasardb_dask")


class QdbPythonApiRequired(ImportError):
    """
    Exception raised when trying to use QuasarDB dask integration, in Python API client mode,
    but quasardb package is not installed.
    """

    pass


try:
    import quasardb
    import quasardb.pandas as qdbpd
except ImportError as err:
    logging.warning(
        "QuasarDB Python API is not installed. "
        "QuasarDB Dask integration can work with either REST API or Python API for client calls "
        "QuasarDB Python API is still required on the server side of Dask. "
    )
    logger.exception(err)


def ensure_python_api_imported():
    """
    qdb-dask-connector can work in two modes client side:
    1. Python API client mode
    2. REST API client mode

    This allows decoupling of QuasarDB Python API from Dask integration (CLIENT SIDE ONLY, Python API is still required on the server side of Dask).
    In big deployments, with many Dask clients it means that clients don't have to install and keep QuasarDB Python API up to date.

    If wants to use Python API client mode we need to check if quasardb package is imported.
    If not, we raise an ImportError.
    """

    if "quasardb" not in sys.modules or "quasardb.pandas" not in sys.modules:
        raise QdbPythonApiRequired(
            "QuasarDB Python API is missing from your environment. "
            "QuasarDB dask integration can work with either REST API or Python API. "
            "Please install 'quasardb' package if you want to use the Python API client mode. "
            "If you want to use the REST API client mode, please provide the 'rest_uri' parameter to `query` function."
        )


# REGEX patterns used to parse the query
general_select_pattern = re.compile(r"(?i)^\s*SELECT\b")
table_pattern = re.compile(r"(?i)\bFROM\b\s+([^\s]+)")
range_pattern = re.compile(r"(?i)\bIN\s+RANGE\s*\(([^,]+),\s*([^,]+)\)")


def _extract_table_name_from_query(query: str) -> str:
    # this function is used to mock part of future c api functionality

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
    # this function is used to mock part of future c api functionality
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
    # this function is used to mock part of future c api functionality
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
        logger.debug("Created query: %s", new_query)
        return new_query

    table_match = re.search(table_pattern, query)
    new_query = re.sub(
        table_pattern,
        f"FROM {table_match.group(1)} IN RANGE ({start_str}, {end_str})",
        query,
    )
    logger.debug("Created query: %s", new_query)
    return new_query


def split_query(conn, query: str) -> list[str]:
    # this function is used to mock part of future c api functionality
    table_name = _extract_table_name_from_query(query)
    shard_size = conn.table(table_name.replace('"', "")).get_shard_size()
    start, end = _extract_range_from_query(conn, query, table_name)
    ranges_to_query = conn.split_query_range(start, end, shard_size)

    split_queries = []
    for rng in ranges_to_query:
        split_queries.append(_create_subrange_query(query, rng))
    return split_queries


def get_meta(conn, query: str, query_kwargs: dict) -> pd.DataFrame:
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


def get_tasks_from_rest_api(
    query: str, rest_api_uri: str, query_args: dict
) -> tuple[list[str], pd.DataFrame]:
    """
    Connects to QuasarDB cluster using QuasarDB Rest API and:
    1. Extracts expected schema of the query result
    2. Splits the query into smaller queries which can be executed on Dask cluster in parallel.

    Requires QuasarDB Rest API to be running and accessible at the provided `rest_api_uri`.
    """
    raise NotImplementedError(
        "REST API client mode is not implemented yet. Please use the Python API client mode."
    )
    # XXX:igor i used this code to validate that dask integration can work even when quasardb api is not installed
    return [query], pd.DataFrame(
        {
            "$timestamp": pd.Series(dtype="datetime64[ns]"),
            "$table": pd.Series(dtype="object"),
            "x": pd.Series(dtype="float64"),
        }
    )


def get_tasks_from_python_api(
    query: str, conn_kwargs: dict, query_kwargs: dict
) -> tuple[list[str], pd.DataFrame]:
    """
    Connects to QuasarDB cluster using Python API and:
    1. Extracts expected schema of the query result
    2. Splits the query into smaller queries which can be executed on Dask cluster in parallel.

    Requires up-to date QuasarDB Python API to be installed in the environment.
    """
    ensure_python_api_imported()
    with quasardb.Cluster(**conn_kwargs) as conn:
        meta = get_meta(conn, query, query_kwargs)
        return split_query(conn, query), meta
