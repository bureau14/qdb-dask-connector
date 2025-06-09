import re
import logging
import datetime
import dask.dataframe as dd
from dask.delayed import delayed
from uuid import uuid4

logger = logging.getLogger("quasardb_dask")

from .server_side import *
from .client_side import *


def _ensure_select_query(query: str):
    if not re.match(general_select_pattern, query):
        raise NotImplementedError(
            "Only SELECT queries are supported. Please refer to the documentation for more information."
        )


def query(
    query: str,
    *,
    cluster_uri: str,
    # rest api options
    rest_uri: str = "",
    # python api options
    user_name: str = "",
    user_private_key: str = "",
    cluster_public_key: str = "",
    timeout: datetime.timedelta = datetime.timedelta(seconds=60),
    enable_encryption: bool = False,
    client_max_parallelism: int = 0,
    # query options
    index: str = None,  # XXX:igor this should be a union str | int, not possible with python<=3.9
):
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

    query_kwargs = {
        "index": index,
    }

    if rest_uri:
        split_queries, meta = get_tasks_from_rest_api(query, rest_uri, query_kwargs)
    else:
        split_queries, meta = get_tasks_from_python_api(
            query, conn_kwargs, query_kwargs
        )

    if len(split_queries) == 0:
        logger.warning("No split queries, returning empty dataframe")
        return meta

    parts = []
    for split_query in split_queries:
        parts.append(
            delayed(read_dataframe, pure=True)(
                split_query, meta, conn_kwargs, query_kwargs
            )
        )
    logger.debug("Assembled %d split queries", len(parts))

    return dd.from_delayed(parts, meta=meta)


def _query_to_queries_from_temp_table(
    query: str, meta: pd.DataFrame, conn_kwargs: dict, query_kwargs: dict
) -> list:
    new_table_name = f"qdb/dask/{uuid4()}"  # $ prefix is reserved
    read_write_back_query_to_cluster(
        query, new_table_name, meta, conn_kwargs, query_kwargs
    )
    new_query = f'SELECT * FROM "{new_table_name}"'
    with quasardb.Cluster(**conn_kwargs) as conn:
        split_queries = split_query(conn, new_query)

    parts = []
    for partial_query in split_queries:
        parts.append(
            delayed(read_dataframe)(
                partial_query, meta, conn_kwargs, query_kwargs
            ).persist()
        )
    logger.debug("Assembled %d split queries", len(parts))
    return parts


def query_persist(
    query: str,
    *,
    cluster_uri: str,
    # rest api options
    rest_uri: str = "",
    # python api options
    user_name: str = "",
    user_private_key: str = "",
    cluster_public_key: str = "",
    timeout: datetime.timedelta = datetime.timedelta(seconds=60),
    enable_encryption: bool = False,
    client_max_parallelism: int = 0,
    # query options
    index: str = None,  # XXX:igor this should be a union str | int, not possible with python<=3.9
):
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

    query_kwargs = {
        "index": index,
    }

    if rest_uri:
        _, meta = get_tasks_from_rest_api(query, rest_uri, query_kwargs)
    else:
        _, meta = get_tasks_from_python_api(query, conn_kwargs, query_kwargs)

    # up to this point its same as query()

    splits = delayed(_query_to_queries_from_temp_table)(
        query, meta, conn_kwargs, query_kwargs
    )

    # currently we run into type incompatibility:
    # `splits` is a Delayed which, once computed will return list[Delayed]
    # dd.from_delayed expects list[Delayed], it gets Delayed[list[Delayed]]
    return dd.from_delayed(splits, meta)
