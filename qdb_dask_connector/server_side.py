import logging
import pandas as pd

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
