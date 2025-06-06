from dask.distributed import LocalCluster, Client

from quasardb_dask import *
import logging


logging.basicConfig(level=logging.DEBUG)
import time


def main():
    df = query_persist(
        'SELECT * FROM "dask_test"',
        cluster_uri="qdb://127.0.0.1:2836",
    )
    df.compute()
    print(df.head())


if __name__ == "__main__":
    main()

    # with LocalCluster(n_workers=2) as cluster:
    #     with Client(cluster) as client:
    #         main()
