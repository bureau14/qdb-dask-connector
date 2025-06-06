### this file is added for convinience and it shouldn't be included in PR!

from dask.distributed import LocalCluster, Client

from quasardb_dask import *
import logging

logging.basicConfig(level=logging.DEBUG)


def example():
    # create a lazy task
    df = query_persist(
        'SELECT * FROM "dask_test"',
        cluster_uri="qdb://127.0.0.1:2836",
    )
    # compute into dataframe
    df = df.compute()
    # print standard pandas dataframe
    print(df.head())


if __name__ == "__main__":

    # this is a super simple example that validates that your function works
    example()

    # if you want to see how dask distribiutes this on a cluster use this code

    # with LocalCluster(n_workers=2) as cluster:
    #     with Client(cluster) as client:
    #         example()
