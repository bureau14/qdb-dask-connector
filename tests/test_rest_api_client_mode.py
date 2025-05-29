import pytest
import quasardb_dask as qdbdsk
import logging

logger = logging.getLogger("test-dask-rest-api-client-mode")


def test_client_in_rest_mode_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        qdbdsk.query(
            "select * from test",
            cluster_uri="qdb://127.0.0.1:2836",
            rest_uri="localhost:40080",
        )
