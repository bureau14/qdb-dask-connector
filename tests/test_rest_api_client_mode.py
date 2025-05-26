import pytest
import qdb_dask_connector
import logging

logger = logging.getLogger("test-dask-rest-api-client-mode")


def test_client_in_rest_mode():
    with pytest.raises(NotImplementedError):
        df = qdb_dask_connector.query(
            "select * from test", "qdb://127.0.0.1:2836", rest_uri="localhost:40080"
        )
