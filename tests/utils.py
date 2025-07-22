import logging
import numpy as np
import numpy.ma as ma
import pandas as pd
import quasardb.pandas as qdbpd


def _to_numpy_masked(xs):
    data = xs.to_numpy()
    mask = xs.isna()
    return ma.masked_array(data=data, mask=mask)


def _assert_series_equal(lhs, rhs):
    lhs_ = _to_numpy_masked(lhs)
    rhs_ = _to_numpy_masked(rhs)

    assert ma.count_masked(lhs_) == ma.count_masked(rhs_)

    logging.debug("lhs: %s", lhs_[:10])
    logging.debug("rhs: %s", rhs_[:10])

    lhs_ = lhs_.torecords()
    rhs_ = rhs_.torecords()

    for (lval, lmask), (rval, rmask) in zip(lhs_, rhs_):
        assert lmask == rmask

        if lmask is False:
            assert lval == rval


def assert_df_equal(lhs, rhs):
    """
    Verifies DataFrames lhs and rhs are equal(ish). We're not pedantic that we're comparing
    metadata and things like that.

    Typically one would use `lhs` for the DataFrame that was generated in code, and
    `rhs` for the DataFrame that's returned by qdbpd.
    """

    np.testing.assert_array_equal(lhs.index.to_numpy(), rhs.index.to_numpy())
    assert len(lhs.columns) == len(rhs.columns)
    for col in lhs.columns:
        _assert_series_equal(lhs[col], rhs[col])


def prepare_query_test(
    df_with_table,
    qdbd_connection,
    columns: str = "*",
    query_range: tuple[pd.Timestamp, pd.Timestamp] = None,
    group_by: str = None,
):
    (_, _, df, table) = df_with_table

    qdbpd.write_dataframe(df, qdbd_connection, table)
    table_name = table.get_name()
    q = 'SELECT {} FROM "{}"'.format(columns, table_name)

    if query_range:
        q += " IN RANGE({}, {})".format(query_range[0], query_range[1])

    if group_by:
        q += " GROUP BY {}".format(group_by)

    return (df, table, q)


def get_subrange(df: pd.DataFrame, slice_size: int = 0.1) -> tuple[str, str]:
    """
    Returns slice of the Dataframe index to be used in the query.
    """
    query_range = ()
    if slice_size != 1:
        start_str = df.index[0].to_pydatetime().strftime("%Y-%m-%dT%H:%M:%S.%f")
        end_row = int((len(df) - 1) * slice_size)
        end_str = df.index[end_row].to_pydatetime().strftime("%Y-%m-%dT%H:%M:%S.%f")
        query_range = (start_str, end_str)
    return query_range
