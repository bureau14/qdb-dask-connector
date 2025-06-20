import os
import json
import random
import string
import pytest
import numpy as np
import numpy.ma as ma
import pandas as pd
import datetime
from functools import partial

import quasardb
import quasardb.pandas as qdbpd


def connect(uri):
    return quasardb.Cluster(uri)


def _random_string():
    return "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(16)
    )


@pytest.fixture
def entry_name(random_string):
    return _random_string()


@pytest.fixture
def random_string():
    return _random_string()


@pytest.fixture
def table_name(entry_name):
    return entry_name


def _random_identifier():
    return "".join(random.choice(string.ascii_lowercase) for _ in range(16))


@pytest.fixture
def column_name():
    return _random_identifier()


def _sparsify(perc, xs):
    # Randomly make a bunch of elements null, we make use of Numpy's masked
    # arrays for this: keep track of a separate boolean 'mask' array, which
    # determines whether or not an item in the array is None.
    #
    # `perc` denotes the percentage of "sparsity", where 100 means no None
    # elements, and 0 means everything is None
    assert perc <= 100
    assert perc >= 0
    assert type(perc) == int

    n = len(xs)

    if perc == 0:
        # When everything is 0, don't use a masked array; it's more "intrusive"
        # if we would just literally return an array with nulls, we don't need
        # a mask here.
        return np.full(n, None)
    elif perc == 100:
        return xs

    choices = np.full(100, True)
    choices[:perc] = False

    mask = np.random.choice(a=choices, size=n)

    return ma.array(data=xs, mask=mask)


no_sparsify = [lambda x: x]


@pytest.fixture(
    params=[100, 50, 0], ids=["sparsify=none", "sparsify=partial", "sparsify=all"]
)
def sparsify(request):
    return partial(_sparsify, request.param)


# Makes it easier to "select" a sparsification mode.
#
# TODO(leon): accept array of sparsify types we accept, so that you can select e.g. ['partial', 'none']
def override_sparsify(x):
    m = {
        "none": partial(_sparsify, 100),
        "partial": partial(_sparsify, 50),
        "all": partial(_sparsify, 0),
    }

    xs = m[x]
    return pytest.mark.parametrize("sparsify", [xs], ids=[x])


def _gen_floating(n, low=-100.0, high=100.0):
    return np.random.uniform(low, high, n)


def _gen_float64(n, low=-100.0, high=100.0):
    return _gen_floating(n, low=low, high=high).astype(np.float64)


def _gen_float32(n, low=-100.0, high=100.0):
    return _gen_floating(n, low=low, high=high).astype(np.float32)


def _gen_float16(n, low=-100.0, high=100.0):
    return _gen_floating(n, low=low, high=high).astype(np.float16)


def _gen_integer(n):
    return np.random.randint(-100, 100, n)


def _gen_int64(n):
    return _gen_integer(n).astype(np.int64)


def _gen_int32(n):
    return _gen_integer(n).astype(np.int32)


def _gen_int16(n):
    return _gen_integer(n).astype(np.int16)


def _gen_unicode_word(n):

    ###
    # XXX(leon): i have no time to fully investigate right now, but under certain environments,
    #            for some reason, this causes problems. it *may* be related to reusing memory
    #            of underlying numpy arrays or something like that, i do not know exactly, but
    #            it complains about unicode conversion errors if you generate multiple dataframes.
    #
    #            need to dig deeper, but don't have time right now. i think maybe we're reusing
    #            the same numpy array memory arenas in multiple dataframes or something like that.
    #            but as of now, 2023-09-14, i am too busy. if this ever causes problems down the
    #            road, Mea Culpa, it is my fault.

    try:
        get_char = unichr
    except NameError:
        get_char = chr

    include_ranges = [
        (0x0021, 0x0021),
        (0x0023, 0x0026),
        (0x0028, 0x007E),
        (0x00A1, 0x00AC),
        (0x00AE, 0x00FF),
        (0x0100, 0x017F),
        (0x0180, 0x024F),
        (0x2C60, 0x2C7F),
        (0x16A0, 0x16F0),
        (0x0370, 0x0377),
        (0x037A, 0x037E),
        (0x0384, 0x038A),
        (0x038C, 0x038C),
    ]

    alphabet = [
        get_char(code_point)
        for current_range in include_ranges
        for code_point in range(current_range[0], current_range[1] + 1)
    ]

    return "".join(random.choice(alphabet) for i in range(n))


def _gen_unicode(n):
    """
    Returns `n` strings of length max<n, 256>
    """

    min_word_length = 1
    max_word_length = n

    # Cap the word length at 256
    if max_word_length > 256:
        max_word_length = 256

    if max_word_length < 16:
        max_word_length = 16

    wordslengths = np.random.randint(8, 100, n)
    xs = list(
        _gen_unicode_word(random.randint(min_word_length, max_word_length))
        for i in range(n)
    )

    return np.array(xs, dtype="U")


def _gen_blob(n):
    return list(x.encode("utf-8") for x in _gen_string(n))


def _gen_datetime64(n, precision):
    start_time = np.datetime64("2017-01-01", precision)

    dt = np.dtype("datetime64[{}]".format(precision))
    return np.array([(start_time + np.timedelta64(i, "s")) for i in range(n)]).astype(
        dt
    )


def _gen_datetime64_s(n):
    return _gen_datetime64(n, "s")


def _gen_datetime64_ms(n):
    return _gen_datetime64(n, "ms")


def _gen_datetime64_ns(n):
    return _gen_datetime64(n, "ns")


def _gen_timestamp(n):
    return _gen_datetime64_ns(n)


def _gen_string(n):
    # Slightly hacks, but for testing purposes the contents of the strings
    # being floats allows us to always cast these strings to other types as
    # well.
    return list(str(x) for x in _gen_floating(n, low=0))


_dtype_to_column_type = {
    np.dtype("int64"): quasardb.ColumnType.Int64,
    np.dtype("int32"): quasardb.ColumnType.Int64,
    np.dtype("int16"): quasardb.ColumnType.Int64,
    np.dtype("float64"): quasardb.ColumnType.Double,
    np.dtype("float32"): quasardb.ColumnType.Double,
    np.dtype("float16"): quasardb.ColumnType.Double,
    np.dtype("unicode"): quasardb.ColumnType.String,
    np.dtype("bytes"): quasardb.ColumnType.Blob,
    np.dtype("datetime64[ns]"): quasardb.ColumnType.Timestamp,
    np.dtype("datetime64[ms]"): quasardb.ColumnType.Timestamp,
    np.dtype("datetime64[s]"): quasardb.ColumnType.Timestamp,
}

_generators = {
    quasardb.ColumnType.Int64: {
        np.dtype("int64"): (True, _gen_int64),
        np.dtype("int32"): (True, _gen_int32),
        np.dtype("int16"): (True, _gen_int16),
    },
    quasardb.ColumnType.Double: {
        np.dtype("float64"): (True, _gen_float64),
        np.dtype("float32"): (True, _gen_float32),
        np.dtype("float16"): (False, _gen_float16),
    },
    quasardb.ColumnType.Timestamp: {
        np.dtype("datetime64[ns]"): (True, _gen_datetime64_ns),
        np.dtype("datetime64[ms]"): (False, _gen_datetime64_ms),
        np.dtype("datetime64[s]"): (False, _gen_datetime64_s),
    },
    quasardb.ColumnType.String: {
        np.dtype("unicode"): (True, _gen_unicode),
        np.dtype("object"): (False, _gen_string),
    },
    quasardb.ColumnType.Blob: {np.dtype("bytes"): (True, _gen_blob)},
}


def _cdtype_to_generator(cdtype):
    (ctype, dtype) = cdtype

    assert ctype in _generators
    assert dtype in _generators[ctype]

    (native, gen) = _generators[ctype][dtype]
    return gen


def _gen_array(cdtype, sparsify, row_count):

    (ctype, dtype) = cdtype

    assert isinstance(ctype, quasardb.ColumnType)
    assert isinstance(dtype, np.dtype)

    fn = _cdtype_to_generator(cdtype)
    xs = fn(row_count)
    xs = sparsify(xs)

    if isinstance(xs, list):
        xs = np.array(xs, dtype=dtype)

    return (ctype, dtype, xs)


@pytest.fixture(params=[np.datetime64("2017-01-01", "ns")], ids=["start_date=20170101"])
def start_date(request):
    yield request.param


@pytest.fixture
def gen_array(gen_cdtype, sparsify, row_count):
    return _gen_array(gen_cdtype, sparsify, row_count)


@pytest.fixture(params=["h"], ids=["frequency=H"])
def frequency(request):
    yield request.param


@pytest.fixture
def gen_index(start_date, row_count, frequency):
    return pd.Index(
        pd.date_range(start_date, periods=row_count, freq=frequency), name="$timestamp"
    )


@pytest.fixture
def gen_df(gen_array, gen_index, column_name):
    return _do_gen_df(gen_array, gen_index, column_name)


def _do_gen_df(gen_array, gen_index, column_name):
    (ctype, dtype, xs) = gen_array
    idx = gen_index

    return (
        ctype,
        dtype,
        pd.DataFrame(data={column_name: np.copy(xs)}, index=np.copy(idx)),
    )


def _qdbd_settings():
    user_key = {}
    cluster_key = ""

    with open("user_private.key", "r") as user_key_file:
        user_key = json.load(user_key_file)
    with open("cluster_public.key", "r") as cluster_key_file:
        cluster_key = cluster_key_file.read()
    return {
        "uri": {"insecure": "qdb://127.0.0.1:2836", "secure": "qdb://127.0.0.1:2838"},
        "security": {
            "user_name": user_key["username"],
            "user_private_key": user_key["secret_key"],
            "cluster_public_key": cluster_key,
            "user_private_key_file": "user_private.key",
            "cluster_public_key_file": "cluster_public.key",
            "enable_encryption": os.environ.get("QDB_ENCRYPT_TRAFFIC", "0") != "0",
        },
    }


@pytest.fixture(scope="module")
def qdbd_settings():
    return _qdbd_settings()


def create_qdbd_connection(settings, purge=False):
    conn = connect(settings.get("uri").get("insecure"))

    if purge is True:
        conn.purge_all(datetime.timedelta(minutes=1))

    return conn


@pytest.fixture(scope="module")
def qdbd_connection(qdbd_settings):
    conn = create_qdbd_connection(qdbd_settings, purge=True)
    yield conn
    conn.close()


@pytest.fixture
def df_with_table(qdbd_connection, table_name, column_name, gen_df):
    (ctype, dtype, df) = gen_df

    table = qdbd_connection.table(table_name)

    columns = [quasardb.ColumnInfo(ctype, column_name)]
    table.create(columns)
    return (ctype, dtype, df, table)


@pytest.fixture(params=[224, 288], ids=["row_count=224", "row_count=288"])
def row_count(request):
    yield request.param


def override_row_count(x):
    return pytest.mark.parametrize("row_count", [x], ids=["row_count={}".format(x)])


all_generators = []
for ctype, m in _generators.items():
    for dtype, (native, gen) in m.items():
        all_generators.append((ctype, dtype, native, gen))


def _filter_cdtypes(fn):
    """
    Returns array of cdtypes from the generators where fn(..) yields
    true.
    """
    return [
        (ctype, dtype)
        for (ctype, dtype, native, gen) in all_generators
        if fn(ctype, dtype, native)
    ]


def _cdtype_to_param_id(cdtype):
    (ctype, dtype) = cdtype

    ctype_str = str(ctype).split(".")
    ctype_str = ctype_str[-1].lower()

    return "ctype={}-dtype={}{}".format(ctype_str, dtype.char, dtype.itemsize)


# Extract a list of all 'cdtypes', a List[Tuple[ctype, dtype]] from these
# gengerators.
all_cdtypes = [(ctype, dtype) for (ctype, dtype, _, _) in all_generators]

# Native 'cdtypes' is a List[Tuple[ctype, dtype]] of generators that are
# natively parsable from the C++ backend.
native_cdtypes = _filter_cdtypes(lambda ctype, dtype, native: native is True)

# Native 'cdtypes' is a List[Tuple[ctype, dtype]] of generators that need to be
# converted to another type before sending over to the C++ backend.
inferrable_cdtypes = _filter_cdtypes(lambda ctype, dtype, native: native is False)


assert set(all_cdtypes) == set([*native_cdtypes, *inferrable_cdtypes])


# Makes it easier to "select" a specific column/dtype combination, also
# ensuring the correct friendly "ids" are injected.
#
# It works by overwriting the output of 'gen_cdtype', which affects the
# entire fixture chain/tree.
def override_cdtypes(args):
    if not isinstance(args, list):
        return override_cdtypes([args])

    m = {"native": native_cdtypes, "inferrable": inferrable_cdtypes, "any": all_cdtypes}

    xs = []

    for arg in args:
        if arg in m:
            xs.extend(m[arg])
        elif isinstance(arg, np.dtype):
            xs.extend(list(_filter_cdtypes(lambda ctype, dtype, native: dtype == arg)))

    param_ids = [_cdtype_to_param_id(x) for x in xs]
    return pytest.mark.parametrize("gen_cdtype", xs, ids=param_ids)


@pytest.fixture(params=native_cdtypes, ids=_cdtype_to_param_id)
def gen_cdtype(request):
    yield request.param


@pytest.fixture
def df_with_table_inserted(df_with_table, qdbd_connection):
    """
    Returns a DataFrame that has been written to the table, and the table itself.
    """
    (_, _, df, table) = df_with_table

    # Write the DataFrame to the table
    qdbpd.write_dataframe(df, qdbd_connection, table)

    return (df, table)
