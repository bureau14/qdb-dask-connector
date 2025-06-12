"""
NOTES:
* Use persist before dealing with actual values, e.g. `len`, filtering values.
* Try to delay the persist call as much as possible to have bigger graphs.
* Use persist before complex operations (e.g. merge asof)
* TODO: Is persist scope aware? e.g. having evrything in a single function will improve performance / reduce persist calls?
"""

import pandas as pd
import quasardb
import quasardb_dask as qdbdask
from dask.distributed import Client
from dask.distributed import LocalCluster, Client

QDB_URI = "qdb://127.0.0.1:2836"
QDB_USER_SECURITY_FILE = ""
QDB_CLUSTER_PUBLIC_KEY_FILE = ""

DATA_START_DATE = "2024-12-24"
DATA_END_DATE = "+90d"
DATA_STEP = "1min"
DATA_N_TRAINING_MINUTES = 1000
DATA_CALC_MEDIAN = True

DATA_RUN_TAG = "GBB-CONV-R380-RWDR.Run Status Calc"
DATA_GRADE_TAG = "GBB-CONV-R380-RWDR.Product Code"
DATA_PROCESS_TAG = "GBB-CONV-R380-RWDR.Helical Roll 1 Amp.Rolling Max"
# DATA_PROCESS_TAG = "ARC-CC251_0146.MEAS"
# DATA_RUN_TAG = "ARC-SCRN7.Run Status Calc" # "ARC-SCRN7.Run Status Calc" # "ARC-IT3166_0146.PNT"
# DATA_GRADE_TAG = None


def get_quasardb_data(
    tag,
    columns,
    start_date,
    end_date,
    step,
    interpolate_method,
    lookback,
    additional_condition="",
):
    query = (
        f"select {', '.join(columns)} "
        f"from find(prefix='process_data' and tag='{tag}') "
        f"asof join range({start_date}, {end_date}, {step}) interpolate(method={interpolate_method}) with lookback {lookback} "
        f"prewhere unique_tagname in find(prefix='process_data' and tag='{tag}') {additional_condition} "
        "order by $timestamp"
    )
    # print(query)
    df = qdbdask.query(query, cluster_uri=QDB_URI)
    return df


def get_run_status_data(tag):
    return get_quasardb_data(
        tag,
        ["$timestamp", "numericvalue as run_value"],
        DATA_START_DATE,
        DATA_END_DATE,
        DATA_STEP,
        interpolate_method="constant",
        lookback="7d",
    )


def get_good_data(
    tag, bad_values, value_col_name, numeric=True, interpolate_method="linear"
):
    bad_values = ", ".join(f"'{v}'" for v in bad_values)
    return get_quasardb_data(
        tag,
        [
            "$timestamp",
            f"{'numericvalue' if numeric else 'stringvalue'} as {value_col_name}",
        ],
        DATA_START_DATE,
        DATA_END_DATE,
        DATA_STEP,
        interpolate_method=interpolate_method,
        lookback="7d",
        additional_condition=f"and stringvalue not in ({bad_values})",
    )


def get_event_filters(tag):
    return 1, 1


def get_known_bad_values():
    return [
        "Good-Off",
        "Bad Lab Data",
        "Over UCL#",
        "BadQ-Alrm-On",
        "\tNo Data\t",
        "Trend Up#",
        "GreaterMM",
        "No Result",
        "Under Range",
        "BadQ-Alrm-Of",
        "Overflow_st",
        "\tnan\t",
        "Trend Down",
        "ActiveBatch",
        "Under WL",
        "Over Center#",
        "CO Bypassed",
        "Lo Alarm/Ack",
        "Bad Narg",
        "Unit Down",
        "No_Sample",
        "Trend Up",
        "\t0\t",
        "BadQ-On",
        "Not Connect",
        "\tIntf Shut\t",
        "Bad_Quality",
        "Sample Bad",
        "No Alarm#",
        "Rate Alm/Ack",
        "Under Centr#",
        "Dig Alarm",
        "Over Center",
        "Arc Off-line",
        "No_Alarm",
        "\tO\t",
        "Good-On",
        "Rate Alarm",
        "Wrong Type",
        "No Lab Data",
        "\tScan Off\t",
        "Bad Input",
        "ISU Saw No Data",
        "Error",
        "Equip Fail",
        "\tZERO\t",
        "Scan Off",
        "\tError\t",
        "snapfix",
        "I/O Timeout",
        "Calc Overflw",
        "Good",
        "Execute",
        "Substituted",
        "Over 1Sigma#",
        "DST Forward",
        "Calc Crash",
        "Under 1Sigma",
        "Set to Bad",
        "Bad Quality",
        "Intf Shut",
        "Bad Output",
        "\tConfigure\t",
        "Out of Serv",
        "Not Converge",
        "No Data",
        "\t0.0\t",
        "Pt Created",
        "Invalid Data",
        "Future Data Unsupported",
        "Scan On",
        "NoAlrm/UnAck",
        "Under Center",
        "Bad Total",
        "High Alarm",
        "Over WL#",
        "Calc Off",
        "Over WL",
        "\tShutdown\t",
        "Failed",
        "Stratified#",
        "Under WL#",
        "AccessDenied",
        "Doubtful",
        "Alarm-On",
        "\tComm Fail\t",
        "Over Range",
        "Dig Alm/Ack",
        "Under LCL",
        "Bad",
        "Mixture#",
        "Filtered",
        "Stratified",
        "Shutdown",
        "Low Alarm",
        "Trend Down#",
        "Hi Alarm/Ack",
        "Scan Timeout",
        "Bad Data",
        "\tLoading...\t",
        "DST Back",
        "\tI/O Timeout\t",
        "Inp OutRange",
        "Calc Timeout",
        "Under LCL#",
        "Coercion Failed",
        "Alarm-Off",
        "Over UCL",
        "\tPt Created\t",
        "Over 1 Sigma",
        "Calc Failed",
        "Mixture",
        "No Sample",
        "Invalid Float",
        "Comm Fail",
        "DCS failed",
        "Configure",
        "\tScan Timeout\t",
        "Under 1Sigm#",
        "Trace",
        "No Alarm",
    ]


def get_run_status_df(run_status_tag):
    """
    Get time intervals when the RUN tag is active.
    """
    # 1. Get raw run-status data
    df = get_run_status_data(run_status_tag)

    # 2. Cast the values properly (float -> int)
    df["run_value"] = df["run_value"].apply(lambda x: 0 if x < 0.5 else 1)

    # 3. Calculate the group id / rank / count
    df["grp"] = ((df["run_value"] != 1) | (df["run_value"].shift() != 1)).cumsum()
    df["rank"] = df.groupby("grp").cumcount()
    df["count"] = df.groupby("grp")["run_value"].transform(
        "count", meta=("count", "int64")
    )

    # 4. Remove mins after the machine starts and before the machine shutdowns
    training_pre_event_filter, training_post_event_filter = get_event_filters(
        run_status_tag
    )
    mask = (
        (df["run_value"] == 1)
        & (df["rank"] >= training_pre_event_filter)
        & (df["rank"] < df["count"] - training_post_event_filter)
    )
    df["run_value"] = df["run_value"].where(mask, 0)

    # 5. Return the proper columns
    df = df[["run_value"]]

    return df


def get_process_df(process_tag, grade_tag=None, run_tag=None):
    """
    Get process and grade dataframe.
    """

    # 1. Get raw good process/grade data
    df = get_good_data(process_tag, get_known_bad_values(), "process_value")

    # 2. Merge both process and grade dataframes
    columns = ["process_value"]
    if grade_tag:
        grade_df = get_good_data(
            grade_tag, get_known_bad_values(), "grade_value", False, "constant"
        )
        df["grade_value"] = grade_df["grade_value"]
        columns.append("grade_value")

    # 3. Filter data by active runs
    if run_tag:
        run_df = get_run_status_df(run_tag)
        df["run_value"] = run_df["run_value"]
        df = df[df["run_value"] == 1]

    # 4. Return the proper columns
    df = df[columns]
    return df


def get_clean_process_df(process_df):
    """
    Get clean data based on thresholds.
    """

    # 1. Get % of training minutes
    process_df = process_df.persist()  # needed for `len` and bounds
    if len(process_df) / DATA_N_TRAINING_MINUTES < 0.8:
        print("The number of minutes is less than the threshold. Exiting")
        return None

    # 2. Get Quantiles
    Q1 = process_df["process_value"].quantile(0.25)
    Q3 = process_df["process_value"].quantile(0.75)
    IQR = Q3 - Q1

    # 3. Get lower/upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 4. Get clean data
    df = process_df[
        (process_df["process_value"] >= lower_bound)
        & (process_df["process_value"] <= upper_bound)
    ]

    return df


def get_calculations(cleaned_df, calc_median):
    """
    Perform the actual calculations.
    """

    # 1. Get aggregations
    agg_list = ["std"]
    agg_list.append("median" if calc_median else "mean")

    # 2. Get mean, std and other stats
    if "grade_value" in cleaned_df.columns:
        cleaned_df["grade_value"] = cleaned_df["grade_value"].fillna("missing")
        grouped = True
        stats = (
            cleaned_df.groupby("grade_value")["process_value"].agg(agg_list).compute()
        )
        stats = stats["process_value"].to_dict(orient="index")
    else:
        grouped = False
        stats = {
            agg: float(getattr(cleaned_df["process_value"], agg)().compute())
            for agg in agg_list
        }

    result = {"grouped": grouped, "results": stats}
    return result


def main():
    process_df = get_process_df(DATA_PROCESS_TAG, DATA_GRADE_TAG, DATA_RUN_TAG)
    cleaned_df = get_clean_process_df(process_df)
    results = get_calculations(cleaned_df, DATA_CALC_MEDIAN)
    print("Results:", results)


if __name__ == "__main__":
    with LocalCluster(n_workers=2) as cluster:
        with Client(cluster) as client:
            main()
