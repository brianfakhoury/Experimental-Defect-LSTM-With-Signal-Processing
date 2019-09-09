#!/usr/bin/env python3
import pandas as pd
import pyodbc
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

sns.set(style="darkgrid")

CONN = pyodbc.connect(
    None
)


def weeks_to_index(weeks):
    """
    Takes list of dataframes and pulls first sunday out of each one to create a week index
    """
    oneweek = datetime.timedelta(days=7)
    first = trunc_datetime(weeks[0]["Date"].min())
    last = trunc_datetime(weeks[-1]["Date"].min())
    curr = first
    out = []
    while curr <= last:
        out.append(curr)
        curr = curr + oneweek
    return out


def trunc_datetime(someDate):
    """
    Converts date to sunday of that same week for indexing purpose
    """
    return someDate - datetime.timedelta(days=someDate.isoweekday() % 7)


def save_df(df, name=""):
    df.to_csv(
        "./data/{}_{}.csv".format(
            name, datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        )
    )
    print("[INFO] Success | Save DataFrame to CSV file: ", name)


def reduce_to_supervised_redact(df):
    """Return a dataframe of dfr,orderqty for each week for each factory"""
    factories = df["Factory"].unique()
    week_dfs = [g for n, g in df.groupby(pd.Grouper(key="Date", freq="W"))]
    factory_dict = {
        factory: [[0, 0] for _ in range(len(week_dfs))] for factory in factories
    }
    factory_insp_total_size = {
        factory: [0] * len(week_dfs) for factory in factories}
    weeks_index = weeks_to_index(week_dfs)
    max_poqty = {factory: 0 for factory in factories}
    for index, row in df.iterrows():
        factory = row["Factory"]
        poqty = int(row["POQty"])
        sample_size = int(row["SampleSize"])
        num_defects = int(row["WmsMajor"]) + int(row["WmsMinor"])
        week = weeks_index.index(trunc_datetime(row["Date"]))
        num_defects_prev = (
            factory_dict[factory][week][0] *
            factory_insp_total_size[factory][week]
        )
        factory_insp_total_size[factory][week] += sample_size
        factory_dict[factory][week][0] = (
            num_defects_prev + num_defects
        ) / factory_insp_total_size[factory][week]
        factory_dict[factory][week][1] += poqty
        if factory_dict[factory][week][1] > max_poqty[factory]:
            max_poqty[factory] = factory_dict[factory][week][1]
    for factory in factory_dict:
        for week in range(len(week_dfs)):
            factory_dict[factory][week][1] /= max_poqty[factory]
    ret = pd.DataFrame(factory_dict)
    ret.index = weeks_index
    return ret


def reduce_to_supervised_redact(df):
    """Return a dataframe of dfr,orderqty for each week for each factory"""
    factories = df[
        "Place"
    ].unique()  # Start with a list of all the factories in the client dataset
    week_dfs = [
        g for n, g in df.groupby(pd.Grouper(key="Date", freq="W"))
    ]  # This pandas function then splits this single dataframe into a list of dataframes by week
    factory_dict = {  # now we create the empty dataframe strcutre to hold all the factories and associated values
        factory: [[0, 0] for _ in range(len(week_dfs))]
        # the structure is columns are factories, rows are dates
        for factory in factories
    }
    factory_insp_total_size = {
        factory: [0] * len(week_dfs) for factory in factories
    }  # make a temporary store for number of inspections to calculate dfr
    weeks_index = weeks_to_index(
        week_dfs
    )  # call the indexing function to extract all the sundays from the dataset and put them into a list to find the week's index
    max_poqty = {
        factory: 0 for factory in factories
    }  # temporarily store the largest poqty seen at each factory to normalize the poqty's at the end
    for index, row in df.iterrows():  # begin looping over all inspection reports
        factory = row["Place"]
        poqty = int(row["TotalQuantity"])
        sample_size = int(row["TotalSampleSize"])
        dfr = float(row["DefectRate"]) / 100  # convert dfr to float number
        num_defects = (
            dfr * sample_size
        )  # calculate the number of defects found in this inspection
        week = weeks_index.index(
            trunc_datetime(row["Date"])
        )  # find which row this data will go into
        num_defects_prev = (
            factory_dict[factory][week][0] *
            factory_insp_total_size[factory][week]
        )
        factory_insp_total_size[factory][
            week
        ] += sample_size  # incremenent inspection counter
        factory_dict[factory][week][0] = (  # update dfr entry for given week/factory
            num_defects_prev + num_defects
        ) / factory_insp_total_size[factory][week]
        # update poqty for week in the factory
        factory_dict[factory][week][1] += poqty
        if (
            factory_dict[factory][week][1] > max_poqty[factory]
        ):  # check if poqty is largest seen
            max_poqty[factory] = factory_dict[factory][week][1]
    for factory in factory_dict:  # main processing finished
        for week in range(len(week_dfs)):
            factory_dict[factory][week][1] /= max_poqty[
                factory
            ]  # normalize poqty [0..1]
    ret = pd.DataFrame(factory_dict)
    ret.index = weeks_index  # set the index to be dates instead of ints
    return ret


def create_joined_format(report, defect_table, mapping):
    defects = pd.merge(mapping, defect_table, how="inner", left_on="PK", right_on="FK")[
        ["FK_x", "Code_x"]
    ]
    defects.columns = ["FK", "Code"]
    return pd.merge(report, defects, how="outer", left_on="PK", right_on="FK")


def load_dataframe(table_name, date_col="Date"):
    df = pd.read_sql(
        "SELECT * FROM {};".format(table_name), CONN, parse_dates=[date_col]
    )
    df.rename(columns={date_col: "Date"}, inplace=True)
    print("[INFO] Success | Load DataDrame: ", table_name)
    return df


def select_def(df, i):
    arr = []
    for index, row in df.iterrows():
        new_row = [l[i] for l in row]
        arr.append(new_row)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)


def process_dataframe(client_name, interpolate=True):
    if ARGS.client_name == "redact":
        report = load_dataframe(
            "[redact]")
        df_clean = report[
            [
                "formID",
                "Place",
                "DefectRate",
                "Date",
                "TotalQuantity",
                "TotalSampleSize",
            ]
        ].dropna()
        df_out = reduce_to_supervised_redact(df_clean)
        print("[INFO] Success | Transform DataFrame to Supervised Dataset")
        a = select_def(df_out, 0)
        b = select_def(df_out, 1)
        if ARGS.no_interpolate:
            save_df(a, "redact-dfr")
            save_df(b, "redact-poqty")

        else:
            upsampled_a = a.resample("D")
            interpolated_a = upsampled_a.interpolate(method="quadratic")
            upsampled_b = b.resample("D")
            interpolated_b = upsampled_b.interpolate(method="quadratic")
            save_df(interpolated_a, "redact-dfr-interp")
            save_df(interpolated_b, "redact-poqty-interp")

    if ARGS.client_name == "puma-footwear":
        report = load_dataframe(
            "[redact]", date_col="InspectionDate"
        )
        df_clean = report[
            ["formID", "Factory", "Date", "POQty",
                "SampleSize", "WmsMajor", "WmsMinor"]
        ].dropna()
        df_out = reduce_to_supervised_redact(df_clean)
        print("[INFO] Success | Transform DataFrame to Supervised Dataset")
        a = select_def(df_out, 0)
        b = select_def(df_out, 1)
        if ARGS.no_interpolate:
            save_df(a, "puma-footwear-dfr")
            save_df(b, "puma-footwear-poqty")

        else:
            upsampled_a = a.resample("D")
            interpolated_a = upsampled_a.interpolate(method="quadratic")
            upsampled_b = b.resample("D")
            interpolated_b = upsampled_b.interpolate(method="quadratic")
            save_df(interpolated_a, "redact-dfr-interp")
            save_df(interpolated_b, "redact-poqty-interp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--client_name", type=str, default="redact")

    parser.add_argument("--no_interpolate", action="store_true")

    ARGS = parser.parse_args()

    process_dataframe(ARGS.client_name, interpolate=(not ARGS.no_interpolate))
