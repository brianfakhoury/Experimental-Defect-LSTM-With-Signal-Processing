#!/usr/bin/env python3
import numpy as np
import tensorflow as tf
import pandas as pd
from analyze_model import load_data_and_model
from topo_defect_net import interp_predict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


if __name__ == "__main__":
    data, model = load_data_and_model("Interp(7)")
    predicted_value = interp_predict(model, data["X"])[-1]
    date = (data["Y"].index)[-1]
    print(
        "Predicted Defect Rate for Delsey FTY for week of {} is: {}%".format(
            (datetime.strptime(date, "%Y-%m-%d") + timedelta(days=7)).strftime(
                "%B %d, %Y"
            ),
            round(predicted_value * 100, 3),
        )
    )

if False:
    test_df = pd.read_csv(
        "./data.csv", index_col=0
    )

    for fac in list(test_df)[:10]:
        plt.figure()
        print(fac)
        plt.plot(test_df[fac])
        plt.show()
