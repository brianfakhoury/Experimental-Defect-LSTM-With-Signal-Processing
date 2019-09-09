#!/usr/bin/env python3
import warnings
import tensorflow as tf
import matplotlib.pyplot as plt
from topo_defect_net import load_data
import os
import json
from topo_defect_net import interp_predict
import argparse
import matplotlib.style as style

style.use("fivethirtyeight")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

CONF = None
ARGS = None


def load_config(name):
    with open("./model_configs.json") as f:
        conf = type("conf", (object,), {})()
        setattr(conf, "VERSION", name)
        for key, value in json.load(f)[name].items():
            setattr(conf, key, value)
        x = {"CONF": conf}
        globals().update(x)


def load_data_and_model(name, column=None):
    if column is not None:
        load_config(name)
        data = load_data(name, column=column)
        return (
            data,
            tf.keras.models.load_model(
                "./models/DefectNet_V1_{}.h5".format(CONF.VERSION)
            ),
        )
    else:
        load_config(name)
        data = load_data(name)
        return (
            data,
            tf.keras.models.load_model(
                "./models/DefectNet_V1_{}.h5".format(CONF.VERSION)
            ),
        )


def o2o_predict(model, x_data, multiplier=1):
    """
    Use single week model to predict n number of next weeks.
    """
    pass
    # predictions = []
    # level = 0
    # temp = []
    # open_history = []
    # # TODO: fix window to predicted values instead of actual
    # for x in xdata:
    #     if level < multiplier:
    #         x_modified = x[0 : ((multiplier - level) * 7)] + open_history
    #         pred = model.predict(x_modified)
    #         temp.append(pred)
    #         level += 1
    #     else:
    #         pred = model.predict(x)
    #         level = 0
    #         predictions.append(temp)
    #         temp = []
    #         temp.append(pred)
    # return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="Interp(7)",
        help="Name of model in config",
    )

    parser.add_argument("--column", type=str, help="Override default column")

    parser.add_argument("--plot", type=str, default="full",
                        help="Set type of plot")

    parser.add_argument("--no_interp", action="store_true")

    ARGS = parser.parse_args()

    data, model = load_data_and_model(ARGS.model_name, column=ARGS.column)

    fig, ax = plt.subplots()

    if ARGS.no_interp:
        actual = data["Y"]
        order_vol_data = data["X"][:, 1, 1]
        # plot predictios for all time steps
        if ARGS.plot == "full":
            predicted_values = model.predict(
                data["X"][::CONF.OUTPUT_LEN]).flatten()
            pred_line, = ax.plot(
                range(
                    (len(actual) + CONF.OUTPUT_LEN) - len(predicted_values),
                    len(actual) + CONF.OUTPUT_LEN,
                ),
                predicted_values,
            )
        # only plot predictions for unseen data
        if ARGS.plot == "partial":
            predicted_values = model.predict(data["X_test"][::CONF.OUTPUT_LEN])
            x_begin = len(data["Y"][:: CONF.OUTPUT_LEN]) - \
                len(predicted_values) + CONF.OUTPUT_LEN
            x_end = len(data["Y"][:: CONF.OUTPUT_LEN]) + CONF.OUTPUT_LEN
            pred_line, = ax.plot(range(x_begin, x_end), predicted_values)
    else:
        actual = data["Y"][::7]
        order_vol_data = data["X"][:, 1, 1][::7]
        # plot predictios for all time steps
        if ARGS.plot == "full":
            predicted_values = interp_predict(model, data["X"])
            pred_line, = ax.plot(
                range(
                    (len(actual) + CONF.OUTPUT_LEN // 7) - len(predicted_values),
                    len(actual) + CONF.OUTPUT_LEN // 7,
                ),
                predicted_values,
            )
        # only plot predictions for unseen data
        if ARGS.plot == "partial":
            predicted_values = interp_predict(model, data["X_test"])
            x_begin = len(data["Y"][:: CONF.OUTPUT_LEN]) - \
                len(predicted_values) + CONF.OUTPUT_LEN // 7
            x_end = len(data["Y"][:: CONF.OUTPUT_LEN]) + CONF.OUTPUT_LEN // 7
            pred_line, = ax.plot(range(x_begin, x_end), predicted_values)

    actual_line, = ax.plot(actual)  # true data

    # plot order vol
    norm_factor = 10
    vol_line = ax.bar(
        list(range(0, len(order_vol_data))),
        (order_vol_data / norm_factor),
        color="white",
    )

    ax.set_xticklabels(data["Y"].index)
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.axvline(
        x=CONF.DATA_SPLIT_RATIO * len(data["Y"][:: CONF.OUTPUT_LEN]), color="black"
    )
    plt.gca().set_ylim(bottom=0)
    ticks = ax.xaxis.get_ticklabels()
    for n, label in enumerate(ticks):
        if n % (len(ticks) // 20) != 0:
            label.set_visible(False)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.title(
        "Single Factory Cumulative Defect Rate Prediction\n Model: {}".format(
            ARGS.model_name
        )
    )
    plt.legend(
        (actual_line, pred_line, vol_line), [
            "Actual", "Predicted", "Rel. Order Vol."]
    )
    plt.show()
