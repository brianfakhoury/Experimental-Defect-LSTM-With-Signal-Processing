#!/usr/bin/env python3
import warnings
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import json
from scipy.signal import savgol_filter
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

CONF = None
ARGS = None


def load_config(name):
    """
    Load the specific model configuration from the config json
    into a global object accessbile by CONF
    """
    with open("./model_configs.json") as f:
        conf = type("conf", (object,), {})()
        setattr(conf, "VERSION", name)
        for key, value in json.load(f)[name].items():
            setattr(conf, key, value)
        if ARGS is not None and ARGS.column is not None:
            setattr(conf, "DEFAULT_COLUMN", ARGS.column)
        x = {"CONF": conf}
        globals().update(x)
    print("[INFO] Success | Configuration loaded")


def process_output_tensor(data):
    """
    Takes 1D array of defect rates, and transforms into labels
    based off desired number of output steps
    """
    out = []
    for i in range(
        0, len(data) - CONF.OUTPUT_LEN + 1
    ):  # loop through data up to point where lookahead reaches end of array
        out.append(
            data[i: i + CONF.OUTPUT_LEN]
        )  # add desired number of timesteps to array and append
    print("[INFO] Success | Output Tensor Processed")
    return np.array(out)


def process_input_tensor(data):
    """
    Creates input feature array, 3 dimmensions for LSTM:
    1. Total array length is arbitrary based off total amount of training data.
    2. First inner array will be the size of the lookback (number of timesteps that the nn sees when called)
    3. Nested array in the first inner array will have length equal to number of features (e.g. 2 for defect rate and poqty)
    inputs match for timesteps leading up to timestep of the label, but not including x value for label timestep
    """
    out = []
    for i in range(
        CONF.LOOKBACK, len(data[0])
    ):  # starting at point where lookback reaches beginning of array, loop over all x values
        # note, len(data[0]) == len(data[1]) == len(data[n])
        t = []
        for j in range(
            CONF.LOOKBACK
        ):  # for each x now, we loop over every previous element and append previous x values
            t.append(
                [feature[i - CONF.LOOKBACK + j] for feature in data]
            )  # make sure for each x value, each feature is appended, x values can have 1,2,3...etc features
        out.append(t)
    print("[INFO] Success | Input Tensor Processed")
    return np.array(out)


def load_data(name, column=None):
    """
    Configures environment for model
    """
    load_config(name)  # load global configs
    column = column or CONF.DEFAULT_COLUMN

    data = [pd.read_csv(path, index_col=0)[column] for path in CONF.DATA_PATHS]
    Y = data[
        0
    ]  # first data path is assumed to contain label class, and will also be used as input

    X = process_input_tensor(data)
    Y = Y[
        CONF.LOOKBACK:
    ]  # the y data labels will now line up with processed input data, lining up at the lookback split
    y = process_output_tensor(Y)  # process that y data into required format

    # split data into two buckets for training and testing
    sample_len = len(X)
    split = int(CONF.DATA_SPLIT_RATIO * sample_len)

    X_train = X[:split]
    y_train = y[:split]

    X_test = X[split:]
    y_test = y[split:]
    print("[INFO] Success | Data Loaded and Sampled")
    return {
        "X": X,  # processed tensor
        "Y": Y,  # processed tensor
        "y": y,  # original labels
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


def create_model():

    # Instantiate a neural network
    model = Sequential(name=" Defect Net " + CONF.VERSION)

    if CONF.NUM_HIDDEN_LAYERS == 0:

        # Add the LSTM layer for time series processing
        model.add(
            LSTM(
                units=CONF.NUM_UNITS,
                input_shape=[CONF.LOOKBACK, CONF.NUM_INPUT_CLASSES],
            )
        )

    else:

        model.add(
            LSTM(
                units=CONF.NUM_UNITS,
                return_sequences=True,
                input_shape=[CONF.LOOKBACK, CONF.NUM_INPUT_CLASSES],
            )
        )

    # add dropout to prevent overfitting
    if CONF.DROPOUT:

        model.add(Dropout(CONF.DROPOUT_RATE))

    # Deep LSTM layers are implemented here, but are experimental for the data
    for i in range(CONF.NUM_HIDDEN_LAYERS):

        model.add(
            LSTM(units=CONF.NUM_UNITS, return_sequences=i <
                 CONF.NUM_HIDDEN_LAYERS - 1)
        )

        if CONF.DROPOUT:

            model.add(Dropout(CONF.DROPOUT_RATE))

    # Add standard feedforward layer to interpret output of LSTM processing
    model.add(Dense(units=CONF.NUM_UNITS / 2))

    # Add another feedfroward layer to output desired number of classes
    model.add(Dense(units=CONF.OUTPUT_LEN))

    # prints a summary to stdout
    model.summary()

    # finalizes the model with standard optimzer and loss calculator
    model.compile(optimizer="adam", loss="mean_squared_error")

    return model


def interp_predict(model, xdata):
    """
    Predicts next week using interpolation model.
    Skip slices input data, applies smoothing filter, then skip slices again to transform days -> weeks
    """
    predicted_values = model.predict(
        xdata[:: CONF.OUTPUT_LEN]
    ).flatten()  # predict next 7 days for beginning of each week (sundays)
    smoothed = savgol_filter(predicted_values, 15, 2)  # apply Savitzgy filter
    selected_labels = smoothed[
        6::7
    ]  # select one of the days from each week of predictions for predicted value
    return selected_labels


def new_run_log_dir():
    """Create new dir for tf run"""
    log_dir = "./runs"
    run_id = len([name for name in os.listdir(log_dir)])
    run_log_dir = os.path.join(log_dir, str(run_id))
    return run_log_dir


def main():
    """
    Loads data and configs, builds model, trains model, evaluates and serializes.
    """
    data = load_data(ARGS.model_name)

    model = create_model()

    model.fit(
        data["X_train"],
        data["y_train"],
        verbose=2,
        batch_size=64,
        epochs=CONF.TRAINING_EPOCHS,
        callbacks=[tf.keras.callbacks.TensorBoard(log_dir=new_run_log_dir())],
    )

    results = model.evaluate(
        data["X_test"][: -CONF.OUTPUT_LEN + 1], data["y_test"])

    model.save("./models/{}_V1_{}.h5".format("DefectNet", CONF.VERSION))

    print("[INFO] Success | Model Trained and Saved: ", CONF.VERSION)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="Interp(7)",
        help="Name of model in config",
    )

    parser.add_argument("--column", type=str, help="Override default column")

    ARGS = parser.parse_args()

    main()
