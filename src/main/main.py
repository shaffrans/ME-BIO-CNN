import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf


def load_data(med=False):
    filename_bio = "../../data/med.gaze.csv"
    filename_meta = "../../data/med_SART_2022-04-08.csv"
    if not med:
        filename_bio = "../../data/unmed.gaze.csv"
        filename_meta = "../../data/unmed_SART_2022-04-08.csv"
    data_bio = pd.read_csv(filename_bio)
    data_meta = pd.read_csv(filename_meta)
    return data_bio, data_meta


def get_windows():
    data_bio, data_meta = load_data(med=True)
    numbers = []
    expected = []
    window_end = []
    responses = []
    results = []
    window_data = []
    bad_columns = list(range(6)) + [17] + list(range(42, 52))
    bio_values = data_bio.values
    times = bio_values[:, 3]
    current_row = 0
    for row in data_meta.values:
        numbers.append(int(row[0]))
        expected.append(int(row[1] == "space"))
        window_end.append(float(row[8]))
        responses.append(int(row[10] == "space"))
        results.append(int(row[11]))
        for i in range(current_row, len(times)):
            if times[i] > window_end[-1]:
                current_row = i
                break
        window_data.append(bio_values[current_row - 105: current_row, :])
    window_data = np.delete(np.array(window_data), bad_columns, 2)
    print(numbers[1], expected[1], window_end[1], responses[1], results[1], window_data[1], window_data.shape, sep="\n")
    return numbers, responses, np.array(window_data, dtype=float)


def trim_data(numbers, responses, data):
    print(numbers, responses, data[0], sep='\n')
    numbers_new, responses_new, data_new = [], [], []
    for row in zip(numbers, responses, data):
        if (row[0] != 3 and random.random() < 0.1) or row[0] == 3:
            numbers_new.append(row[0])
            responses_new.append(row[1])
            data_new.append(row[2])
    data_new = np.array(data_new)
    print(numbers_new, responses_new, data_new[0], sep='\n')
    return numbers_new, responses_new, data_new


def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(35, 3, strides=2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv1D(25, 3, strides=2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(50, activation="relu"),
        tf.keras.layers.Dense(25, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    return model


if __name__ == "__main__":
    numbers, responses, X = trim_data(*get_windows())
    # numbers, responses, X = get_windows()
    model = get_model()
    y = np.array([int(n != 3 or r == 0) for n, r in zip(numbers, responses)])
    print(list(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model.compile(optimizer="adam", loss=tf.losses.BinaryCrossentropy(), metrics=["accuracy"])
    model.fit(X_train, y_train)
    model.summary()
    y_pred = model.predict(X_test)
    print(y_pred, y_test, sep='\n')
    print(model.test_on_batch(X_test, y_test))
