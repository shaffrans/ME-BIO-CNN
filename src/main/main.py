import pandas as pd

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
    print(numbers[1], expected[1], window_end[1], responses[1], window_data[1], sep="\n")


if __name__ == "__main__":
    get_windows()
