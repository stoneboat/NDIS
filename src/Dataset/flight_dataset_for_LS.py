import time

from nycflights13 import flights

import pandas as pd
import numpy as np

from analysis.commons import compute_xopt, get_w, twoNorm, data_normalize_by_features


def save_flight_dataset_matrices_for_LS(file_X_name="./Dataset/flight-LR-X.txt",
                                        file_y_name="./Dataset/flight-LR-y.txt"):
    X = pd.DataFrame(flights['dep_delay'])
    X['Constant'] = 1
    X['arr_delay'] = flights['arr_delay']
    X = X.dropna(axis=0, how='any')

    y = X['arr_delay']
    X = X.drop(columns=['arr_delay'])
    X.reset_index(drop=True, inplace=True)

    np.savetxt(file_y_name, y, delimiter=',')
    np.savetxt(file_X_name, X.to_numpy(), delimiter=',')


def load_flight_dataset_matrices_for_LS(file_X_name="./Dataset/flight-LR-X.txt",
                                        file_y_name="./Dataset/flight-LR-y.txt"):
    B = np.loadtxt(file_X_name, delimiter=',')  # B is an array
    b = np.loadtxt(file_y_name, delimiter=',')  # b is an array
    return B, b


if __name__ == "__main__":
    file_X_name = "./flight-LR-X.txt"
    file_y_name = "./flight-LR-y.txt"
    save_flight_dataset_matrices_for_LS(file_X_name, file_y_name)

    tic = time.perf_counter()
    try:
        B, b = load_flight_dataset_matrices_for_LS(file_X_name, file_y_name)
    except:
        save_flight_dataset_matrices_for_LS(file_X_name, file_y_name)
        B, b = load_flight_dataset_matrices_for_LS(file_X_name, file_y_name)
    toc = time.perf_counter()
    print(f"generate flight dataset in {toc - tic:0.4f} seconds")

    B, b = data_normalize_by_features(B, b)
    xopt = compute_xopt(B, b).reshape((-1, 1))
    print(xopt)

    print(twoNorm(get_w(B, b)))