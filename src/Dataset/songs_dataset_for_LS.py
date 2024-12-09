import numpy as np
import pandas as pd
import time

from analysis.commons import compute_xopt, get_w, twoNorm, data_normalize_by_features


def load_song_dataset_matrices_for_LS(file_dataset_name="./Dataset/YearPredictionMSD.txt"):
    df = pd.read_csv(file_dataset_name, header=None)
    X = df.drop([0], axis=1)
    X['Constant'] = 1
    X.reset_index(drop=True, inplace=True)
    y = df[0]

    return np.array(X), np.array(y)


if __name__ == "__main__":
    file_dataset_name = "./YearPredictionMSD.txt"

    tic = time.perf_counter()
    B, b = load_song_dataset_matrices_for_LS(file_dataset_name)
    toc = time.perf_counter()
    print(f"generate song dataset in {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    xopt = compute_xopt(B, b).reshape((-1, 1))
    toc = time.perf_counter()
    print(xopt)
    print(f"Compute exact solutions needs {toc - tic:0.4f} seconds")

    tic = time.perf_counter()
    print(twoNorm(get_w(B, b)))
    toc = time.perf_counter()
    print(f"Compute error needs {toc - tic:0.4f} seconds")
