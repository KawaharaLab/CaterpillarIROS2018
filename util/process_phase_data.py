import os
import sys
import numpy as np
import pandas as pd

USAGE = "Usage: python process_phase_data.py <phase_data_file_path>"


def mod2pi(r: np.array) -> np.array:
    return r % (2*np.pi)


if __name__ == '__main__':
    args = sys.argv[1:]
    assert len(args) > 0, USAGE
    phase_data_file = args[0]

    df = pd.read_csv(phase_data_file)
    data = {}
    for c in df.columns[1:]:
        data[c] = mod2pi(df[c])

    file_name = os.path.basename(phase_data_file).split('.')[0]
    pd.DataFrame(data, index=df[df.columns[0]]).to_csv(os.path.join(os.path.dirname(phase_data_file), "{}_processed.txt".format(file_name)))
