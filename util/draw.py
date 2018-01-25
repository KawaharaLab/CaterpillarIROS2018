from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import re
from optparse import OptionParser
import sys
import os


def plot_and_save(csv_data_path, save_file_path=None, vmin=None, vmax=None, notitle=False):
    df = pd.read_csv(csv_data_path, header=0)
    cols = df.columns.values.tolist()

    df = df[list(filter(lambda x: x != cols[0], cols))]
    y_range = (vmin, vmax) if vmin is not None and vmax is not None else None
    if notitle:
        ax = df.plot(ylim=y_range)
    else:
        title = re.search(r'/([^/]+?)\.((csv)|(txt))$', csv_data_path).group(1)
        ax = df.plot(title=title, ylim=y_range)
    ax.set_ylabel(cols[1])
    ax.set_xlabel(cols[0])

    if len(cols) <= 2:
        ax.legend().set_visible(False)

    # tics = 500 if df.shape[0] > 1000 else 100
    tics = df.shape[0] // 10
    x_major_ticks = np.arange(0, df.shape[0], tics)
    ax.set_xticks(x_major_ticks)
    if tics // 5 > 0:
        x_minor_ticks = np.arange(0, df.shape[0], tics // 5)
        ax.set_xticks(x_minor_ticks, minor=True)

    ax.grid(which='both')
    ax.grid(which='major', alpha=0.4)
    ax.grid(which='minor', alpha=0.1)

    if save_file_path is None:
        save_file_path = re.sub(r'\.((csv)|(txt))$', '.png', csv_data_path)
    plt.savefig(save_file_path)


def plot_files(dir_path: str):
    """
        Plot csv files (.csv or .txt)
    """
    dir_path = re.sub(r'(?<=[^/])/$', '', dir_path)
    for file_name in os.listdir(dir_path):
        if re.match(r'[^.]+\.((csv)|(txt))', file_name):
            file_path = "{}/{}".format(dir_path, file_name)
            plot_and_save(file_path)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-d", action="store", default=None, dest="dir_path")
    parser.add_option("-o", action="store", default=None, dest="save_file_path")
    parser.add_option("--vmin", action="store", default=None, dest="vmin", type="float")
    parser.add_option("--vmax", action="store", default=None, dest="vmax", type="float")
    parser.add_option("--no_title", action="store_true", default=False, dest="no_title")
    opts, args = parser.parse_args()

    if opts.dir_path is not None:
        plot_files(opts.dir_path)
    else:
        csv_data_path = args[0]
        save_file_path = opts.save_file_path
        plot_and_save(csv_data_path, save_file_path, vmin=opts.vmin, vmax=opts.vmax, notitle=opts.no_title)
