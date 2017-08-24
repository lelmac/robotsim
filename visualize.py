import argparse
import json
import time
import numpy
import matplotlib.pyplot as plt


def smooth(x, window_len=11, window='flat'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s = numpy.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')

    y = numpy.convolve(w / w.sum(), s, mode='valid')
    return y


def visualize_log(filename, figsize=None, output=None):
    with open(filename, 'r') as f:
        data = json.load(f)
    if 'episode' not in data:
        raise ValueError(
            'Log file "{}" does not contain the "episode" key.'.format(filename))
    episodes = data['episode']

    # Get value keys. The x axis is shared and is the number of episodes.
    keys = sorted(list(set(data.keys()).difference(set(['episode']))))
    keys = [keys[1], keys[2], keys[3]]
    if figsize is None:
        figsize = (15., 5. * len(keys))
    f, axarr = plt.subplots(len(keys), sharex=True, figsize=figsize)
    for idx, key in enumerate(keys):
        #date = numpy.array(data[key])
        date = smooth(numpy.array(data[key]))
        axarr[idx].plot(episodes, data[key])

        if key != 'mean_q':
            axarr[idx].plot(range(len(date)), date)
        axarr[idx].legend(['Original Data', 'Smoothed'], loc='upper left')
        axarr[idx].set_ylabel(key)
        plt.xlabel('episodes')
        plt.tight_layout()
    if output is None:
        plt.savefig('./diagrams/ddpg_circle.pdf')
        plt.show()
    else:
        plt.savefig(output)


parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str,
                    help='The filename of the JSON log generated during training.')
parser.add_argument('--output', type=str, default=None,
                    help='The output file. If not specified, the log will only be displayed.')
parser.add_argument('--figsize', nargs=2, type=float, default=None,
                    help='The size of the figure in `width height` format specified in points.')
args = parser.parse_args()

# You can use visualize_log to easily view the stats that were recorded during training. Simply
# provide the filename of the `FileLogger` that was used in `FileLogger`.
visualize_log(args.filename, output=args.output, figsize=args.figsize)
