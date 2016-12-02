import matplotlib
matplotlib.use('agg')

import tempfile
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import numpy as np
import scipy.misc
import shutil
import os

metadata = np.load('neon-racer/terrible.npy')

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

for m in metadata:
    actions = m['policy_dist']
    actions = np.maximum(actions - 0.5, 0)
#    actions = softmax(actions)
#    actions = actions**8
    actions = actions / np.sum(actions)
#    actions = (10 * (actions - 0.5)) / actions.shape[0]
    m['policy_dist'] = actions


tmpdir = tempfile.mkdtemp(prefix='a3c-visualize')

print(tmpdir)

def screen_from_metadatum(metadatum):
    return metadatum['state']

def array_from_figure(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    width, height = fig.canvas.get_width_height()
    return data.reshape((height, width, 3))

for index in range(1, len(metadata)-1):
    print('{0}/{1}'.format(index+1, len(metadata)))
    x = np.arange(max(0, index-50), index+1)

    def plot_rewards():
        rewards = [metadata[other_index]['scalars']['reward'] for other_index in x]
        plt.subplot(3, 1, 1)
        plt.plot(x, rewards)
        plt.xlim(index-50, index)
#        plt.axis([index-50, index, 0, 120])
        plt.ylabel('reward')
        ax = plt.gca()
        ax.set_autoscale_on(False)

    def plot_value_predictions():
        value_predictions = [metadata[other_index]['scalars']['value'] for other_index in x]
        plt.subplot(3, 1, 2)
        plt.plot(x, value_predictions)
        plt.xlim(index-50, index)
        plt.ylabel('value')

    racing_keys = ['left', 'right', 'up', 'down', 'x', 'n', 'space', 'z', 'a', 's', 'd', 'w']
    colors = ScalarMappable(cmap=plt.get_cmap('Paired')).to_rgba(np.linspace(0, 1, num=12))

    def plot_actions():
        actions = [metadata[other_index]['policy_dist'] for other_index in x]
        actions = np.array(actions).T
        plt.subplot(3, 1, 3)
        plt.stackplot(x, actions, labels=racing_keys, colors=colors)
        plt.legend(ncol=2, frameon=True, loc='center left')
        plt.axis([x[0], x[-1], 1, 0])
        ax = plt.gca()
        ax.set_autoscale_on(False)
        ax.get_yaxis().set_visible(False)

#    def plot_diagnostics_time_ranges(prefix, count):
#        lower_bounds = [
#            metadata[i]['scalars'].get('diagnostics/' + prefix + '_lb', 0)
#            for i in x
#        ]
#        upper_bounds = [
#            metadata[i]['scalars'].get('diagnostics/' + prefix + '_ub', 0)
#            for i in x
#        ]
#        plt.subplot(3, 2, count*2)
#        plt.fill_between(x, lower_bounds, upper_bounds)
#        plt.xlim(x[0], x[-1])
#        plt.ylabel(prefix)


    fig = plt.figure()
    plot_rewards()
    plot_value_predictions()
    plot_actions()
#    plot_diagnostics_time_ranges('observation_lag', 1)
#    plot_diagnostics_time_ranges('clock_skew', 2)
#    plot_diagnostics_time_ranges('action_lag', 3)

    image = array_from_figure(fig)
    plt.close(fig)

    # make room for screen pixels
    image = np.concatenate((255 * np.ones((440, 640, 3)), image), axis=1)

    metadatum = metadata[index+1]

    screen = screen_from_metadatum(metadatum) * 255
    zoom=3
    screen = np.repeat(screen, zoom, 0)
    screen = np.repeat(screen, zoom, 1)
    screen_top = (440 - 128*zoom)//2
    screen_left = (640 - 200*zoom)//2
    print(screen_top, screen_left, 128*zoom, 200*zoom)
    image[screen_top:128*zoom+screen_top, screen_left:200*zoom+screen_left] = screen
    imgpath = os.path.join(tmpdir, 'image-{0}.png'.format(index))
    scipy.misc.imsave(imgpath, image)

os.system("ffmpeg -f image2 -r 15 -i {0}/image-%d.png -vcodec mpeg4 -q:v 1 -y output.mp4".format(tmpdir))
shutil.rmtree(tmpdir)

