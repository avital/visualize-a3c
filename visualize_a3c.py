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

metadata = np.load('Slither-2.npy')

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

for m in metadata:
    actions = m['policy_dist']
#    actions = softmax(actions)
#    actions = actions**2
    actions = actions / np.sum(actions)
    m['policy_dist'] = actions


tmpdir = tempfile.mkdtemp(prefix='a3c-visualize')

print(tmpdir)

def screen_from_metadatum(metadatum):
    return metadatum['state']

def array_from_figure(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8, sep='')
    width, height = fig.canvas.get_width_height()
    return data.reshape((height, width, 4))

#metadata = metadata[0:4]
for index in range(2, len(metadata)):
    print('{0}/{1}'.format(index+1, len(metadata)))
    history_size = 1
    x = np.arange(max(0, index-history_size), index)

    def plot_rewards():
        rewards = [metadata[other_index]['scalars']['reward'] for other_index in x]
        plt.subplot(3, 1, 1)
        plt.plot(x, rewards)
        plt.axis([x[0], x[-1], 0, 4])
#        plt.xlim(x[0], x[-1])
        plt.ylabel('reward')
        ax = plt.gca()
        ax.set_autoscale_on(False)

    def plot_value_predictions():
        value_predictions = [metadata[other_index]['scalars']['value'] for other_index in x]
        plt.subplot(3, 1, 2)
        plt.plot(x, value_predictions)
        plt.axis([x[0], x[-1], -0.4, 0.4])
#        plt.xlim(x[0], x[-1])
        plt.ylabel('value estimate')

    colors = ScalarMappable(cmap=plt.get_cmap('hsv')).to_rgba(np.linspace(0, 1, num=36))

    def plot_actions():
        actions = [metadata[other_index]['policy_dist'] for other_index in x]
        actions = np.array(actions).T
        axes = plt.subplot(1, 1, 1, polar=True, axisbg='none')
        R, theta = np.meshgrid(1 / np.linspace(x.shape[0], 2, num=x.shape[0]+1), np.pi - np.linspace(0, 2*np.pi, num=37) + np.pi/36)
        plt.ylim(0, 1)
        axes.spines['polar'].set_visible(False)
        plt.xticks([])
        plt.yticks([])
        plt.pcolormesh(theta, R, actions, cmap='gray', alpha=0.3)
#        plt.stackplot(x, actions, labels=slither_keys, colors=colors)
#        plt.legend(ncol=3, fontsize='xx-small', borderpad=0.2, labelspacing=0.2, handletextpad=0.65, borderaxespad=0.2, columnspacing=0.2, frameon=True, loc='center left')
#        plt.axis([x[0], x[-1], 1, 0])
#        ax = plt.gca()
#        ax.set_autoscale_on(False)
#        ax.get_yaxis().set_visible(False)
#        plt.ylabel('snake direction')

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


    metadatum = metadata[index]
    screen = screen_from_metadatum(metadatum)
#    screen = np.expand_dims(screen, axis=-1)
#    screen = np.repeat(screen, 3, axis=-1)
    zoom=2
    screen = np.repeat(screen, zoom, 0)
    screen = np.repeat(screen, zoom, 1)

    fig = plt.figure(figsize=screen.shape[:2][::-1], dpi=1)
    fig.patch.set_facecolor('none')
#    plot_rewards()
#    plot_value_predictions()
    plot_actions()
#    plot_diagnostics_time_ranges('observation_lag', 1)
#    plot_diagnostics_time_ranges('clock_skew', 2)
#    plot_diagnostics_time_ranges('action_lag', 3)

    overlay = array_from_figure(fig)
    plt.close(fig)

    alpha = overlay[:,:,1] / 255
    alpha = alpha * overlay[:,:,0] / 255
    alpha = np.expand_dims(alpha, axis=-1)
    overlay_rgb = overlay[:,:,1:]
    screen = screen + (255-screen) * alpha
    imgpath = os.path.join(tmpdir, 'image-{0}.png'.format(index))
    scipy.misc.imsave(imgpath, screen)

os.system("ffmpeg -f image2 -r 15 -i {0}/image-%d.png -vcodec mpeg4 -q:v 1 -y output.mp4".format(tmpdir))
#shutil.rmtree(tmpdir)

