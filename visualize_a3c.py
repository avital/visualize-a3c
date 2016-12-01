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

#metadata = metadata[0:30]
for index in range(2, len(metadata)):
    print('{0}/{1}'.format(index+1, len(metadata)))
    history_size = 10
    x = np.arange(max(0, index-history_size), min(len(metadata)-1, index+history_size))

    def plot_scalar(prop, color_func, label, val_min, val_max):
        scalars = [metadata[other_index]['scalars'][prop] for other_index in x]
        y_pos = 1

        fig = plt.figure(figsize=(1.5, 1), dpi=50)
        fig.patch.set_facecolor('none')
        plt.subplot(1, 1, 1, axisbg='none')

        cropped_rel = metadata[index]['scalars'][prop]
        cropped_rel = (cropped_rel - val_min) / (val_max - val_min)
        cropped_rel = max(cropped_rel, 0)
        cropped_rel = min(cropped_rel, 1)
        print(prop, cropped_rel)

        plt.plot(x, scalars, alpha=0.3, color='white')
        plt.axis('off')
        plt.ylim(val_min, val_max)
        plt.xlim(index-history_size, index+history_size)
        plt.axvline(index, color='white')
        plt.axhline(0, color='white', linestyle='dotted')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
#        plt.ylabel(label)
        ax = plt.gca()
        ax.set_autoscale_on(False)
        return fig

    colors = ScalarMappable(cmap=plt.get_cmap('hsv')).to_rgba(np.linspace(0, 1, num=36))

    def plot_actions():
        actions = [metadata[index]['policy_dist']]
        actions = np.array(actions).T
        axes = plt.subplot(1, 1, 1, polar=True, axisbg='none')
        R, theta = np.meshgrid([0.8, 0.4], np.pi - np.linspace(0, 2*np.pi, num=37) + np.pi/36)
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

    metadatum = metadata[index]
    screen = screen_from_metadatum(metadatum)
#    screen = np.expand_dims(screen, axis=-1)
#    screen = np.repeat(screen, 3, axis=-1)
    zoom=2
    screen = np.repeat(screen, zoom, 0)
    screen = np.repeat(screen, zoom, 1)

    fig = plt.figure(figsize=screen.shape[:2][::-1], dpi=1)
    fig.patch.set_facecolor('none')
    plot_actions()

    action_overlay = array_from_figure(fig)
    plt.close(fig)

    green = np.array([0, 1, 0])
    white = np.array([1, 1, 1])
    red = np.array([1, 0, 0])
    reward = metadatum['scalars']['reward']
    value = metadatum['scalars']['value']

    fig = plot_scalar('reward', lambda rel: green * rel + white * (1-rel), 'R', -0.05, 2.1)
    reward_overlay = array_from_figure(fig)
    argb = [1, 0, 1, 0]
    argb_multiplier = np.broadcast_to(argb, reward_overlay.shape)
    reward_overlay = reward_overlay * argb_multiplier
    plt.close(fig)

    fig = plot_scalar('value', lambda rel: green * rel + red * (1-rel), 'V', -0.4, 0.4)
    value_overlay = array_from_figure(fig)
    plt.close(fig)

    overlay = action_overlay
    small_graph_height, small_graph_width = reward_overlay.shape[:2]
    overlay[10:30+small_graph_height*2, 10:20+small_graph_width] = [0.1 * 255, 255, 255, 255]
    overlay[
        15:15+small_graph_height,
        15:15+small_graph_width
    ] += (reward_overlay * 0.9).astype('uint8')
    overlay[
        15+10+small_graph_height:15+10+2*small_graph_height,
        15:15+small_graph_width
    ] += (value_overlay * 0.9).astype('uint8')

    alpha = overlay[:,:,1] / 255 # ???
    alpha = alpha * overlay[:,:,0] / 255
    alpha = np.expand_dims(alpha, axis=-1)
    overlay_rgb = overlay[:,:,1:]
    screen = screen + (255-screen) * alpha
    imgpath = os.path.join(tmpdir, 'image-{0}.png'.format(index))
    scipy.misc.imsave(imgpath, screen)

os.system("ffmpeg -f image2 -r 15 -i {0}/image-%d.png -vcodec mpeg4 -q:v 1 -y output.mp4".format(tmpdir))
#shutil.rmtree(tmpdir)

