import matplotlib
from matplotlib.lines import Line2D
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import matplotlib.pylab as plt
import geomstats.backend as gs
import geomstats.visualization as visualization
from geomstats.learning.frechet_mean import FrechetMean
import numpy as np
#matplotlib.use("Agg")  # NOQA


def initial_mean(pu, metric):
    """
    Initialize mean geodesic
    """
    # compute mean of base points
    mean_p = FrechetMean(metric)
    mean_p.fit(pu[:, 0])
    # compute mean of tangent vectors
    PT = lambda p, u: metric.parallel_transport(u, p, end_point=mean_p.estimate_)
    mean_v = gs.mean([PT(*pu_i) for pu_i in pu], 0)
    return gs.array([mean_p.estimate_, mean_v])


def visSphere(points_list, color_list,  size=15):
    """
    Visualize groups of points on the 2D-sphere
    """
    #label_list = ['Random geodesic', 'Mean geodesic']
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111, projection="3d")
    sphere = visualization.Sphere()
    sphere.draw(ax, marker=".")
    for i in range(len(points_list)):
        for points in points_list[i]:
            points = gs.to_ndarray(points, to_ndim=2)
            sphere.draw_points(ax, points=points, color=color_list[i], marker=".")
    #ax.set_title("")
    plt.show()


def visKen(points_list, color_list, marker_list='.', size=11):
    """
    Visualize landmarks
    """
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111)
    points = gs.to_ndarray(points_list, to_ndim=2)
    points = points.reshape(-1, 2)
    plt_plot, = plt.plot(points[:, 0], points[:, 1], 'o', markersize=5, markerfacecolor=color_list[0], markeredgewidth=.5, markeredgecolor='k')

    plt.text(0.38, 0.25, 'Brg')
    plt.text(-0.02, 0.25, 'Lara')
    plt.text(-0.22, 0.2, 'IPP')
    plt.text(-0.3, 0.0, 'Opi')
    plt.text(-0.25, -0.15, 'Bas')
    plt.text(-0.05, -0.15, 'SOS')
    plt.text(0.2, -0.11, 'ISS')
    plt.text(0.45, -0.07, 'SES')
    
    ax.set_title(f'Rat skull measurements: 8 landmarks at 8 time points per subject (Subject=0 plots all data)')
    def upd(Subject):
        subj = Subject - 1
        if subj >= 0:
            points = gs.to_ndarray(points_list[int(subj * 8):int((subj + 1) * 8)], to_ndim=2)
        else:
            points = gs.to_ndarray(points_list, to_ndim=2)
        points = points.reshape(-1, 2)
        plt_plot.set_xdata(points[:, 0])
        plt_plot.set_ydata(points[:, 1])
        fig.canvas.draw_idle()
    #

    interact(upd, Subject=widgets.IntSlider(min=0, max=18, step=1, value=0));


def visKenGeo(geo_list, mean, size=10):
    """
    Visualize landmark-components of geodesics and their mean geodesic
    """
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111)
    k_landmarks = geo_list.shape[2]
    for geo in geo_list:
        #points = gs.to_ndarray(points, to_ndim=2)
        p, q = geo[0], geo[0] + geo[1]
        for j in range(k_landmarks):
            plt.plot([p[j, 0], q[j, 0]], [p[j, 1], q[j, 1]], color='#008c04', linewidth=1.5, alpha=0.8)

    p, q = mean[0], mean[0] + mean[1]
    for j in range(k_landmarks):
        plt.plot([p[j, 0], q[j, 0]], [p[j, 1], q[j, 1]], color='k', linewidth=2.5)

    plt.text(0.38, 0.25, 'Brg')
    plt.text(-0.02, 0.25, 'Lara')
    plt.text(-0.22, 0.2, 'IPP')
    plt.text(-0.3, 0.0, 'Opi')
    plt.text(-0.25, -0.15, 'Bas')
    plt.text(-0.05, -0.15, 'SOS')
    plt.text(0.2, -0.11, 'ISS')
    plt.text(0.45, -0.07, 'SES')

    ax.set_title("Mean geodesic and individual geodesics")
    indivGeo = Line2D([0], [0], label='Individual geodesics', color='#008c04')
    meanLGeo = Line2D([0], [0], label='Mean geodesic', color='k')
    plt.legend(handles=[indivGeo, meanLGeo], loc='center')

    plt.show()


def visKenTPCA(explained_variance, scores, size=10):
    """
    Visualize explained variance and distribution of tPCA scores of input shapes
    """
    rel_cum_var = np.cumsum(explained_variance)
    tot_var = rel_cum_var[-1]
    rel_cum_var = 100.0 * rel_cum_var / tot_var

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(2 * size, size))

    fig.suptitle('Tangent principal component analysis')
    ax1.set(ylabel='Relative cumulative variance [%]')
    ax1.set_title('tPCA relative cumulative variance')

    ax1.bar([f'{i+1}' for i in range(len(rel_cum_var))], rel_cum_var, color='grey')

    ax2.set(xlabel='1st axis of variation', ylabel='2nd axis of variation')
    ax2.set_title('tPCA scores')
    ax2.plot(scores[:, 0], scores[:, 1], 'o', markersize=10, markerfacecolor='#008c04', markeredgewidth=.5,
             markeredgecolor='k')

    for i, txt in enumerate(range(1, len(scores) + 1)):
        ax2.annotate(txt, (scores[i, 0] + 0.00075, scores[i, 1] + 0.00075))

    plt.show()


def load_data():
    return np.load('rat_skulls.npy')