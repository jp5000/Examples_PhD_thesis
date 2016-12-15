import numpy as np
import scipy as sp
import scipy.stats as stats
import scipy.special as special
import scipy.optimize as so
import pandas as pd
import chaospy as cp
import sys

import matplotlib as mpl
import matplotlib.pylab as plt
from matplotlib.mlab import griddata
import matplotlib.tri as tri
import seaborn as sns

from colormaps import cmaps

sns.set_style("ticks")#'whitegrid')#
#sns.set_context("talk")
sns.set_style({'axes.linewidth':0.5,
               'xtick.direction': u'in',
               'xtick.major.size': 1.,
               'xtick.minor.size': 0.5,
               'ytick.direction': u'in',
               'ytick.major.size': 1.,
               'ytick.minor.size': 0.5})

def find_confidence_interval(x, pdf, confidence_level):
    return pdf[pdf > x].sum() - confidence_level

def density_contour(hexbin, ax=None, c='k', colors='grey', alpha=1, clabels=True):
    """ Create a density contour plot.
    Parameters
    ----------
    xdata : numpy.ndarray
    ydata : numpy.ndarray
    nbins_x : int
        Number of bins along x dimension
    nbins_y : int
        Number of bins along y dimension
    ax : matplotlib.Axes (optional)
        If supplied, plot the contour to this axis. Otherwise, open a new figure
    contour_kwargs : dict
        kwargs to be passed to pyplot.contour()
    """

    #H, xedges, yedges = np.histogram2d(xdata, ydata, bins=(nbins_x,nbins_y), normed=True)
    #x_bin_sizes = (xedges[1:] - xedges[:-1]).reshape((1,nbins_x))
    #y_bin_sizes = (yedges[1:] - yedges[:-1]).reshape((nbins_y,1))
    #pdf = (H*(x_bin_sizes[0,0]*y_bin_sizes[0,0]))
    #X, Y = 0.5*(xedges[1:]+xedges[:-1]), 0.5*(yedges[1:]+yedges[:-1])

    pdf = hexbin.get_array()/np.sum(hexbin.get_array().flatten())
    verts = hexbin.get_offsets()
    x = [verts[offc][0] for offc in xrange(verts.shape[0])]
    y = [verts[offc][1] for offc in xrange(verts.shape[0])]

    # Create the Triangulation; no triangles so Delaunay triangulation created.
    triang = tri.Triangulation(x, y)

    # Predict iso-pdf confidence levels
    one_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.68), xtol=1e-20)
    two_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.95), xtol=1e-20)
    three_sigma = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.997), xtol=1e-20)
    #extreme = so.brentq(find_confidence_interval, 0., 1., args=(pdf, 0.9999))
    levels = [one_sigma, two_sigma, three_sigma]#, extreme]

    # masking badly shaped triangles at the border of the triangular mesh.
    #min_circle_ratio = .01
    #mask = tri.TriAnalyzer(triang).get_flat_tri_mask(min_circle_ratio)
    #triang.set_mask(mask)

    # Refine data
    #refiner = tri.UniformTriRefiner(triang)
    #triang, pdf = refiner.refine_field(pdf, subdiv=3)

    # define grid.
    X = np.sort(np.unique(x))
    dX = X[1]-X[0]
    X = np.linspace(np.min(X)-dX,np.max(X)+dX,1000)

    Y = np.sort(np.unique(y))
    dY = Y[1]-Y[0]
    Y = np.linspace(np.min(Y)-dY,np.max(Y)+dY,1000)
    # grid the data.
    #Z = griddata(x, y, pdf, X, Y, interp='linear')#'nn')#

    # Interpolate to regularly-spaced quad grid.
    X, Y = np.meshgrid(X,Y)
    interp_tri = tri.CubicTriInterpolator(triang, pdf, kind='geom')
    Z = interp_tri(X, Y)

    fmt = {}
    strs = ['68%', '95%', '99.7%', '99.99%']

    # Label every other level using strings
    for l, s in zip(levels, strs):
        fmt[l] = s

    if ax == None:
        CS = plt.contour(X, Y, Z, levels=levels, origin="lower",linewidths=1., colors=colors, alpha=alpha)
        #CS = plt.tricontour(triang, pdf, levels=levels, origin="lower",linewidths=1., colors=colors, alpha=alpha)
        if clabels:
            labels=plt.clabel(CS, CS.levels, inline=False, fmt=fmt, fontsize=6, colors=c, alpha=1.)
            for l in labels: l.set_rotation(0)
    else:
        CS = ax.contour(X, Y, Z, levels=levels, origin="lower", linewidths=1., colors=colors, alpha=alpha)
        #CS = ax.tricontour(triang, pdf, levels=levels, origin="lower",linewidths=1., colors=colors, alpha=alpha)
        if clabels:
            labels=ax.clabel(CS, CS.levels, inline=False, fmt=fmt, fontsize=6, colors=c, alpha=1.)
            for l in labels: l.set_rotation(0)
    return ax

Reds=cmaps['plasma_r']#sns.cubehelix_palette(light=.8,dark=0.004, gamma=0.15, as_cmap=True)
Reds._init()
Reds._lut[0,:] = (1,1,1,1)

Blues=cmaps['viridis_r']#sns.cubehelix_palette(light=.8,dark=0.004,rot=-.25, gamma=0.15, as_cmap=True)
Blues._init()
Blues._lut[0,:] = (1,1,1,1)

#sns.palplot(Reds(np.linspace(0,1,10)))
#sns.palplot(Blues(np.linspace(0,1,10)))

def pair_plot(df,var_list,var_lims,var_labels,
              num_inputs=4, vmax=4,
              bins=30,alpha=0.7,colors=None, marker='.', ms=20, clabels=True,
              opt='hexbin', fig=None,ax=None,figsize=[16,12]):

    if fig==None:
        Nx = len(var_list)
        Ny = len(var_list)
        fig, ax = plt.subplots(Nx, Ny, gridspec_kw={'width_ratios':[2*figsize[0]/figsize[1] for i in range(Ny-1)]+[1],
                                            'height_ratios':[1]+[2 for i in range(Nx-1)],},figsize=figsize)

    for i,var_name in enumerate(var_list):
        for j,var_name2 in enumerate(var_list):
            if (j<i):

                y = df[var_name].values
                x = df[var_name2].values
                if i>=num_inputs:
                    if opt=='hexbin':
                        hexbin=ax[i,j].hexbin(x, y, gridsize=bins, cmap=Reds,
                                   mincnt=0, extent=var_lims[var_name2]+var_lims[var_name])
                        ax[i,j].hexbin(x, y, gridsize=bins, bins='log', cmap=Reds, vmin=0, vmax=vmax, alpha=alpha,
                                   edgecolors='none',mincnt=0, extent=var_lims[var_name2]+var_lims[var_name])
                        density_contour(hexbin, ax=ax[i,j], alpha=1., colors='grey', c='k', clabels=clabels)
                    elif opt=='scatter':
                        if colors==None:
                            ax[i,j].scatter(x, y, alpha=alpha, marker=marker, s=ms, color=sns.color_palette('Reds_d')[1], zorder=10000)
                        else:
                            ax[i,j].scatter(x, y, alpha=alpha, marker=marker, s=ms, color=colors,zorder=10000)
                else:
                    if opt=='hexbin':
                        hexbin=ax[i,j].hexbin(x, y, gridsize=bins, cmap=Blues,
                                   mincnt=0, extent=var_lims[var_name2]+var_lims[var_name])
                        ax[i,j].hexbin(x, y, gridsize=bins, bins='log', cmap=Blues, vmin=0, vmax=vmax, alpha=alpha,
                                   edgecolors='none',mincnt=0, extent=var_lims[var_name2]+var_lims[var_name])
                        density_contour(hexbin, ax=ax[i,j], alpha=1, colors='grey', c='k', clabels=clabels)
                    elif opt=='scatter':
                        if colors==None:
                            ax[i,j].scatter(x, y, alpha=alpha, marker=marker, s=ms, color=sns.color_palette("Blues_d")[1],zorder=10000)
                        else:
                            ax[i,j].scatter(x, y, alpha=alpha, marker=marker, s=ms, color=colors,zorder=10000)
                ax[i,j].set_ylim(var_lims[var_name])
                ax[i,j].set_xlim(var_lims[var_name2])

            elif (i==0)&(j<len(var_list)-1):
                y = df[var_name2].values
                if colors==None:
                    if i>=num_inputs: ax[i,j].hist(y, bins=bins, normed=True, range=var_lims[var_name2],lw=0,
                                                   color=sns.color_palette('Reds_d')[1])
                    else: ax[i,j].hist(y, bins=bins, normed=True, range=var_lims[var_name2], lw=0,
                                       color=sns.color_palette('Blues_d')[1])
                else:
                    ax[i,j].hist(y, bins=bins, normed=True, range=var_lims[var_name2],lw=1,color='k', histtype=u'step')
                ax[i,j].set_xlim(var_lims[var_name2])
                ax[i,j].set_yticklabels([])
            elif (j==len(var_list)-1)&(i>0):
                x = df[var_name].values
                if colors==None:
                    if i>=num_inputs: ax[i,j].hist(x, bins=bins, normed=True, range=var_lims[var_name],lw=0,
                                                   color=sns.color_palette('Reds_d')[1], orientation=u'horizontal')
                    else: ax[i,j].hist(x, bins=bins, normed=True, range=var_lims[var_name], lw=0,
                                       color=sns.color_palette('Blues_d')[1], orientation=u'horizontal')
                else:
                    ax[i,j].hist(x, bins=bins, normed=True, range=var_lims[var_name],lw=1,color='k', histtype=u'step', orientation=u'horizontal')
                ax[i,j].set_ylim(var_lims[var_name])
                ax[i,j].set_xticklabels([])
            else:
                ax[i,j].set_visible(False)
            if  i!=0:
                ax[j,i].set_yticklabels([])
            if  i==0:
                if  j>0:
                    ax[j,i].set_ylabel(var_labels[var_name2])
                else: ax[j,i].set_ylabel('PDF')

            if  i!=len(var_list)-1:
                ax[i,j].set_xticklabels([])
            if  i==len(var_list)-1:
                if  j<len(var_list)-1:
                    ax[i,j].set_xlabel(var_labels[var_name2])
                else:
                    ax[i,j].set_xlabel('PDF')
                #xticks = ax[i,j].xaxis.get_major_ticks()
                #xticks[-1].label1.set_visible(False)
                #if len(xticks)>8:
                #    for itick in range(1,len(xticks)-1,2):
                #        xticks[itick].label1.set_visible(False)

    plt.subplots_adjust(left  = .1,  # the left side of the subplots of the figure
                        right = .99,  # the right side of the subplots of the figure
                        bottom = .10, # the bottom of the subplots of the figure
                        top = .99,    # the top of the subplots of the figure
                        wspace = 0.1, # the amount of width reserved for blank space between subplots
                        hspace = 0.1) # the amount of height reserved for white space between subplots
    return fig,ax


def pair_plot_last_row(df,var_list,var_lims,var_labels,
              num_inputs=4, vmax=4,
              bins=30,alpha=0.7,colors=None, marker='.', ms=20, clabels=True,
              opt='hexbin', fig=None,ax=None,figsize=[12,3], xlabels=True):

    if fig==None:
        fig, ax = plt.subplots(1, len(var_list), gridspec_kw={'width_ratios':[2 for i in range(len(var_list)-1)]+[1]},
                               figsize=figsize)

    i = 0
    var_name = var_list[-1]
    for j,var_name2 in enumerate(var_list):
        if (j<len(var_list)-1):
            y = df[var_name].values
            x = df[var_name2].values

            if opt=='hexbin':
                hexbin=ax[j].hexbin(x, y, gridsize=bins, cmap=Reds,
                           mincnt=0, extent=var_lims[var_name2]+var_lims[var_name])
                ax[j].hexbin(x, y, gridsize=bins, bins='log', cmap=Reds, vmin=0, vmax=vmax, alpha=alpha,
                           edgecolors='none',mincnt=0, extent=var_lims[var_name2]+var_lims[var_name])
                density_contour(hexbin, ax=ax[j], alpha=1., colors='grey', c='k', clabels=clabels)
            elif opt=='scatter':
                if colors==None:
                    ax[j].scatter(x, y, alpha=alpha, marker=marker, s=ms, color=sns.color_palette('Reds_d')[1], zorder=10000)
                else:
                    ax[j].scatter(x, y, alpha=alpha, marker=marker, s=ms, color=colors, zorder=10000)
            ax[j].set_ylim(var_lims[var_name])
            ax[j].set_xlim(var_lims[var_name2])

        else:
            x = df[var_name].values
            if colors==None:
                ax[j].hist(x, bins=bins, normed=True, range=var_lims[var_name],lw=0,
                           color=sns.color_palette('Reds_d')[1], orientation=u'horizontal')
            else:
                ax[j].hist(x, bins=bins, normed=True, range=var_lims[var_name],lw=1, color='k',
                           histtype=u'step', orientation=u'horizontal')
            ax[j].set_ylim(var_lims[var_name])
            ax[j].set_yticklabels([])

        if  j!=0:
            ax[j].set_yticklabels([])
        else:
            ax[j].set_ylabel(var_labels[var_name])
        if  xlabels:
            if  j<len(var_list)-1:
                ax[j].set_xlabel(var_labels[var_name2])
                xticks = ax[0].xaxis.get_major_ticks()
                xticks[0].label1.set_visible(False)
            else:
                ax[j].set_xticklabels([])
                ax[j].set_xlabel('PDF')
        else:
            ax[j].set_xticklabels([])

    plt.subplots_adjust(left  = .07,  # the left side of the subplots of the figure
                        right = .99,  # the right side of the subplots of the figure
                        bottom = .25, # the bottom of the subplots of the figure
                        top = .95,    # the top of the subplots of the figure
                        wspace = 0.1, # the amount of width reserved for blank space between subplots
                        hspace = 0.1) # the amount of height reserved for white space between subplots
    return fig,ax



