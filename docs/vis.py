import pandas as pd
import numpy as np
import cmasher as cmr
from matplotlib import colors

def draw_compare(ax, pred, true, xrange=[-2, 0.5], cmap='cmr.dusk'):
    
    xx = np.linspace(xrange[0], xrange[1])
    
    norm_res = pred-true
    
    ax.plot(xx, xx, ls='--', lw=5, c='k', zorder=5)   
    ax.hist2d(true,pred, bins=100, cmap=cmap, zorder=4, norm=colors.LogNorm())
    ax.set_xlim(xrange);
    ax.set_ylim(xrange);
    return ax

def draw_res(ax, true, res, xrange=[-2, 0.5], yrange=[-0.5, 0.5]):
    ax.hist2d(true, res, bins=100, cmap=cmap, zorder=4, norm=LogNorm())
    ax.set_xlim(xrange);
    ax.set_ylim(yrange);
    ax.axhline(y=0, c='k', zorder=6, lw=5, ls="--")
    return ax

def draw_meddiagram(ax, df, xlabel, ylabel, xedges, yedges,
                    color_name='MH', vrange=[-2, 0.5],
                    color_map='jet', scale='log', clb_title='<[M/H]>'):
    d1 = df.assign(
        x_bins = pd.cut(df[xlabel], xedges),
        y_bins = pd.cut(df[ylabel], yedges)
    )
    
    H = d1.groupby(
        ['y_bins', 'x_bins']
    )[color_name].median().values.reshape((len(yedges)-1, len(xedges)-1))
    H = np.log10(H+1e-6) if scale=='log' else H
    
    X, Y = np.meshgrid(xedges, yedges)
    img = ax.pcolormesh(X, Y, H, cmap=color_map, vmin=vrange[0], vmax=vrange[1])
    
    return ax, X, Y, H