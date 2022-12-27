import pandas as pd
import numpy as np
import cmasher as cmr
from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(res):
    return np.sqrt(np.mean(res**2))

def mae(res):
    return np.median(np.abs(res))

def draw_compare(ax, true, pred, xrange=[-2, 0.5], C=None, bins=100, cmap='cmr.dusk_r', **kwargs):
    
    xx = np.linspace(xrange[0], xrange[1])
    
    res = pred-true
    ax.plot(xx, xx, ls='--', lw=5, c='k', zorder=5)
    
    # if kwargs['vmin'] is not None:
    #     norm = colors.LogNorm(vmin=kwargs['vmin'], vmax=kwargs['vmax'])
    # else:
    norm = colors.LogNorm()
        
    # img = ax.hexbin(true,pred, C=C, bins=bins, cmap=cmap, zorder=4, norm=norm)
    img = ax.hist2d(true, pred, bins=bins, cmap=cmap, zorder=4, norm=norm)
    ax.set_xlim(xrange);
    ax.set_ylim(xrange);
    
    ax.text(0.6, 0.2, "RMSE = %.2f"%(rmse(true-pred)),
            transform=ax.transAxes, zorder=3)
    ax.text(0.6, 0.1, " MAE  = %.2f"%(mae(true-pred)),
            transform=ax.transAxes, zorder=3)

    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", size="33%", pad=0)

    ax.figure.add_axes(ax2)
    ax2.hist2d(true, res, cmap=cmap, bins=bins, zorder=5, norm=colors.LogNorm())
    ax2.axhline(y=0, c='k', zorder=6, lw=5, ls="--")
    ax2.axhline(y=np.percentile(res, 14), c='grey', zorder=6, lw=3, ls='--')
    ax2.axhline(y=np.percentile(res, 86), c='grey', zorder=6, lw=3, ls='--')
    ax2.set_xlabel(r"APOGEE");
    ax2.set_ylim(xrange);
    ax2.set_xlim(xrange);
    ax.set_xticks([]);
    
    if C is None:
        return ax, ax2
    else:
        return ax, ax2, img
    
    
def compare_scatter(ax, pred, true, xrange=[-2, 0.5], C=None, cmap='cmr.dusk_r', **kwargs):
    
    xx = np.linspace(xrange[0], xrange[1])
    res = pred-true
    ax.plot(xx, xx, ls='--', lw=5, c='k', zorder=5)
    
    if kwargs['vmin'] is not None:
        # norm = colors.LogNorm(vmin=kwargs['vmin'], vmax=kwargs['vmax'])
        norm = colors.Normalize(vmin=kwargs['vmin'], vmax=kwargs['vmax'])
        # norm = colors.LogNorm()
        
    img = ax.scatter(true,pred, c=C, cmap=cmap, s=1, zorder=4, norm=norm)
    ax.set_xlim(xrange);
    ax.set_ylim(xrange);
    
    ax.text(0.6, 0.2, "RMSE = %.2f"%(rmse(true-pred)),
            transform=ax.transAxes, zorder=3)
    ax.text(0.6, 0.1, " MAE  = %.2f"%(mae(true-pred)),
            transform=ax.transAxes, zorder=3)

    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", size="33%", pad=0)

    ax.figure.add_axes(ax2)
    ax2.scatter(true, res, s=1, cmap=cmap, zorder=5, c=C, norm=norm)
    ax2.axhline(y=0, c='k', zorder=6, lw=5, ls="--")
    ax2.axhline(y=np.percentile(res, 14), c='grey', zorder=6, lw=3, ls='--')
    ax2.axhline(y=np.percentile(res, 86), c='grey', zorder=6, lw=3, ls='--')
    ax2.set_xlabel(r"APOGEE");
    ax2.set_ylim(xrange);
    ax2.set_xlim(xrange);
    ax.set_xticks([]);
    
    if C is None:
        return ax, ax2
    else:
        return ax, ax2, img


def draw_hist2d(ax, true, pred, xrange=[-2, 0.5], C=None, bins=100, cmap='cmr.dusk_r', **kwargs):
    
    xx = np.linspace(xrange[0], xrange[1])
    res = pred-true
    ax.plot(xx, xx, ls='--', lw=5, c='k', zorder=5)
    
    norm = colors.LogNorm()
        
    img = ax.hist2d(true,pred, bins=bins, cmap=cmap, zorder=4, norm=norm)
    ax.set_xlim(xrange);
    ax.set_ylim(xrange);
    ax.text(0.6, 0.2, "RMSE = %.2f"%(rmse(true-pred)),
            transform=ax.transAxes, zorder=3)
    ax.text(0.6, 0.1, " MAE  = %.2f"%(mae(true-pred)),
            transform=ax.transAxes, zorder=3)

    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", size="33%", pad=0)

    ax.figure.add_axes(ax2)
    ax2.hist2d(true, res, cmap=cmap, bins=bins, zorder=5, norm=colors.LogNorm())
    ax2.axhline(y=0, c='k', zorder=6, lw=5, ls="--")
    ax2.axhline(y=np.percentile(res, 14), c='grey', zorder=6, lw=3, ls='--')
    ax2.axhline(y=np.percentile(res, 86), c='grey', zorder=6, lw=3, ls='--')
    ax2.set_xlabel(r"APOGEE");
    ax2.set_ylim(xrange);
    ax2.set_xlim(xrange);
    ax.set_xticks([]);
    
    if C is None:
        return ax, ax2
    else:
        return ax, ax2, img

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


def draw_attention(attn, ax, vmax=0.03, cmap='cmr.eclipse'):
    num_coeff_grid = np.linspace(1,114,114)
    xx, yy = np.meshgrid(num_coeff_grid, num_coeff_grid)
    img = ax.pcolormesh(
        xx, yy, attn, 
        norm=colors.Normalize(vmin=0, vmax=vmax), 
        cmap=cmap
    )
    # ax.set_xscale('log')
    ax.set_xticks([1, 10, 55, 110]);
    ax.set_xticklabels([1, 10, 55, 110]);

    clb = plt.colorbar(img, ax=ax, extend='max');
    clb.set_label(r"Weight", rotation=0, y=1.15, labelpad=-35)

    ax.set_xlabel("XP coefficient index + Photometric");
    return ax