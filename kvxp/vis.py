import pandas as pd
import numpy as np
import cmasher as cmr
from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import mean_squared_error, mean_absolute_error

def rmse(res):
    return np.sqrt(np.nanmean(res**2))

def mae(res):
    return np.nanmedian(np.abs(res))

def draw_compare(ax, true, pred, xrange=[-2, 0.5], C=None, bins=100, cmap='cmr.dusk_r', if_hex=False, vmin=None, vmax=None):
    
    xx = np.linspace(xrange[0], xrange[1])
    
    res = pred-true
    ax.plot(xx, xx, ls='--', lw=3, c='k', zorder=5)
    
    if vmin is not None:
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
    else:
        norm = colors.LogNorm()
        
    # img = ax.hexbin(true,pred, C=C, bins=bins, cmap=cmap, zorder=4, norm=norm)
    if hex:
        img = ax.hexbin(true,pred, C=C, gridsize=bins, cmap=cmap, zorder=4, norm=norm)
    else:
        img = ax.hist2d(true, pred, bins=bins, cmap=cmap, zorder=4, norm=norm)
        
    ax.set_xlim(xrange);
    ax.set_ylim(xrange);
    
    # ax.text(0.45, 0.2, "RMSE = %.2f"%(rmse(true-pred)),
    #         transform=ax.transAxes, zorder=5)
    # ax.text(0.45, 0.1, " MAE  = %.2f"%(mae(true-pred)),
    #         transform=ax.transAxes, zorder=5)

    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", size="33%", pad=0)

    ax.figure.add_axes(ax2)
    
    if hex:
        ax2.hexbin(true, res, C=C, gridsize=bins, cmap=cmap, zorder=4, norm=norm)
    else:
        ax2.hist2d(true, res, cmap=cmap, bins=bins, zorder=5, norm=colors.LogNorm())
        
    ax2.axhline(y=0, c='k', zorder=6, lw=3, ls="--")
    ax2.axhline(y=np.percentile(res, 14), c='grey', zorder=6, lw=1, ls='--')
    ax2.axhline(y=np.percentile(res, 86), c='grey', zorder=6, lw=1, ls='--')
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
    ax.plot(xx, xx, ls='--', lw=3, c='k', zorder=5)
    
    norm = colors.LogNorm()
        
    img = ax.hist2d(true,pred, bins=bins, cmap=cmap, zorder=4, norm=norm)
    ax.set_xlim(xrange);
    ax.set_ylim(xrange);
    ax.text(0.01, 0.9, "Scatter = %.2f"%(np.std(true-pred)),
            transform=ax.transAxes, zorder=6)
    ax.text(0.01, 0.8, "  Bias  = %.2f"%(np.mean(true-pred)),
            transform=ax.transAxes, zorder=6)

    divider = make_axes_locatable(ax)
    ax2 = divider.append_axes("bottom", size="33%", pad=0)

    ax.figure.add_axes(ax2)
    ax2.hist2d(true, res, cmap=cmap, bins=bins, zorder=5, norm=colors.LogNorm())
    ax2.axhline(y=0, c='k', zorder=6, lw=3, ls="--")
    ax2.axhline(y=np.percentile(res, 14), c='w', zorder=6, lw=1, ls='--')
    ax2.axhline(y=np.percentile(res, 86), c='w', zorder=6, lw=1, ls='--')
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


def draw_coeff(ax1, ax2, bp, rp, teff, cmap='cmr.prinsenvlag_r'):
    # cmap = plt.get_cmap('cmr.pride')
    
    norm = Normalize(vmin=4000, vmax=5000)
    colors_ = [cmap(norm(_)) for _ in teff]
    
    for i,r in enumerate(bp[:500]):
        ax1.plot(r, c=colors_[i], lw=0.5, alpha=0.5)
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    clb1 = plt.colorbar(sm, ax=ax1, ticks=[4000, 4300, 4600, 4900], extend='both')
    clb1.ax.set_yticklabels([4000, 4300, 4600, 4900]) 
    clb1.set_label(r"T$_{\rm eff}$/K", rotation=0, y=1.12, labelpad=-35)
    
    for i,r in enumerate(rp[:500]):
        ax2.plot(r, c=colors_[i], lw=0.5, alpha=0.5)
    
    sm = plt.cm.ScalarMappable(cmap='cmr.prinsenvlag_r', norm=norm)
    sm.set_array([])
    clb2 = plt.colorbar(sm, ax=ax2, ticks=[4000, 4300, 4600, 4900], extend='both')
    clb2.ax.set_yticklabels([4000, 4300, 4600, 4900]) 
    clb2.set_label(r"T$_{\rm eff}$/K", rotation=0, y=1.12, labelpad=-35)
    
    ax1.set_xlabel("Basis index of BP");
    ax1.set_ylabel("Normalized coefficient");
    
    ax2.set_xlabel("Basis index of RP");
    ax2.set_ylabel("Normalized coefficient");

    # ax1.annotate("BP", (0.8, 0.8), xycoords='figure fraction')
    # ax2.annotate("RP", (0.8, 0.8), xycoords='figure fraction')
    ax1.set_xticks(np.arange(1,11,1));
    ax2.set_xticks(np.arange(1,11,1));
    ax1.set_xticklabels(np.arange(1,11,1));
    ax2.set_xticklabels(np.arange(1,11,1));

    return ax1, ax2

def draw_pars(ax1, ax2, teff, logg, moh, aom, gridsize=(50,50), C1=None, C2=None, vmin=None, vmax=None, cmap='cmr.prinsenvlag_r'):
    
    norm_kiel = colors.Normalize(vmin=vmin[0], vmax=vmax[0]) if vmin[0] is not None else colors.LogNorm()
    norm_abund = colors.Normalize(vmin=vmin[1], vmax=vmax[1]) if vmin[1] is not None else colors.LogNorm(1, 100)
    
    img1 = ax1.hexbin(teff, logg, C=C1, norm=norm_kiel, cmap=cmap, gridsize=gridsize)

    ax1.set_xlim([6500, 3800]);
    ax1.set_ylim([4.5, 0.1]);
    
    img2 = ax2.hexbin(moh, aom, gridsize=gridsize, C=C2, 
                      norm=norm_abund, cmap=cmap, )
    ax2.set_xlim([-2.5, 0.5]);
    ax2.set_ylim([-0.3, 0.5]);
    
    if C1 is None and C2 is None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(-2, 0.2))
        sm.set_array([])
        clb1 = plt.colorbar(sm, ax=ax1, extend='both', ticks=[-2.0, -1.5, -1.0, -0.5, 0.0]);
        clb1.ax.set_yticklabels([-2.0, -1.5, -1.0, -0.5, 0.0]) 
        clb1.set_label(r"[M/H]", rotation=0, y=1.12, labelpad=-35)

        clb2 = fig.colorbar(img2, ax=ax2, extend='max');
        clb2.set_label(r"Counts", rotation=0, y=1.12, labelpad=-35)
        
    else:
        clb1 = plt.colorbar(img1, ax=ax1, extend='both')
        clb2 = plt.colorbar(img2, ax=ax2, extend='max')
    
    ax1.set_xlabel(r"T$_{\rm eff}$/K");
    ax1.set_ylabel(r"$\log$ g");
    ax2.set_xlabel('[M/H]');
    ax2.set_ylabel(r'[$\alpha$/M]');
    
    return ax1, ax2, img1, img2, clb1, clb2