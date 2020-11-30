import os
from glob import glob
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pandas as pd
import geopandas as gpd
import contextily as ctx
from tqdm import tqdm
import imageio

def draw_hist2d_ctx(ax, x, y, lon_lim, lat_lim, norm=None, title=None):
    _, _, _, im = ax.hist2d(x, y, cmap='coolwarm', alpha=0.4, bins=60, norm=norm)
    ax.set_xlim(*lon_lim)
    ax.set_ylim(*lat_lim)
    if title is not None:
        ax.set_title(title)
    ctx.add_basemap(ax, crs="EPSG:4326", zoom=10)
    return im

def write_gif(output_dir):
    files = sorted(glob(os.path.join(output_dir, '*.png')))
    images = [imageio.imread(f)[:,:,:3] for f in files]
    imageio.mimsave(os.path.join(output_dir, 'heatmap.gif'), images, fps=2)

def draw_heatmap_single(output_dir, requests, lon_lim, lat_lim, lon_col, lat_col):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    grouped = requests.groupby(by=requests['t1'].dt.hour)
    counts = []
    for t in tqdm(range(24)):
        df = grouped.get_group(t)
        fig, ax = plt.subplots(figsize=(7,5))
        cnt, _, _, im = ax.hist2d(df[lon_col], df[lat_col], cmap='coolwarm', alpha=0.4, bins=60,
                                  norm=LogNorm(vmin=1, vmax=100))
        counts.append(cnt)
        ax.set_xlim(*lon_lim)
        ax.set_ylim(*lat_lim)
        counts.append(cnt)
        fig.colorbar(im, ax=ax)
        ctx.add_basemap(ax, crs="EPSG:4326", zoom=10)
        plt.title(f'Hour on a workday: {t}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{t:02}.png'), dpi=150)
        plt.close()
    counts = np.array(counts)
    write_gif(output_dir)
    return counts

def draw_heatmap_together(output_dir, requests, lon_lim, lat_lim):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    grouped = requests.groupby(by=requests['t1'].dt.hour)


    for t in tqdm(range(24)):
        df = grouped.get_group(t)
        fig = plt.figure(figsize=(13,5))
        gs = mpl.gridspec.GridSpec(ncols=3, nrows=1, figure=fig, width_ratios=[3,3,0.2])
        gs.update(wspace=0.03)
        norm = LogNorm(vmin=1, vmax=100)

        ax = fig.add_subplot(gs[0,0])
        draw_hist2d_ctx(ax, df['lon1'], df['lat1'], lon_lim, lat_lim, norm=norm, title='Pick up')

        ax = fig.add_subplot(gs[0,1])
        im = draw_hist2d_ctx(ax, df['lon2'], df['lat2'], lon_lim, lat_lim, norm=norm, title='Drop down')
        ax.set_yticks([])

        ax = fig.add_subplot(gs[:,2])
        cb = mpl.colorbar.ColorbarBase(ax, cmap=mpl.cm.get_cmap('coolwarm'), norm=norm)

        plt.suptitle(f'Hour on a workday: {t}')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(output_dir, f'{t:02}.png'), dpi=150)
        plt.close()
    write_gif(output_dir)

if __name__ == '__main__':
    # Load and preprocessing
    orderfile = '../data/data4/total_ride_request/order_20161101'
    requests = pd.read_csv(orderfile,
                           names=['order', 't1', 't2', 'lon1', 'lat1', 'lon2', 'lat2', 'r'])
    requests['t1'] = requests['t1'].apply(pd.Timestamp, unit='s', tz='Asia/Shanghai')
    requests['t2'] = requests['t2'].apply(pd.Timestamp, unit='s', tz='Asia/Shanghai')
    lon_lim = (103.78, 104.30)
    lat_lim = (30.45, 30.9)
    requests = requests[
        (requests[['lon1', 'lon2']].min(axis=1) > lon_lim[0]) &
        (requests[['lon1', 'lon2']].max(axis=1) < lon_lim[1]) &
        (requests[['lat1', 'lat2']].min(axis=1) > lat_lim[0]) &
        (requests[['lat1', 'lat2']].max(axis=1) < lat_lim[1])
    ]

    # Draw pickup
    counts = draw_heatmap_single('pickup', requests, lon_lim, lat_lim, 'lon1', 'lat1')

    # Draw pickup and dropoff
    draw_heatmap_together('together', requests, lon_lim, lat_lim)

