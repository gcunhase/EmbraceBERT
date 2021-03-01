import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import seaborn as sns

import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from utils import ensure_dir


"""Source: https://www.datacamp.com/community/tutorials/introduction-t-sne?utm_source=adwords_ppc&utm_campaignid=1455363063&utm_adgroupid=65083631748&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=278443377092&utm_targetid=aud-763347114660:dsa-429603003980&utm_loc_interest_ms=&utm_loc_physical_ms=1009879&gclid=Cj0KCQiAh4j-BRCsARIsAGeV12DQJS7ohBmQnG23HHneNF_PY_0g0JQI3pC9z9LWoMy1Rp5LUr2w8V0aAuCzEALw_wcB"""

# To maintain reproducibility, the random state variable RS=1
RS = 1
RESULTS_ROOT_DIR = './visualizations/'
ensure_dir(RESULTS_ROOT_DIR)


def calculate_pca(X, y, y_pred, n_components=50, res_subdir=RESULTS_ROOT_DIR):
    # n_components = 70
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X)

    pca_df = pd.DataFrame(columns=['pca1', 'pca2', 'pca3', 'pca4'])

    pca_df['pca1'] = pca_result[:, 0]
    pca_df['pca2'] = pca_result[:, 1]
    pca_df['pca3'] = pca_result[:, 2]
    pca_df['pca4'] = pca_result[:, 3]
    pca_df['y'] = y
    pca_df['y_pred'] = y_pred

    print('Variance explained per principal component: {}'.format(pca.explained_variance_ratio_))

    top_two_comp = pca_df[['pca1', 'pca2', 'y_pred']]  # taking first and second principal component
    seaborn_scatter(top_two_comp, axis_names=["pca1", "pca2", "y_pred"],
                    title='PCA-{} plot'.format(n_components), fig_filename='{}pca_{}.png'.format(res_subdir, n_components))
    top_two_comp = pca_df[['pca1', 'pca2', 'y']]  # taking first and second principal component
    seaborn_scatter(top_two_comp, axis_names=["pca1", "pca2", "y"],
                    title='PCA-{} plot'.format(n_components),
                    fig_filename='{}pca_{}_target.png'.format(res_subdir, n_components))
    # f, ax, sc, txts = fashion_scatter(top_two_comp.values, y_pred)  # Visualizing the PCA output
    # f.show()
    return pca_result


def calculate_tsne(pca_result, y, y_pred, perplexity=30, n_iter=300, pca_n_components=50, res_subdir=RESULTS_ROOT_DIR):
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=n_iter, random_state=RS)
    tsne_result = tsne.fit_transform(pca_result)
    tsne_df = pd.DataFrame(columns=['tsne-pca-1', 'tsne-pca-2', 'y', 'y_pred'])
    tsne_df['tsne-pca-1'] = tsne_result[:, 0]
    tsne_df['tsne-pca-2'] = tsne_result[:, 1]
    tsne_df['y'] = y
    tsne_df['y_pred'] = y_pred

    tsne_df_pred = tsne_df[["tsne-pca-1", "tsne-pca-2", "y_pred"]]
    seaborn_scatter(tsne_df_pred, axis_names=["tsne-pca-1", "tsne-pca-2", "y_pred"],
                    title='t-SNE-{} with PCA-{} plot'.format(perplexity, pca_n_components),
                    fig_filename='{results_dir}pca_{n_comp}_tsne_perp{perp}_niter{n_iter}.png'.format(
                        results_dir=res_subdir, n_comp=pca_n_components, perp=perplexity, n_iter=n_iter))

    tsne_df_target = tsne_df[["tsne-pca-1", "tsne-pca-2", "y"]]
    seaborn_scatter(tsne_df_target, axis_names=["tsne-pca-1", "tsne-pca-2", "y"],
                    title='t-SNE-{} with PCA-{} plot'.format(perplexity, pca_n_components),
                    fig_filename='{results_dir}pca_{n_comp}_tsne_perp{perp}_niter{n_iter}_target.png'.format(
                        results_dir=res_subdir, n_comp=pca_n_components, perp=perplexity, n_iter=n_iter))
    # f, ax, sc, txts = fashion_scatter(tsne_result, y_pred)  # Visualizing the PCA output
    # f.show()


def plot_visualization(X, y, y_pred, model_type):
    res_subdir = '{}{}/'.format(RESULTS_ROOT_DIR, model_type)
    ensure_dir(res_subdir)
    X = X.cpu()  # torch.transpose(X, 0, 1).cpu()
    #y = np.transpose(y)
    #y_pred = np.transpose(y_pred)
    sns.set_style('darkgrid')
    sns.set_palette('muted')
    sns.set_context("notebook", font_scale=1.5,
                    rc={"lines.linewidth": 2.5})

    for n_components in [20, 30, 50, 70]:
        # PCA
        # n_components = 50
        pca_result = calculate_pca(X, y, y_pred, n_components=n_components, res_subdir=res_subdir)

        for perplexity in [30, 50, 70, 100]:
            # t-SNE
            calculate_tsne(pca_result, y, y_pred, perplexity=perplexity, n_iter=300, pca_n_components=n_components,
                           res_subdir=res_subdir)
            calculate_tsne(pca_result, y, y_pred, perplexity=perplexity, n_iter=1000, pca_n_components=n_components,
                           res_subdir=res_subdir)
            calculate_tsne(pca_result, y, y_pred, perplexity=perplexity, n_iter=3000, pca_n_components=n_components,
                           res_subdir=res_subdir)

    print("Done")


def seaborn_scatter(df, axis_names=["pca1", "pca2", "y_pred"], title='Plot', fig_filename='plot.png'):
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(1, 1, 1)
    sns.scatterplot(
        x=axis_names[0], y=axis_names[1],
        hue=axis_names[2],
        palette=sns.color_palette("hls", 2),
        data=df,
        legend="full",
        alpha=0.8,
        ax=ax
    )
    ax.set_title(title)
    plt.savefig(fig_filename)
    plt.close(f)
    # f.show()


# Utility function to visualize the outputs of PCA and t-SNE
def fashion_scatter(x, colors):
    # choose a color palette with seaborn.
    num_classes = len(np.unique(colors))
    palette = np.array(sns.color_palette("hls", num_classes))

    # create a scatter plot.
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:, 0], x[:, 1], lw=0, s=40, c=palette[colors.astype(np.int)])
    #plt.xlim(-25, 25)
    #plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

    # add the labels for each digit corresponding to the label
    txts = []

    for i in range(num_classes):

        # Position of each label at median of data points.

        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        txts.append(txt)

    return f, ax, sc, txts
