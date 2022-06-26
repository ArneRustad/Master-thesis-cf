import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy as np

def heatmap2D_classifier(ax, classifier_proba, x1_lim, x2_lim,
                 heat_map_res = 200, incl_axis_labels = True, incl_colorbar = True, ret_contour=False,
                 n_cols = 11):
    x1 = np.linspace(x1_lim[0], x1_lim[1], heat_map_res)
    x2 = np.linspace(x2_lim[0], x2_lim[1], heat_map_res)
    x1_mesh, x2_mesh = np.meshgrid(x1, x2)
    x1_mesh_collapsed = x1_mesh.flatten()
    x2_mesh_collapsed = x2_mesh.flatten()

    y_mesh_collapsed = classifier_proba(np.column_stack((x1_mesh_collapsed, x2_mesh_collapsed)))
    y_mesh = y_mesh_collapsed.reshape(x1_mesh.shape)

    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    colors1 = matplotlib.cm.get_cmap('Blues_r')(np.linspace(0.2, 0.6, n_cols))
    colors2 = matplotlib.cm.get_cmap('Reds')(np.linspace(0.4, 0.8, n_cols))
    # combine them and build a new colormam
    colors = np.vstack((colors1, colors2))
    cmap_br = matplotlib.colors.LinearSegmentedColormap.from_list('my_colormap', colors)
    cont1 = ax.contourf(x1_mesh, x2_mesh, y_mesh, levels = np.linspace(0, 1, n_cols), cmap = cmap_br)
    #cont1 = ax.imshow(y_mesh, cmap=cmap_br, extent=[0,2, 1,5])
    if (incl_axis_labels):
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
    if incl_colorbar:
        plt.colorbar(cont1, ax=ax)
    if ret_contour:
        return cont1