def compare_nmi_matrices(tgans, extra_datasets = None, include_true_data = True, n_q_bins = 40, ncol = None, nrow = None,
                         average_method = "arithmetic", subplot_title_true_dataset = "True dataset",
                         subplot_titles_tgans = None, subplot_titles_extra_datasets = None, figsize = [14,5],
                         compute_diff_nmi_matrices = False, save_dir = None, save_name = None, title = None,
                         data_test = None):
    if (not save_dir is None) and save_name is None:
        if compute_diff_nmi_matrices:
            save_name = "nmi_diff_matrices"
        else:
            save_name = "nmi_matrices"

    if subplot_titles_tgans == None:
        subplot_titles_tgans = [None] * len(tgans)
    else:
        if (len(subplot_titles_tgans) != len(tgans)):
            raise ValueError("Number of tgan subplot titles must be equal to number of tgans")

    if include_true_data:
        subplot_titles_tgans = [subplot_title_true_dataset] + subplot_titles_tgans

    if not extra_datasets is None:
        datasets = [None] * len(tgans) + extra_datasets
        tgans = tgans + [tgans[0]] * len(extra_datasets)
        if (subplot_titles_extra_datasets == None):
            subplot_titles_extra_datasets = [None] * len(extra_datasets)
        else:
            if (len(subplot_titles_extra_datasets) != len(extra_datasets)):
                raise ValueError("Number of subplot titles for the extra datasets must be equal to the number of extra datasets")
        subplot_titles = subplot_titles_tgans + subplot_titles_extra_datasets
    else:
        subplot_titles = subplot_titles_tgans
        datasets = [None] * len(tgans)

    n_subplots = len(tgans) + (1 if include_true_data else 0)

    def map_fig_n_to_indices(curr_fig, ncol, nrow):
        if ncol == 1:
            return curr_fig
        elif nrow == 1:
            return curr_fig
        else:
            curr_fig_col = floor(curr_fig // ncol)
            curr_fig_row = curr_fig - curr_fig_col * ncol
            return curr_fig_col, curr_fig_row

    if ncol == None and nrow == None:
        nrow = 1
        ncol = n_subplots
    elif ncol == None:
        ncol = ceil(n_subplots / nrow)
    elif nrow == None:
        nrow = ceil(n_subplots / ncol)
    else:
        if (nrow * ncol < n_subplots):
            raise ValueError("ncol times nrow must be larger than number of subfigures to plot")

    if data_test is None:
        n_samples = tgans[0].data.shape[0]
    else:
        n_samples = data_test.shape[0]

    fig, axes = plt.subplots(nrow, ncol, figsize = figsize, sharey = True, sharex=True)
    plt.tight_layout()

    plt.title(title)

    curr_fig = 0
    nmi_matrix_truth_train, bins = compute_nmi_matrix(tgans[curr_fig], bins = None, n_q_bins = n_q_bins, generated_data = False,
                                                      retbins = True, average_method = average_method)

    if compute_diff_nmi_matrices:
        #         colors_blue = plt.cm.Blues(np.linspace(0., 1, 128))
        #         colors_red = np.flip(plt.cm.Reds(np.linspace(0, 1, 128)))
        #         colors = np.vstack((colors_red, colors_blue))
        #         cmap_diff_nmi = mcolors.LinearSegmentedColormap.from_list('my_blue_red_colormap', colors)
        cmap_diff_nmi = plt.cm.bwr_r
    #         cmap_diff_nmi = sns.diverging_palette(250, 10, sep = 1, n=200, s=100, as_cmap=True)

    if (include_true_data):
        if data_test is None:
            nmi_matrix_truth = nmi_matrix_truth_train
        else:
            nmi_matrix_truth = compute_nmi_matrix(tgans[curr_fig], dataset = data_test, bins = None, n_q_bins = n_q_bins,
                                                  generated_data = True, retbins = False, average_method = average_method,
                                                  n_samples = n_samples)

        axes_ind = map_fig_n_to_indices(0, ncol, nrow)
        if compute_diff_nmi_matrices:
            axes[axes_ind].imshow(nmi_matrix_truth - nmi_matrix_truth_train, cmap = cmap_diff_nmi, vmin = -1, vmax = 1)
        else:
            axes[axes_ind].imshow(nmi_matrix_truth, cmap = plt.cm.Blues)
        axes[axes_ind].set_xticks([])
        axes[axes_ind].set_yticks([])
        if subplot_titles != None:
            axes[axes_ind].set_title(subplot_titles[curr_fig])

    curr_tgan = 0
    for curr_fig in range(1 if include_true_data else 0, nrow * ncol):
        if (curr_fig < n_subplots):
            nmi_matrix = compute_nmi_matrix(tgans[curr_tgan], dataset = datasets[curr_tgan], n_q_bins = n_q_bins,
                                            generated_data = True, retbins = False, average_method = average_method,
                                            n_samples = n_samples)
            axes_ind = map_fig_n_to_indices(curr_fig, ncol, nrow)
            if compute_diff_nmi_matrices:
                nmi_matrix -= nmi_matrix_truth_train
                im = axes[axes_ind].imshow(nmi_matrix, cmap = cmap_diff_nmi, vmin = -1, vmax = 1)
            else:
                im = axes[axes_ind].imshow(nmi_matrix, cmap = plt.cm.Blues)
            #axes[axes_ind].set_xticks([])
            #axes[axes_ind].set_yticks([])
            xticks = axes[axes_ind].set_xticks(np.arange(0, nmi_matrix_truth.shape[0]))
            yticks = axes[axes_ind].set_yticks(np.arange(0, nmi_matrix_truth.shape[0]))
            xticklabels = axes[axes_ind].set_xticklabels(tgans[curr_tgan].columns, rotation = 90)
            yticklabels = axes[axes_ind].set_yticklabels(tgans[curr_tgan].columns)
            if subplot_titles != None:
                axes[axes_ind].set_title(subplot_titles[curr_fig])
            curr_tgan += 1
        else:
            axes[map_fig_n_to_indices(curr_fig, ncol, nrow)].axis("off")

    fig.colorbar(im, ax=axes.ravel().tolist())
    if not save_dir is None:
        plt.savefig(os.path.join(save_dir, save_name))
    plt.close(fig)
    return fig
    