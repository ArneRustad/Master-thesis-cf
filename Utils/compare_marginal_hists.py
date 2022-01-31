def compare_hist_real_vs_generated(model, other_gen_dataset = None, n_img_horiz = 4, n_bins = 20, figsize = None,
                                   discrete_xtick_rotation = 45,
                                   title = None, seed = None, save_path = None, save_dir = None):
    if (seed != None):
        tf.random.set_seed(seed)
    if other_gen_dataset is None:
        gen_data = model.generate_data()
    else:
        gen_data = other_gen_dataset
    n_img_vert = (model.n_columns-1) // n_img_horiz + 1
    if(figsize == None):
        figsize = (15, 3*n_img_vert)
    fig, ax = plt.subplots(n_img_vert, n_img_horiz, figsize = figsize)
    fig.suptitle(title)
    img_counter_horiz = 0
    img_counter_vert = 0
    img_counter = 0
    for num_col in model.columns_num:
        min_val = min(min(gen_data[num_col]), min(model.data[num_col]))
        max_val = max(max(gen_data[num_col]),  max(model.data[num_col]))
        bins = np.linspace(min_val, max_val, n_bins + 1, dtype = np.float)
        ax[img_counter_vert, img_counter_horiz].hist(model.data[num_col], density = True, alpha = 0.5,
                                                     bins = bins, label = "Real")
        ax[img_counter_vert, img_counter_horiz].hist(gen_data[num_col], density = True, alpha = 0.5,
                                                     bins = bins, label = "Gen")
        ax[img_counter_vert, img_counter_horiz].set_title(num_col)
        ax[img_counter_vert, img_counter_horiz].legend()
        img_counter_horiz += 1
        img_counter += 1
        if (img_counter_horiz == n_img_horiz):
            img_counter_horiz = 0
            img_counter_vert += 1
    for discrete_col in model.columns_discrete:
        unique_values = np.unique(model.data[discrete_col])
        map_discr_to_int = {s : i for i,s in enumerate(unique_values)}
        bins = np.arange(0, unique_values.size, 0.5) - 0.25
        x_ticks = np.arange(0, unique_values.size)
        ax[img_counter_vert, img_counter_horiz].hist(model.data[discrete_col].map(map_discr_to_int),
                                                     bins = bins, density = True, alpha = 0.5, label = "Real")
        ax[img_counter_vert, img_counter_horiz].hist(gen_data[discrete_col].map(map_discr_to_int),
                                                     bins = bins, density = True, alpha = 0.5, label = "Gen")
        ax[img_counter_vert, img_counter_horiz].set_title(discrete_col)
        ax[img_counter_vert, img_counter_horiz].set_xticks(x_ticks)
        ax[img_counter_vert, img_counter_horiz].set_xticklabels(unique_values)
        ax[img_counter_vert, img_counter_horiz].tick_params(axis='x', labelrotation = discrete_xtick_rotation)
        ax[img_counter_vert, img_counter_horiz].legend()
        img_counter_horiz += 1
        img_counter += 1
        if (img_counter_horiz == n_img_horiz):
            img_counter_horiz = 0
            img_counter_vert += 1
    for i in range(n_img_horiz*n_img_vert - img_counter):
        ax[img_counter_vert, img_counter_horiz].axis("off")
        img_counter_horiz += 1
        if (img_counter_horiz == n_img_horiz):
            img_counter_horiz = 0
            img_counter_vert += 1
    fig.tight_layout()

    if not (save_path is None):
        if not save_dir is None:
            os.makedirs(save_dir, exist_ok = True)
            save_path = os.path.join(save_dir, save_path)
        plt.savefig(save_path)

    plt.close(fig)

    return(fig)

def compare_evolution_hist_real_vs_generated(tg, epochs = None, name = "compare_hist_real_vs_generated", fps = 1,
                                             mult_time_first_image = 1, mult_time_last_image = 1, **kwargs):
    name += ".gif"
    if epochs is None:
        ckpts = tg.ckpt_manager.checkpoints
        epochs = [int(ckpt.replace(tg.ckpt_prefix + "-", "")) for ckpt in ckpts]

    dir_path = ".//temp_gif_compare_hist_real_vs_generated//"
    os.makedirs(dir_path, exist_ok = True)
    filenames = []
    for i in tqdm(range(len(epochs))):
        # plot the line chart
        tg.restore_checkpoint(epoch = epochs[i])

        # create file name and append it to a list
        filename = f'{dir_path}{epochs[i]}.jpg'
        filenames.append(filename)

        # Create figure for current epoch
        fig = compare_hist_real_vs_generated(tg, title = "Epoch %d" % (epochs[i]), save_path = filename)

    # build gif
    last_image_i = len(epochs)
    with imageio.get_writer(name, mode='I', fps = 1) as writer:
        for i, filename in enumerate(filenames):
            image = imageio.imread(filename)
            writer.append_data(image)
            if (i == 0):
                for j in range(1, mult_time_first_image):
                    writer.append_data(image)
            if (i == last_image_i):
                for j in range(1, mult_time_last_image):
                    writer.append_data(image)

    # Remove files
    shutil.rmtree(dir_path)

    return name

pris = lambda: print(20)