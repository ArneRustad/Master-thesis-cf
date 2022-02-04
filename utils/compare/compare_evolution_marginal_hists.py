def compare_evolution_hists_real_vs_generated(tg, epochs = None, name = "compare_hist_real_vs_generated", fps = 1,
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