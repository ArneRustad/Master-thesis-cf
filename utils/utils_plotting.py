def map_str_to_color(vec, color_map = "viridis"):
    cmap = matplotlib.cm.get_cmap(color_map)
    vec_set = set(vec)
    colors = cmap(np.linspace(0, 1, len(vec_set)))
    clr_map = {string : colors[i] for i, string in enumerate(vec_set)}
    return([clr_map[s] for s in vec])