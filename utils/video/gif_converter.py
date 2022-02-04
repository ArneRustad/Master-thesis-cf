def play_gif(filename, fps = None):
    if (fps == None):
        return(Image(filename))
    else:
        gif = imageio.mimread(filename)
        gif_change_speed = "temp_gif_change_speed.gif"
        imageio.mimsave(gif_change_speed, gif, fps=fps)
        video = Image(gif_change_speed)
        os.remove(gif_change_speed)
        return(video)

def gif_to_mp4(filename_gif, fps = None, new_filename = None):
    if(new_filename == None):
        new_filename = f"{filename_gif[:-4]}.mp4"

    if (fps == None):
        gif_changed_speed = filename_gif
    else:
        gif = imageio.mimread(filename_gif_changed_speed)
        gif_changed_speed = "temp_gif_change_speed.gif"
        imageio.mimsave(gif_change_speed, gif, fps=fps)
        video = Image(gif_change_speed)

    clip = moviepy.editor.VideoFileClip(gif_changed_speed)
    clip.write_videofile(new_filename)
    if(fps != None):
        os.remove(gif_changed_speed)
    return(new_filename)