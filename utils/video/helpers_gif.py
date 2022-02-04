from IPython.display import Image
import imageio
import os

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