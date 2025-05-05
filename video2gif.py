import os
from moviepy.editor import VideoFileClip

folder_name = "video"


def convert_file_to_gif(file_name):
    """
    Convert a video file to gif using moviepy
    :param file_name: name of the video file
    :return: None
    """
    # create gif name
    gif_name = os.path.splitext(file_name)[0] + ".gif"
    # create gif
    videoClip = VideoFileClip(os.path.join(folder_name, file_name))
    videoClip.write_gif(os.path.join("gifs", gif_name))

def main():

    # loop through all videos in the folder
    if not os.path.exists("gifs"):
        os.makedirs("gifs")
    # loop through all videos in the folder
    for file_name in os.listdir(folder_name):
        print(f'file name {file_name}')
        if file_name.endswith(".mp4"):
            videoClip = VideoFileClip(os.path.join(folder_name, file_name))
            # create gif name
            gif_name = os.path.splitext(file_name)[0] + ".gif"
            # create gif
            videoClip.write_gif(os.path.join("gifs", gif_name))


    return
if __name__ == '__main__':
    main()
