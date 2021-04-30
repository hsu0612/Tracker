import subprocess
import os

input_video_path = "D:/GOT/val/"
input_video_dir = os.listdir(input_video_path)

for index, i in enumerate(input_video_dir):
    input_path = os.path.join(input_video_path, i) + "/"
    print(input_path)
    
    subprocess.run(['python', './segmentation/my_approach.py'
        , '--data_path', input_path
        , '--save_num', str(index)] , shell=True)
