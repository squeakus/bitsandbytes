#!/usr/bin/python3
from utils import run_cmd

images = {'cat.jpg':'https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Felis_catus-cat_on_snow.jpg/512px-Felis_catus-cat_on_snow.jpg',
        'dog.jpg':'https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Dog-2016-11-b.jpg/512px-Dog-2016-11-b.jpg',
        'plane.jpg': 'https://upload.wikimedia.org/wikipedia/commons/8/8a/It_S_A_Bird_No_It_S_Just_A_Plane_%2842237726%29.jpeg'}

for key in images:
    cmd = "wget -O " + key + " " + images[key]
    run_cmd(cmd)
    run_cmd('mkdir -p images')
    run_cmd('mv *.jpg images')