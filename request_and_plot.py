import os
import requests
import argparse    #argparsing is basically a user dragging dropping a folder in the Analyse button

from plot_gps import *


ap = argparse.ArgumentParser()
ap.add_argument("-f","--image_folder",required=True,help="path to image folder")
args = vars(ap.parse_args())


path = str(args["image_folder"])
list_dir = os.listdir(path)



KERAS_REST_API_URL = "http://localhost:5000/predict"


abs_path = '/home/kartikeya/Desktop/erc_flask_api/simple-keras-rest-api-master'+'/'+path




preds = []
no_clip_ids =[]

for i in list_dir:
    locs = str(abs_path +'/'+i)
    image = open(locs,"rb").read()
    payload = {"image":image}
    r = requests.post(KERAS_REST_API_URL, files=payload).json()

    if r['Predicted Class']=='Noclip':
        no_clip_ids.append(i)

    preds.append(r)


img_gps_dict = {}

for i in no_clip_ids:
    im_paths = str(abs_path +'/'+i)
    image = Image.open(im_paths)
    exif_data = get_exif_data(image)
    img_gps_dict[i] = get_lat_lon(exif_data)



plot_on_maps(return_gps_coor(no_clip_ids,img_gps_dict))



