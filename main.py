from configparser import ConfigParser
from utils import detect_gun, inpainting_detected_gun, smile_expression, replacing_gun_with_musical_instrument
from roboflow import Roboflow
import cv2
import numpy as np
from PIL.Image import Image
import io
import base64
import os

filename = "man_with_gun.png"
instrument_file = "guitar.png"
temp_path = "Temp"

isExist = os.path.exists(temp_path)
if not isExist:
   os.makedirs(temp_path)


config = ConfigParser()
print (config.read('config.ini'))

res = detect_gun(filename)
cv2.imwrite(temp_path+"/gun_detected.png",res)

file, msg = inpainting_detected_gun(filename,temp_path+"/gun_detected.png")
print("!!! ",msg)

file, msg = smile_expression(temp_path+"/"+file)
print("!!! ",msg)

res = replacing_gun_with_musical_instrument(temp_path+"/"+file,instrument_file)
cv2.imwrite("final.png",res)
  

