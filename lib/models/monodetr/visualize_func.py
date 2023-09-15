# yf 预测出的点在该物体的ground truth内的比例
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import os
import cv2
import numpy as np
torch.set_grad_enabled(False);

def plot_sample_location_point(img_sizes, point_location, info):
    r20 = []
    r40 = []
    r60 = []
    h = img_sizes[0][0]
    w = img_sizes[0][1]
    for n in info['img_id']:
        point_location_n = point_location[n]
        if len(point_location_n) > 0:
            i = 0
            while i < len(point_location_n):
                point_location_n_i = point_location_n[i]
                b_xr = point_location_n_i[2]
                b_xl = point_location_n_i[4]
                b_yr = point_location_n_i[3]
                b_yl = point_location_n_i[5]
                depth0 = point_location_n_i[11]
                sample_location_point = point_location_n_i[14:270]
                j = 0
                sum = 0
                if depth0 < 20:
                    while j < 255:
                        x1 = sample_location_point[j] * h
                        y1 = sample_location_point[j + 1] * w
                        if x1 > b_xr and x1 < b_xl and y1 < b_yl and y1 > b_yr:
                            sum += 1
                        j = j + 2
                    r0 = sum / 128
                    r20.append(r0)
                elif depth0 > 20 and depth0 < 40:
                    while j < 255:
                        x1 = sample_location_point[j] * h
                        y1 = sample_location_point[j + 1] * w
                        if x1 > b_xr and x1 < b_xl and y1 < b_yl and y1 > b_yr:
                            sum += 1
                        j = j + 2
                    r0 = sum / 128
                    r40.append(r0)
                elif depth0 > 40 and depth0 < 60:
                    while j < 255:
                        x1 = sample_location_point[j] * h
                        y1 = sample_location_point[j + 1] * w
                        if x1 > b_xr and x1 < b_xl and y1 < b_yl and y1 > b_yr:
                            sum += 1
                        j = j + 2
                    r0 = sum / 128
                    r60.append(r0)
                i += 1
    return r20, r40, r60
