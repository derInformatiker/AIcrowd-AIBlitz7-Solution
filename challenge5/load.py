import cv2
import math
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from trianglesolver import solve, degree
import albumentations as A
import torch
import random

def load(path, aug = None):
    i = cv2.imread(path)
    i = cv2.cvtColor(i,cv2.COLOR_BGR2RGB)
    if aug:
        try:
            i = aug(image = i)['image']
        except ValueError:
            i = cv2.rotate(i,cv2.cv2.ROTATE_90_CLOCKWISE)
            i = aug(image = i)['image']
    return i

def rotate_image(mat, angle):
    angle = -angle
    height, width = mat.shape[:2]
    image_center = (width/2, height/2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def mse(mat1,mat2):
    return ((mat1-mat2)**2).mean()

def rotateResize(img,x,y,aug=None):
    bord = np.zeros(img.shape[:2])
    bord[(img.mean(axis = 2) < 24).astype(bool)] = 255
    if y < x and img.shape[0] < img.shape[1] or y > x and img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        bord = cv2.rotate(bord, cv2.cv2.ROTATE_90_CLOCKWISE)
    edges = cv2.Canny(img,200,100)
    rows=[edges[1,:],edges[:,1]]
    indexes = []
    for row in rows:
        s = row
        index = np.where(s==np.amax(s))
        try:
            indexes.append(int(index[0]))
        except:
            indexes.append(int(index[0][0]))
    a = x
    b = indexes[0]
    c = indexes[1]
    try:
        a,ba,c,AA,B,C = solve(c=indexes[1], a =indexes[0], B=90*degree)
        
        if abs(ba-y) < abs(ba-x):
            degr = C/degree
        else:
            degr = C/degree+90
            
        if img.shape[0]-x < 11 or img.shape[1]-y < 11:
            degr = 90-degr
            degr = -degr
        rotImg = rotate_image(img,degr)
        bord_ori = bord.copy()
        bord = rotate_image(bord,degr)
    except AssertionError:
        rotImg = img

    if y < x and rotImg.shape[0] < rotImg.shape[1] or y > x and rotImg.shape[0] > rotImg.shape[1]:
        rotImg = rotate_image(img,96.5-degr)
        bord = rotate_image(bord_ori,96.5-degr)
    if aug:
        rotImg = aug(image = rotImg)['image']
        bord = aug(image=bord)['image']
    
    rotImg = cv2.inpaint(rotImg,np.expand_dims(bord,-1).astype('uint8'),3,cv2.INPAINT_TELEA)
    #rotImg = correctImg(rotImg,bord)
    return rotImg, bord

def preprocess(mode,n:int):
    start = load(f'data/{mode}/Corrupted_Images/{n}/start_image.jpg')
    last = load(f'data/{mode}/Corrupted_Images/{n}/last_image.jpg')
    if mode == 'train' or mode == 'val':
        label = load(f'data/{mode}/Labels/{n}.jpg')
    else:
        label = None
        
    x = start.shape[0]
    y = start.shape[1]
    aug = A.Compose([A.CenterCrop(p=1, height=x-4, width=y-4),
                    A.Resize(x,y,always_apply = True)])
    
    paths = [i.replace('\\','/') for i in glob(f'data/{mode}/Corrupted_Images/{n}/*.jpg') if not('last_image.jpg' in i or 'start_image.jpg' in i)]
    images = []
    borders = []
    
    for path in paths:
        i = load(path)
        i, b = rotateResize(i,x,y,aug)
        images.append(i)
        borders.append(b)
        
    return (start, last, label), np.array(images), np.array(borders)