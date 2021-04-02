import cv2
import math
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from trianglesolver import solve, degree
import albumentations as A
import torch
import random
from load import *

#TOP, LEFT, BOTTOM, RIGHT
left = 0
bottom = 1
right = 2
top = 3

idx_to_num = {left:'left',bottom:'bottom',right:'right',top:'top'}

def getBorders(img,gap,n=None):
    if n != None:
        border = [img[:,gap],img[img.shape[0]-gap-1,:],img[:,img.shape[1]-gap-1,:],img[gap,:]]
        return border[n]
    else:
        border = [img[:,gap]]
        for i in range(3):
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
            border.append(img[:,gap])
            
        return border

def compareBorder(img,c):
    img = img.T
    scores = []
    for u, border in enumerate(c):
        border = border.T
        if border.shape[1] == img.shape[1]:
            mask = np.array([(border.mean(axis = 0)>48).astype(bool)]*3)
            #border = cv2.Canny(border.astype('uint8'),0,30)
            #img = cv2.Canny(img.astype('uint8'),0,30)
            mse_out = []
            img_flip = np.flip(img,1)
            mse = min(((border-img)**2).mean(),((border-img_flip)**2).mean())
            #mse = ((border-img**2)).mean()
            mse_out.append(mse)
            scores.append(min(mse_out))
        else:
            scores.append(100000)
            
    return scores

def lowest(images,q,down= False):
    scores = []
    for i in images:
        c = getBorders(i,1)
        if down:
            c = [np.flip(i,1) for i in c]
        scores.append(compareBorder(q,c))
    scores = np.array(scores)
    scores[scores == 0] = 100000
    indexes = np.where(scores.min() == scores)
    if indexes[0].shape[0] > 1:
        return int(indexes[0][0]), int(indexes[1][0]),scores.min()
    else:
        return int(indexes[0]), int(indexes[1]),scores.min()
    
def findN(start_image,images,x_num,d = 0,last = None,remove = True,asList = False):
    outImage = start_image
    last_image = start_image
    out_images = [start_image]
    x, y = start_image.shape[:2]
    axis = 0
    if d == 0 or d == 2:
        axis = 1
    for i in range(x_num-1):
        if images.shape[0] != 0:
            img = getBorders(last_image,1,d)
            image_index, rot, confidence = lowest(images,img,False if d== bottom or d == top else True)
            last_image = images[image_index]
            if remove:
                images = np.delete(images,image_index,axis = 0)
            
            if d== 0 or d == 2:
                rot = (rot+d-2)%4
            else:
                rot = (rot+d)%4
            if x == y:
                if rot > 0:
                    last_image = rotate_image(last_image,rot*90)
            elif rot == 2:
                last_image = rotate_image(last_image,180)
            out_images.append(last_image)
            if True:
                if d == 0 or d== 3:
                    outImage = np.concatenate((last_image,outImage), axis = axis)
                else:
                    outImage = np.concatenate((outImage,last_image), axis = axis)
            else:
                outImage = last_image
        else:
            outImage = np.concatenate((outImage,last), axis = axis)
    outImage = outImage.astype(int)
    if not remove:
        return outImage,images,confidence,image_index
    if asList:
        return out_images,images
    return outImage,images

def findN2(start_image,images,x_num,d = 0,last = None,remove = True):
    outImage = start_image
    last_image = start_image
    x, y = start_image.shape[:2]
    axis = 0
    if d == 0 or d == 2:
        axis = 1
    for i in range(x_num-1):
        if images.shape[0] != 0:
            img = getBorders(last_image,1,d)
            image_index, rot, confidence = lowest(images,img,False if d== bottom or d == top else True)
            last_image = images[image_index]
            if remove:
                images = np.delete(images,image_index,axis = 0)
            
            if d== 0 or d == 2:
                rot = (rot+d-2)%4
            else:
                rot = (rot+d)%4
            if x == y:
                if rot > 0:
                    last_image = rotate_image(last_image,rot*90)
            elif rot == 2:
                last_image = rotate_image(last_image,180)
                
            if x_num != 2:
                if d == 0 or d== 3:
                    outImage = np.concatenate((last_image,outImage), axis = axis)
                else:
                    outImage = np.concatenate((outImage,last_image), axis = axis)
            else:
                outImage = last_image
        else:
            outImage = np.concatenate((outImage,last), axis = axis)
    outImage = outImage.astype(int)
    if not remove:
        return outImage,images,confidence,image_index
    return outImage,images,confidence

def fillup(out):
    if out.shape[1] != 512:
        add = np.array([getBorders(out,0,right)]*(512-out.shape[1])).reshape(out.shape[0],512-out.shape[1],3)
        out = np.concatenate((out,add),axis = 1)

    if out.shape[0] != 512:
        add = np.array([getBorders(out,0,bottom)]*(512-out.shape[0])).reshape(512-out.shape[0],512,3)
        out = np.concatenate((out,add),axis = 0)
    return out

def arangeNew(mat):
    out = []
    for line in mat:
        c = []
        for col in line:
            c.append(col)
        c = np.concatenate(c,axis = 0)
        out.append(c)
    return np.concatenate(out,axis = 1).astype(int)

def getNei(out):
    x_num, y_num = out.shape[:2]
    filled = np.zeros((x_num,y_num))

    for i in range(x_num):
        for j in range(y_num):
            if out[i][j].sum() != 0:
                filled[i][j] = 1
    nCounts = np.zeros((x_num,y_num))
    for i in range(x_num):
        for j in range(y_num):
            if filled[i][j] == 0:
                count = 0
                if i < x_num-1:
                    count += filled[i+1][j]
                if i > 0:
                    count += filled[i-1][j]
                if j < y_num-1:
                    count += filled[i][j+1]

                if j > 0:
                    count += filled[i][j-1]
                nCounts[i][j] = count
    return filled,nCounts

def getImgNei(out,x,y,d):
    if d == left:
        return out[x-1][y]
    if d == right:
        return out[x+1][y]
    if d == top:
        return out[x][y-1]
    if d == bottom:
        return out[x][y+1]