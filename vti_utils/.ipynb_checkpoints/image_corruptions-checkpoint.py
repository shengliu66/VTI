# -*- coding: utf-8 -*-

import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

import skimage as sk
from skimage.filters import gaussian
from io import BytesIO
from wand.image import Image as WandImage
from wand.api import library as wandlibrary
import wand.color as WandColor
import ctypes
from PIL import Image as PILImage
import cv2
from scipy.ndimage import zoom as scizoom
from scipy.ndimage.interpolation import map_coordinates
import warnings

def get_random_bounding_box(
    image_shape, ratio, margin= 0.0
):
    img_h, img_w = image_shape
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y)
    cx = np.random.randint(0 + margin_x, img_w - margin_x)
    y_min = np.clip(cy - cut_h // 2, 0, img_h)
    y_max = np.clip(cy + cut_h // 2, 0, img_h)
    x_min = np.clip(cx - cut_w // 2, 0, img_w)
    x_max = np.clip(cx + cut_w // 2, 0, img_w)
    return y_min, y_max, x_min, x_max

class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)

def plasma_fractal(mapsize=1024, wibbledecay=3):
    """
    Generate a heightmap using diamond-square algorithm.
    Return square 2d array, side length 'mapsize', of floats in range 0-255.
    'mapsize' must be a power of two.
    """
    assert (mapsize & (mapsize - 1) == 0)
    maparray = np.empty((mapsize, mapsize), dtype=np.float_)
    maparray[0, 0] = 0
    stepsize = mapsize
    wibble = 100

    def wibbledmean(array):
        return array / 4 + wibble * np.random.uniform(-wibble, wibble, array.shape)

    def fillsquares():
        """For each square of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        cornerref = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        squareaccum = cornerref + np.roll(cornerref, shift=-1, axis=0)
        squareaccum += np.roll(squareaccum, shift=-1, axis=1)
        maparray[stepsize // 2:mapsize:stepsize,
        stepsize // 2:mapsize:stepsize] = wibbledmean(squareaccum)

    def filldiamonds():
        """For each diamond of points stepsize apart,
           calculate middle value as mean of points + wibble"""
        mapsize = maparray.shape[0]
        drgrid = maparray[stepsize // 2:mapsize:stepsize, stepsize // 2:mapsize:stepsize]
        ulgrid = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
        ldrsum = drgrid + np.roll(drgrid, 1, axis=0)
        lulsum = ulgrid + np.roll(ulgrid, -1, axis=1)
        ltsum = ldrsum + lulsum
        maparray[0:mapsize:stepsize, stepsize // 2:mapsize:stepsize] = wibbledmean(ltsum)
        tdrsum = drgrid + np.roll(drgrid, 1, axis=1)
        tulsum = ulgrid + np.roll(ulgrid, -1, axis=0)
        ttsum = tdrsum + tulsum
        maparray[stepsize // 2:mapsize:stepsize, 0:mapsize:stepsize] = wibbledmean(ttsum)

    while stepsize >= 2:
        fillsquares()
        filldiamonds()
        stepsize //= 2
        wibble /= wibbledecay

    maparray -= maparray.min()
    return maparray / maparray.max()

def clipped_zoom(img, zoom_factor):
    h = img.shape[0]
    w = img.shape[1]
    #print("h:",h)
    #print("w:",w)
    
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / zoom_factor))
    cw = int(np.ceil(w / zoom_factor))
    #print("ch:",ch)
    #print("cw:",cw)

    top1 = (h - ch) // 2
    top2 = (w - cw) // 2
    img = scizoom(img[top1:top1 + ch, top2:top2 + cw], (zoom_factor, zoom_factor, 1), order=1)
    #print("img:", img.shape)
    # trim off any extra pixels
    trim_top1 = (img.shape[0] - h) // 2
    trim_top2 = (img.shape[1] - w) // 2
    
    temp = img[trim_top1:(trim_top1 + h), trim_top2:(trim_top2 + w)]
    #print("temp:", temp.shape)

    return img[trim_top1:(trim_top1 + h), trim_top2:(trim_top2 + w)]
        
def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)
 
def getFileList(dir,Filelist, ext=None):
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)   
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir=os.path.join(dir,s)
            getFileList(newDir, Filelist, ext)
    return Filelist

 
def gaussian_noise(x, severity):
    c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]
    x = np.array(x) / 255.
    return np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255

def shot_noise(x, severity):
    c = [60, 25, 12, 5, 3][severity - 1]
    x = np.array(x) / 255.
    return np.clip(np.random.poisson(x * c) / c, 0, 1) * 255

def impulse_noise(x, severity):
    c = [.03, .06, .09, 0.17, 0.27][severity - 1]

    x = sk.util.random_noise(np.array(x) / 255., mode='s&p', amount=c)
    return np.clip(x, 0, 1) * 255

def speckle_noise(x, severity):
    c = [.15, .2, 0.35, 0.45, 0.6][severity - 1]

    x = np.array(x) / 255.
    return np.clip(x + x * np.random.normal(size=x.shape, scale=c), 0, 1) * 255

def gaussian_blur(x, severity):
    c = [1, 2, 3, 4, 6][severity - 1]

    x = gaussian(np.array(x) / 255., sigma=c, channel_axis=-1)
    return np.clip(x, 0, 1) * 255

def defocus_blur(x, severity):
    c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

    x = np.array(x) / 255.
    kernel = disk(radius=c[0], alias_blur=c[1])

    channels = []
    for d in range(3):
        channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
    channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

    return np.clip(channels, 0, 1) * 255

def zoom_blur(x, severity=3):
    
    #print("x:",x.shape)
    c = [np.arange(1, 1.11, 0.01),
         np.arange(1, 1.16, 0.01),
         np.arange(1, 1.21, 0.02),
         np.arange(1, 1.26, 0.02),
         np.arange(1, 1.33, 0.03)][severity - 1]

    x = (np.array(x) / 255.).astype(np.float32)
    out = np.zeros_like(x)
    for zoom_factor in c:
        #print(zoom_factor)
        tmp= clipped_zoom(x, zoom_factor)
        #print("tmp:", tmp.shape)
        out = tmp+out

    x = (x + out) / (len(c) + 1)
    return np.clip(x, 0, 1) * 255

def fog(x, severity):
    h=x.shape[0]
    w= x.shape[1]
    c = [(1.5, 2), (2, 2), (2.5, 1.7), (2.5, 1.5), (3, 1.4)][severity - 1]

    x = np.array(x) / 255.
    max_val = x.max()
    tmp= c[0] * plasma_fractal(wibbledecay=c[1])[:h, :w][..., np.newaxis]
    x = x+tmp
    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255

def frost_224(x, severity=5):
    
    #print("x:", x.shape)
    h=x.shape[0]
    w= x.shape[1]
    
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][severity - 1]
    idx = np.random.randint(5)
    #print("idx:",idx)
    #filename = ['frost1.png', 'frost2.png', 'frost3.png', 'frost4.jpg', 'frost5.jpg', 'frost6.jpg'][idx]
    filename = ['frost1.png', 'frost2.png','frost3.png', 'frost4.jpg', 'frost5.jpg', 'frost6.jpg'][idx]
    #print("filename:",filename)
    frost = cv2.imread(filename)
    #print("frost:", frost.shape)
    
    h_f= frost.shape[0]
    w_f= frost.shape[1]
    
    
    if (h>h_f and w>w_f):
        img_h, img_w = np.random.randint(0, h - h_f), np.random.randint(0, w - w_f)
        x = x[img_h:img_h + h_f, img_w:img_w + w_f][..., [2, 1, 0]]
        x_start, y_start = 0,0
        frost = frost[x_start:x_start + h_f, y_start:y_start + w_f][..., [2, 1, 0]]
        
    elif (h>h_f and w<=w_f):
        img_h, img_w = np.random.randint(0, h - h_f), 0
        x = x[img_h:img_h + h_f, img_w:img_w + w][..., [2, 1, 0]]
        x_start, y_start = 0,0
        frost = frost[x_start:x_start + h_f, y_start:y_start + w][..., [2, 1, 0]]
    
    elif (h<=h_f and w>w_f):
        img_h, img_w = 0, np.random.randint(0, w - w_f)
        x = x[img_h:img_h + h, img_w:img_w + w_f][..., [2, 1, 0]]
        x_start, y_start = 0,0
        frost = frost[x_start:x_start + h, y_start:y_start + w_f][..., [2, 1, 0]]
    
    else:
        img_h, img_w = 0, 0
        x = x[img_h:img_h + h, img_w:img_w + w][..., [2, 1, 0]]
        x_start, y_start = 0,0
        frost = frost[x_start:x_start + h, y_start:y_start + w][..., [2, 1, 0]]
    
    #print("frost:", frost.shape)
        

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)

def snow_224(x, severity=1):
    
    #print("x:", x.shape)
    h=x.shape[0]
    w= x.shape[1]
    
    if h>224 and w>224:
        img_h, img_w = np.random.randint(0, h - 224), np.random.randint(0, w - 224)
    else:
        img_h, img_w = 0, 0
    #x = x[img_h:img_h + 224, img_w:img_w + 224][..., [2, 1, 0]]
    x = x[img_h:img_h + min(224,h), img_w:img_w + min(224,w)][..., [2, 1, 0]]
    #print("x:", x.shape)
    
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
                              cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]
    
    if h>224 and w>224:
        x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(224, 224, 1) * 1.5 + 0.5)
    else:
        x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(min(224,h), min(224,w), 1) * 1.5 + 0.5)
        
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255

def spatter(x, severity):
    c = [(0.65, 0.3, 4, 0.69, 0.6, 0),
         (0.65, 0.3, 3, 0.68, 0.6, 0),
         (0.65, 0.3, 2, 0.68, 0.5, 0),
         (0.65, 0.3, 1, 0.65, 1.5, 1),
         (0.67, 0.4, 1, 0.65, 1.5, 1)][severity - 1]
    x = np.array(x, dtype=np.float32) / 255.

    liquid_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    liquid_layer = gaussian(liquid_layer, sigma=c[2])
    liquid_layer[liquid_layer < c[3]] = 0
    if c[5] == 0:
        liquid_layer = (liquid_layer * 255).astype(np.uint8)
        dist = 255 - cv2.Canny(liquid_layer, 50, 150)
        dist = cv2.distanceTransform(dist, cv2.DIST_L2, 5)
        _, dist = cv2.threshold(dist, 20, 20, cv2.THRESH_TRUNC)
        dist = cv2.blur(dist, (3, 3)).astype(np.uint8)
        dist = cv2.equalizeHist(dist)
        #     ker = np.array([[-1,-2,-3],[-2,0,0],[-3,0,1]], dtype=np.float32)
        #     ker -= np.mean(ker)
        ker = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        dist = cv2.filter2D(dist, cv2.CV_8U, ker)
        dist = cv2.blur(dist, (3, 3)).astype(np.float32)

        m = cv2.cvtColor(liquid_layer * dist, cv2.COLOR_GRAY2BGRA)
        m /= np.max(m, axis=(0, 1))
        m *= c[4]

        # water is pale turqouise
        color = np.concatenate((175 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1]),
                                238 / 255. * np.ones_like(m[..., :1])), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_BGR2BGRA)
        x = cv2.cvtColor(x, cv2.COLOR_BGR2BGRA)

        return cv2.cvtColor(np.clip(x + m * color, 0, 1), cv2.COLOR_BGRA2BGR) * 255
    else:
        m = np.where(liquid_layer > c[3], 1, 0)
        m = gaussian(m.astype(np.float32), sigma=c[4])
        m[m < 0.8] = 0
        #         m = np.abs(m) ** (1/c[4])

        # mud brown
        color = np.concatenate((63 / 255. * np.ones_like(x[..., :1]),
                                42 / 255. * np.ones_like(x[..., :1]),
                                20 / 255. * np.ones_like(x[..., :1])), axis=2)

        color *= m[..., np.newaxis]
        x *= (1 - m[..., np.newaxis])

        return np.clip(x + color, 0, 1) * 255

def contrast(x, severity, ratio=0.3):

    c = [0.4, .3, .2, .1, .0][severity - 1]
    # y_min, y_max, x_min, x_max = get_random_bounding_box(x.shape[-3:-1], ratio=ratio, margin=0)

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    # x[y_min:y_max, x_min:x_max, :] = (x[y_min:y_max, x_min:x_max, :] - means) * c + means
    x  = (x  - means) * c + means
    return np.clip(x, 0, 1) * 255 #np.clip((x - means) * c + means, 0, 1) * 255

def get_edge_bounding_box(image_shape, k):
    img_h, img_w = image_shape
    
    # Bounding box for the top edge
    top_y_min, top_y_max = 0, k
    top_x_min, top_x_max = 0, img_w
    
    # Bounding box for the bottom edge
    bottom_y_min, bottom_y_max = img_h - k, img_h
    bottom_x_min, bottom_x_max = 0, img_w
    
    # Bounding box for the left edge
    left_y_min, left_y_max = 0, img_h
    left_x_min, left_x_max = 0, k
    
    # Bounding box for the right edge
    right_y_min, right_y_max = 0, img_h
    right_x_min, right_x_max = img_w - k, img_w
    
    return (top_y_min, top_y_max, top_x_min, top_x_max), \
           (bottom_y_min, bottom_y_max, bottom_x_min, bottom_x_max), \
           (left_y_min, left_y_max, left_x_min, left_x_max), \
           (right_y_min, right_y_max, right_x_min, right_x_max)

def contrast_edges(x, severity, k):
    c = [0.4, .3, .2, .1, .0][severity - 1]
    edges = get_edge_bounding_box(x.shape[-3:-1], k)
    img_h, img_w = x.shape[-3:-1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    
    indices = []

    for edge in edges:
        y_min, y_max, x_min, x_max = edge
        for y in range(int(y_min), int(y_max)):
            for x_idx in range(int(x_min), int(x_max)):
                index = y * img_w + x_idx
                indices.append(index)
                x[y, x_idx, :] = (x[y, x_idx, :] - means) * c + means
    
    return np.clip(x, 0, 1) * 255, indices

def brightness(x, severity):
    c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255

def saturate(x, severity):
    c = [(0.3, 0), (0.1, 0), (2, 0), (5, 0.1), (20, 0.2)][severity - 1]

    x = np.array(x) / 255.
    x = sk.color.rgb2hsv(x)
    x[:, :, 1] = np.clip(x[:, :, 1] * c[0] + c[1], 0, 1)
    x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255

def pixelate(x, severity):
    
    h=x.shape[0]
    w= x.shape[1]   
    c = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]

    #x = x.resize((int(224 * c), int(224 * c)), PILImage.BOX)
    x= cv2.resize(x, (int(224 * c),int(224 * c)), interpolation = cv2.INTER_AREA)
    #x = x.resize((224, 224), PILImage.BOX)
    x= cv2.resize(x, (h,w), interpolation = cv2.INTER_AREA)

    return x

def elastic_transform(image, severity):
    c = [(244 * 2, 244 * 0.7, 244 * 0.1),   # 244 should have been 224, but ultimately nothing is incorrect
         (244 * 2, 244 * 0.08, 244 * 0.2),
         (244 * 0.05, 244 * 0.01, 244 * 0.02),
         (244 * 0.07, 244 * 0.01, 244 * 0.02),
         (244 * 0.12, 244 * 0.01, 244 * 0.02)][severity - 1]

    image = np.array(image, dtype=np.float32) / 255.
    shape = image.shape
    shape_size = shape[:2]

    # random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size,
                       [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                   c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
    dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
    return np.clip(map_coordinates(image, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255

def glass_blur(x, severity=1):
    
    h = x.shape[0]
    w = x.shape[1]
    
    # sigma, max_delta, iterations
    c = [(0.7, 1, 2), (0.9, 2, 1), (1, 2, 3), (1.1, 3, 2), (1.5, 4, 2)][severity - 1]

    x = np.uint8(gaussian(np.array(x) / 255., sigma=c[0], channel_axis=-1) * 255)

    # locally shuffle pixels
    for i in range(c[2]):
        for h in range(min(h,224) - c[1], c[1], -1):
            for w in range(min(w,224) - c[1], c[1], -1):
                dx, dy = np.random.randint(-c[1], c[1], size=(2,))
                h_prime, w_prime = h + dy, w + dx
                # swap
                x[h, w], x[h_prime, w_prime] = x[h_prime, w_prime], x[h, w]

    return np.clip(gaussian(x / 255., sigma=c[0], channel_axis=-1), 0, 1) * 255

def motion_blur_ori(x, severity=1):
    c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][severity - 1]
    
    x = np.array(x, dtype=np.float32) / 255.
    x = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])
    #print("input:", x.shape)
    x = PILImage.fromarray((np.clip(x.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    #print(x.size)

    output = BytesIO()
    x.save(output, format='PNG')
    x = MotionImage(blob=output.getvalue())

    #x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))
    x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))
    #print("x.motion_blur:", x.size)

    x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                     cv2.IMREAD_UNCHANGED)
    #print("cv2.imdecode:", x.shape)
    
    #tmp = np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)
    #print("tmp:", tmp.shape)

    #if x.shape != (224, 224):
    #    return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
    #else:  # greyscale to RGB
    #    return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)
    return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)

def motion_blur(x, severity=1):
    from imagenet_c import corrupt
    img_new = corrupt(x, corruption_name='motion_blur', severity=severity)
    return img_new

def jpeg_compression(x, severity=1):
    from imagenet_c import corrupt
    img_new = corrupt(x, corruption_name='jpeg_compression', severity=severity)
    return img_new

def snow(x, severity=1):
    
    h=x.shape[0]
    w= x.shape[1]
    
    #x = x[..., [2, 1, 0]]
    #print("x:", x.shape)
    
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8),
         (0.2, 0.3, 2, 0.5, 12, 4, 0.7),
         (0.55, 0.3, 4, 0.9, 12, 8, 0.7),
         (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65),
         (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])  # [:2] for monochrome

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = PILImage.fromarray((np.clip(snow_layer.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
    output = BytesIO()
    snow_layer.save(output, format='PNG')
    snow_layer = MotionImage(blob=output.getvalue())

    snow_layer.motion_blur(radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))

    snow_layer = cv2.imdecode(np.fromstring(snow_layer.make_blob(), np.uint8),
                              cv2.IMREAD_UNCHANGED) / 255.
    snow_layer = snow_layer[..., np.newaxis]
    
    x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(h, w, 1) * 1.5 + 0.5)
        
    return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255

def frost(x, severity=5):
    h=x.shape[0]
    w= x.shape[1]
    #print("h:",h)
    #print("w:",w)
    
    #x = x[..., [2, 1, 0]]
    #print("x:", x.shape)
    
    c = [(1, 0.4),
         (0.8, 0.6),
         (0.7, 0.7),
         (0.65, 0.7),
         (0.6, 0.75)][severity - 1]
    idx = np.random.randint(5)
    filename = ['frost1.png', 'frost2.png', 'frost3.png', 'frost4.jpg', 'frost5.jpg', 'frost6.jpg'][idx]
    frost = cv2.imread(filename)
    
    h2=frost.shape[0]
    w2=frost.shape[1]
    #print("h2:",h2)
    #print("w2:",w2)
    # randomly crop and convert to rgb
    
    frost = frost[0:h, 0:w][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost, 0, 255)

def get_corruptions(corruption_name):
    corruptions_dict = {
        'Gaussian Noise': gaussian_noise,
        'Shot Noise': shot_noise,
        'Impulse Noise': impulse_noise,
        'Defocus Blur': defocus_blur,
        'Glass Blur': glass_blur,
        'Motion Blur': motion_blur,
        'Zoom Blur': zoom_blur,
        'Snow': snow,
        'Frost': frost,
        'Fog': fog,
        'Brightness': brightness,
        'Contrast': contrast,
        'Contrast Edges': contrast_edges,
        'Elastic': elastic_transform,
        'Pixelate': pixelate,
        'JPEG': jpeg_compression,
        'Speckle Noise': speckle_noise,
        'Gaussian Blur': gaussian_blur,
        'Spatter': spatter,
        'Saturate': saturate
    }
    return corruptions_dict.get(corruption_name, None)

# # /////////////// End Distortions ///////////////

# import collections

# print('Using CIFAR-10 data')

# d = collections.OrderedDict()
# d['Gaussian Noise'] = gaussian_noise
# d['Shot Noise'] = shot_noise
# d['Impulse Noise'] = impulse_noise
# d['Defocus Blur'] = defocus_blur
# d['Glass Blur'] = glass_blur
# d['Motion Blur'] = motion_blur
# d['Zoom Blur'] = zoom_blur
# d['Snow'] = snow
# d['Frost'] = frost
# d['Fog'] = fog
# d['Brightness'] = brightness
# d['Contrast'] = contrast
# d['Elastic'] = elastic_transform
# d['Pixelate'] = pixelate
# d['JPEG'] = jpeg_compression

# d['Speckle Noise'] = speckle_noise
# d['Gaussian Blur'] = gaussian_blur
# d['Spatter'] = spatter
# d['Saturate'] = saturate


# test_data = dset.CIFAR10('/share/data/vision-greg/cifarpy', train=False)
# convert_img = trn.Compose([trn.ToTensor(), trn.ToPILImage()])


# for method_name in d.keys():
#     print('Creating images for the corruption', method_name)
#     cifar_c, labels = [], []

#     for severity in range(1,6):
#         corruption = lambda clean_img: d[method_name](clean_img, severity)

#         for img, label in zip(test_data.data, test_data.targets):
#             labels.append(label)
#             cifar_c.append(np.uint8(corruption(convert_img(img))))

#     np.save('/share/data/vision-greg2/users/dan/datasets/CIFAR-10-C/' + d[method_name].__name__ + '.npy',
#             np.array(cifar_c).astype(np.uint8))

#     np.save('/share/data/vision-greg2/users/dan/datasets/CIFAR-10-C/labels.npy',
#             np.array(labels).astype(np.uint8))
