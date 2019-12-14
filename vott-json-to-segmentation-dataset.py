#! /usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################
## 
## https://github.com/divamgupta/image-segmentation-keras
## に使う学習データを作成するスクリプト。
##
################################################################################

import numpy as np
import cv2
import json
from collections import OrderedDict
import pprint
import sys
import os
import urllib.parse
import random
import glob

# タグとラベルの対応
tags = {
    "Low intensity intima": 1,
}

# 各クラスに割り当てる BGR カラー値。
# https://github.com/divamgupta/image-segmentation-keras#preparing-the-data-for-training
# B チャンネルの値がクラスのラベルとして使われる。
colors = {
    1: (1, 0, 255),
}

# 出力画像サイズ
# https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html
#   Choosing the input size
#   Apart from choosing the architecture of the model, choosing the model input size is 
#   also very important. If there are a large number of objects in the image, the input 
#   size shall be larger. In some cases, if the input size is large, the model should 
#   have more layers to compensate. The standard input size is somewhere from 200x200 
#   to 600x600. A model with a large input size consumes more GPU memory and also would 
#   take more time to train.
size = 608 # 32 で割り切れる数にしておく

# 出力先ディレクトリ
output_x_train = './dataset/train_images/'
output_y_train = './dataset/train_annotations/'
output_x_test = './dataset/test_images/'
output_y_test = './dataset/test_annotations/'
output_preview = './preview/'
os.makedirs(output_x_train, exist_ok=True)
os.makedirs(output_y_train, exist_ok=True)
os.makedirs(output_x_test, exist_ok=True)
os.makedirs(output_y_test, exist_ok=True)
os.makedirs(output_preview, exist_ok=True)

# トレーニング用とテスト用の比率
split_test  = 0.1
split_train = 0.9
random.seed(0)

def read_json(filename, source_dir):
    fp = open(filename)
    _d = json.load(fp)

    _asset = _d['asset']
    id     = _asset['id']
    name   = _asset['name']
    path   = _asset['path']
    width  = _asset['size']['width']
    height = _asset['size']['height']

    # URLエンコードされてるのでデコード
    name = urllib.parse.unquote(name)

    if (width > height):
        scale = size / width
    else:
        scale = size / height

    shape = [size, size, 3] # BGR
    img = np.zeros(shape, dtype=np.uint8)

    for _region in _d['regions']:
        _tags    = _region['tags']
        _points  = _region['points']

        if (len(_tags) != 1):
            sys.exit('len(tags) is not 1')

        _tag = _tags[0]
        tag_index = tags[_tag]
        tag_color = colors[tag_index]

        points = [[p['x'] * scale, p['y'] * scale] for p in _points]
        points = np.array(points).astype(np.int32)
        img = cv2.fillPoly(img, pts=[points], color=tag_color)
    
    if random.random() <= (split_train / (split_train + split_test)):
        output_y = output_y_train
        output_x = output_x_train
    else:
        output_y = output_y_test
        output_x = output_x_test

    # アノテーション画像
    filename_annotation = output_y + id + '.png'
    cv2.imwrite(filename_annotation, img)
    # print("wrote: " + filename_annotation)

    # RGB 画像（サイズ統一）
    filename_orig = source_dir + name

    img_orig = cv2.imread(filename_orig)
    img_orig = cv2.resize(img_orig, None, fx=scale, fy=scale)
    _h, _w = img_orig.shape[:2]
    img_dest = np.zeros(shape, dtype=np.uint8)
    img_dest[0:_h, 0:_w] = img_orig

    filename_input = output_x + id + '.png' # JPEG 圧縮で劣化するのは嫌なので
    cv2.imwrite(filename_input, img_dest)
    # print('wrote: ' + filename_input)

    # プレビュー画像
    img_preview = cv2.addWeighted(img_dest, 1.0, img, 0.25, 0)
    filename_preview = output_preview + id + '.jpg'
    cv2.imwrite(filename_preview, img_preview)
    # print('wrote: ' + filename_preview)

    print(id + ' : ' + filename_orig)

# 実行用
if __name__ == '__main__':
    args = sys.argv
    if (len(args) != 3):
        exit('Usage: python3 vott-json-to-segmentation-dataset.py {vott-target-dir} {vott-source-dir}')

    vott_target_dir = args[1]
    vott_source_dir = args[2]

    if (not(os.path.exists(vott_source_dir)) or not(os.path.isdir(vott_source_dir))):
        exit('source-dir does not exists')

    if (not(os.path.exists(vott_target_dir)) or not(os.path.isdir(vott_target_dir))):
        exit('target-dir does not exists')

    for json_file in glob.glob(vott_target_dir + "/*.json"):
        read_json(json_file, vott_source_dir)
