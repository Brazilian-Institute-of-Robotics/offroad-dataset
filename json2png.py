#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:34:31 2018

@author: nelson
"""
from __future__ import absolute_import, division, print_function

import argparse
import base64
import json
import os
import os.path as osp
import warnings
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import PIL.Image
from labelme import utils
from imgaug import augmenters as iaa
import imgaug as ia
import skimage


def augmentationImage(img, lbl, depth_map):
    #seq = iaa.Sequential([iaa.Clouds(),iaa.Fog(),iaa.Snowflakes()])
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Crop(percent=(0, 0.1)), # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        # Strengthen or weaken the contrast in each image.
        iaa.ContrastNormalization((0.75, 1.5)),
        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # Make some images brighter and some darker.
        # In 20% of all cases, we sample the multiplier once per channel,
        # which can end up changing the color of the images.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            rotate=(-25, 25),
            shear=(-8, 8)
        )
    ], random_order=True) # apply augmenters in random order
    
    
    segmap = ia.SegmentationMapOnImage(lbl, shape=img.shape, nb_classes=1+np.max(lbl))
    #    images = load_batch(batch_idx)  # you have to implement this function
    depthmap = ia.SegmentationMapOnImage(depth_map, shape=depth_map.shape, nb_classes=256)
    
    
    seq_det = seq.to_deterministic()
    #    images_aug.append(seq_det.augment_image(image))
    #    segmaps_aug.append(seq_det.augment_segmentation_maps([segmap])[0])
        
    images_aug = seq_det.augment_images([img])  # done by the library
    labels_aug = seq_det.augment_segmentation_maps([segmap])
    depthmap_aug = seq_det.augment_segmentation_maps([depthmap])
    #    train_on_images(images_aug)  # you have to implement this function

    return [images_aug[0], labels_aug[0].get_arr_int(), depthmap_aug[0].get_arr_int()]
    
    
def writeimages(img, lbl, depth_map, jsonFile, sufix, out_images_dir, out_labels_dir, out_labels_raw_dir, debug_dir, label_name_to_value, args):
    #Plot visualization if in debug mode
    if args.debug:
        label_names = [None] * (max(label_name_to_value.values()) + 1)
        for name, value in label_name_to_value.items():
            label_names[value] = name
        lbl_viz = utils.draw_label(lbl, img, label_names)
        PIL.Image.fromarray(lbl_viz).save(osp.join(debug_dir, jsonFile.replace('./','').replace('../','').replace('/','_').replace('.json','-'+sufix+'-label_viz.png')))

    jsonFileName = jsonFile[jsonFile.rfind('/')+1:]

    #Save image
    if out_images_dir is not None:
        PIL.Image.fromarray(img).save(
            osp.join(out_images_dir, jsonFileName.replace('.json','.jpg'))
        )
    #Save labels
    utils.lblsave(
        osp.join(out_labels_dir, jsonFileName.replace('.json','-'+sufix+'-label.png')), lbl
    )
    #Save labels with raw annotation
    PIL.Image.fromarray(lbl.astype(dtype=np.uint8)).save(
        osp.join(out_labels_raw_dir, jsonFileName.replace('.json','-'+sufix+'-label_raw.png'))
    )
    if  args.depth:
        #Save depth image 
#        PIL.Image.fromarray(depth_map.astype(dtype=np.uint8)).save(
#            osp.join(out_images_dir, jsonFile.replace('.json','-depth.png'))
#        )
        PIL.Image.fromarray(depth_map.astype(dtype=np.uint8)).save(
            osp.join(out_images_dir, jsonFileName.replace('.json','-'+sufix+'-depth.png'))
        )


def convertListOfFiles(listOfFile, sufix,  out_dir, label_name_to_value, args, augmentation=False):
    #Create label _background_
    
    for jsonFile in listOfFile:
        #Load json file
        print("Converting ",jsonFile)
        data = json.load(open(jsonFile))
        
        #Load image
        if data['imageData']:
            imageData = data['imageData']
        else:
            imagePath = os.path.join(os.path.dirname(jsonFile), data['imagePath'])
            with open(imagePath, 'rb') as f:
                imageData = f.read()
                imageData = base64.b64encode(imageData).decode('utf-8')
        img = utils.img_b64_to_arr(imageData)
        depth_map = []
        if  args.depth:
            imagePath = os.path.join(os.path.dirname(jsonFile), data['imagePath'])
            depth_map = skimage.io.imread(imagePath.replace('.jpg', '.exr'), plugin="tifffile")
            depth_map = np.uint8(depth_map*255)#change to png scale

    
        #Create new label into label_name_to_value if it don't exists
        for shape in sorted(data['shapes'], key=lambda x: x['label']):
            #ignore instance id if semantic segmentation
            if args.semantic_segmentation :
                shape['label'] = shape['label'].split(sep='-')[0]
            #Get label name
            label_name = shape['label']
            #Create and get value of position if it don't exists
            if label_name not in label_name_to_value:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value
    
        #put road and car to be render first
        shapes = data['shapes']
        car_index = next((index for (index, d) in enumerate(shapes) if d["label"] == "car"), None)
        if car_index!=None:
            shapes.append(shapes.pop(car_index))
        road_index = next((index for (index, d) in enumerate(shapes) if d["label"] == "road"), None)
        if road_index!=None:
            shapes.append(shapes.pop(road_index))
        shapes.reverse()
        
        #Create array os labels and shapes
        lbl = utils.shapes_to_label(img.shape, shapes, label_name_to_value)

        
        #TODO: Implement scale and crop
        if args.scale is not None:
            warnings.warn("Feature scale is not implemented yet!")
#                   w, h = img.size
#                   filename, extension = name.split(".")
#                   img.resize((int(w/4), int(h/4))).save(root + '/' + 'resized_4_' + name)

        json_path = args.json_path
        new_path = jsonFile[:jsonFile.rfind('/')]
        new_path = new_path.replace(json_path, out_dir)

        out_images_dir = None if json_path == out_dir else new_path
        out_labels_dir = new_path
        out_labels_raw_dir = new_path

        writeimages(img, lbl, depth_map, jsonFile, sufix,  out_images_dir, out_labels_dir, out_labels_raw_dir, out_dir, label_name_to_value, args)
        if augmentation:
            [img, label, depth_map] = augmentationImage(img, lbl, depth_map)
            writeimages(img, label, depth_map, 'aug-'+jsonFile, sufix, out_images_dir, out_labels_dir, out_labels_raw_dir, out_dir, label_name_to_value, args)

    
        print('Saved to: %s' % out_dir)
    return label_name_to_value




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-json_path', default='./')
    parser.add_argument('-d', '--debug', action='store_true',  default=True)
    parser.add_argument('-ss','--semantic_segmentation', action='store_true',  default=True)
    parser.add_argument('-da','--data_augmentation', help='enable data augmentation', action='store_true',  default=False)
    parser.add_argument('-dpt','--depth', help='enable depth mode', action='store_true',  default=False)


    # parser.add_argument('json_path')
    # parser.add_argument('-d', '--debug', help='enable the generation of images for debug', action='store_true')
    # parser.add_argument('-ss','--semantic_segmentation', help='enable semantic segmentation instead of entity segmentation', action='store_true')
    # parser.add_argument('-da','--data_augmentation', help='enable data augmentation', action='store_true')
    # parser.add_argument('-dpt','--depth', help='enable depth mode', action='store_true')


    parser.add_argument('-vs', '--test_split', help='define the percentage to be used as train. It shuold be between 0.0 and 1.0 (default value is 0.20)', type=float, default=0.20)
    parser.add_argument('-o', '--out', help='define the output path (default is ./dataset_out)', default=None)
    parser.add_argument('-s', '--scale', help='apply scale for the images (it\'s not implemented yet)', type=float)
    args = parser.parse_args()

    print(args.json_path)
    json_path = args.json_path
    
    
    ############################Create output dir##############################
    if args.out is None:
        out_dir = osp.basename(json_path).replace('.', '_')
        out_dir = osp.join(osp.dirname(json_path), out_dir)
    else:
        out_dir = args.out
    # if not osp.exists(out_dir):
    #     os.mkdir(out_dir)
    # if not osp.exists(out_dir+'/train'):
    #     os.mkdir(out_dir+'/train')
    # if not osp.exists(out_dir+'/test'):
    #     os.mkdir(out_dir+'/test')
    # #Create output dir train
    # out_train_images_dir = out_dir+'/train/images'
    # if not osp.exists(out_train_images_dir):
    #     os.mkdir(out_train_images_dir)
    # out_train_labels_dir = out_dir+'/train/labels'
    # if not osp.exists(out_train_labels_dir):
    #     os.mkdir(out_train_labels_dir)
    # out_train_labels_raw_dir = out_dir+'/train/labels_raw'
    # if not osp.exists(out_train_labels_raw_dir):
    #     os.mkdir(out_train_labels_raw_dir)
    # #Create output dir test
    # out_test_images_dir = out_dir+'/test/images'
    # if not osp.exists(out_test_images_dir):
    #     os.mkdir(out_test_images_dir)
    # out_test_labels_dir = out_dir+'/test/labels'
    # if not osp.exists(out_test_labels_dir):
    #     os.mkdir(out_test_labels_dir)
    # out_test_labels_raw_dir = out_dir+'/test/labels_raw'
    # if not osp.exists(out_test_labels_raw_dir):
    #     os.mkdir(out_test_labels_raw_dir)
    
    ################################Find Files#################################
    #Load labels path and generate images path 
    labelFilesPath = glob.glob(json_path + '/**/*.json', recursive=True)
    imageFilesPath = [labelPath.replace('.json', '.jpg') for labelPath in labelFilesPath]
    print(str(len(labelFilesPath)) + "images with annotation was found.")

    #Split train test
    imageTrain, imageTest, labelTrain, labelTest = train_test_split(
          imageFilesPath, labelFilesPath, test_size=args.test_split, random_state=42)
    
    ###############################Convert Files###############################
    label_name_to_value = {'_background_': 0}

    label_name_to_value = convertListOfFiles(labelTrain, 'train', out_dir,
        label_name_to_value, args, augmentation=args.data_augmentation)

    label_name_to_value = convertListOfFiles(labelTest, 'test', out_dir,
        label_name_to_value, args, augmentation=False)
    
    #Generate and save labels txt        
    with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
        for name in sorted(label_name_to_value):
            f.write(name + '\n')
    


if __name__ == '__main__':
    
    main()
