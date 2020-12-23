#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 11:59:03 2019

@author: nelson
"""


import glob
import json


countable_classes = ['animal', 'bike', 'bus', 'car', 'cone', 'moto', 'person', 'truck']
def generate_list(path, search_fragment):
    my_list = glob.glob(path+'**.json')
    count=0
    with open(path+'small.txt', 'w') as f:
        for item in my_list:
            if search_fragment in item:
                f.write("%s\n" % item[item.rfind('/')+1:-5])
                count +=1
            elif  any([name in shape['label'] for shape in json.load(open(item))['shapes'] for name in countable_classes]):
                f.write("%s\n" % item[item.rfind('/')+1:-5])
                count +=1
    return count





# root_path = '/home/nelson/projects/da_art_perception/data/dataset'
root_path = './'
count = 0


path = root_path + '/off-road/night/cimatec-industrial/'
search_fragment = '005cc'
count_ = generate_list(path, search_fragment)
print(path+' has ' + str(count_) + ' images')
count += count_

path = root_path + '/off-road/day/cimatec-industrial/'
search_fragment = '001_'
count_ = generate_list(path, search_fragment)
print(path+' has ' + str(count_) + ' images')
count += count_

path = root_path + '/off-road/evening/cimatec-industrial/'
search_fragment = '001cc'
count_ = generate_list(path, search_fragment)
print(path+' has ' + str(count_) + ' images')
count += count_

path = root_path + '/off-road/rain/cimatec-industrial/'
search_fragment = 'cim_'
count_ = generate_list(path, search_fragment)
print(path+' has ' + str(count_) + ' images')
count += count_



path = root_path + '/unpaved/day/estrada-dos-tropeiros/'
search_fragment = '001_'
count_ = generate_list(path, search_fragment)
print(path+' has ' + str(count_) + ' images')
count += count_

path = root_path + '/unpaved/day/jaua/'
search_fragment = 'jt_'
count_ = generate_list(path, search_fragment)
print(path+' has ' + str(count_) + ' images')
count += count_

path = root_path + '/unpaved/day/praia-do-forte/'
search_fragment = 'k1_'
count_ = generate_list(path, search_fragment)
print(path+' has ' + str(count_) + ' images')
count += count_



path = root_path + '/unpaved/rain/estrada-dos-tropeiros/'
search_fragment = 'c0_'
count_ = generate_list(path, search_fragment)
print(path+' has ' + str(count_) + ' images')
count += count_

path = root_path + '/unpaved/rain/jaua/'
search_fragment = 'c0_'
count_ = generate_list(path, search_fragment)
print(path+' has ' + str(count_) + ' images')
count += count_

path = root_path + '/unpaved/rain/praia-do-forte/'
search_fragment = 'd1_'
count_ = generate_list(path, search_fragment)
print(path+' has ' + str(count_) + ' images')
count += count_


print('The subset has ' + str(count) + ' images')