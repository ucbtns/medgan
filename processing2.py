

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 17:49:53 2018
@author: noorsajid
"""

import pandas as pd
import os
import cv2

import concurrent.futures
from pathlib import Path
import lungs_finder as lf
import multiprocessing as mp
mp.cpu_count()


os.chdir(Path('E:\PhD'))
df = pd.read_csv('suspects.csv')
df.columns = ['id'] + list(df)[1:]

# true individuals who are patients with suspected cases of TB:
true = list(df['id'].dropna())

# Now compare against the full image database:   
path_base = Path('E:\PhD\jpg2')

limit = 10000000000
file_name = []  

i=0
for index, filename in enumerate(os.listdir(path_base)):        
        i += 1
        file_name.append(filename)
        if i == limit:
            break
      
ufalse = [file[7:-4]for file in file_name]

def returnNotMatches(a, b):
    a = set(a)
    b = set(b)
    return [list(b - a), list(a - b)]

false = returnNotMatches(true, ufalse)[0]

# two lists:
    # false : normal -- 88654
    # true : anomalous  -- 512
    
# Join them FALSE : TRUE
files = false + true 
             
final = [str(file) +'.jpg' for file in files]    

    
names = []   
for i in final: 
    print(i)
    names.append([f for f in os.listdir(path_base) if i in f])
    print(len(names))
    
df = pd.DataFrame(names)
df.columns = ['names', 'names2', 'names3', 'names4', 'names5', 'names6', 'names7']
# df.columns = ['names', 'names2', 'names3']
len(df['names'].dropna())

df['class'] = 1
df['class'].iloc[:88654] = 0
  
  
orig = df[['names', 'class']].dropna()  
orig2 = df[['names2', 'class']].dropna() 
orig2.columns = ['names', 'class']
orig3 = df[['names3', 'class']].dropna() 
orig3.columns = ['names', 'class']
orig4 = df[['names4', 'class']].dropna() 
orig4.columns = ['names', 'class']
orig5 = df[['names5', 'class']].dropna() 
orig5.columns = ['names', 'class']
orig6 = df[['names6', 'class']].dropna() 
orig6.columns = ['names', 'class']
orig7 = df[['names7', 'class']].dropna() 
orig7.columns = ['names', 'class']

final = pd.concat([orig, orig2, orig3, orig4, orig5, orig6, orig7], axis=0)

#final = pd.concat([orig, orig2, orig3], axis=0)
true = list(final[final['class'] == 1]['names']) # anomalous
false = list(final[final['class'] == 0]['names']) # normal
  

def loader(data):
    img = cv2.imread(data, 0)
    return img


def pp_load_image(path, file_name):
    
    ''' The purpose of below is to load 
    white-scale images from the chest 
    scans. 
    
    Please note this is looking up an 
    extremely small sample of the total.
    These may include both TB / Health 
    images. '''
    
    images = []
    os.chdir(path)
    
    for file in file_name:
         print(file)
         images.append(loader(file))   
        
    return images, file_name


def lung_finder(data):
    ''' The purpose of below is to find 
    just the lungs from the chest 
    scans. '''
    comb = []
    for i in data:
        found_lungs = lf.get_lungs(i)
        
        if found_lungs is not None:
            comb.append(found_lungs)
    
    return comb

def slices_saved(images, filenames):
    
    
    import image_slicer as sl
    
    for i, name in zip(images, filenames):
            
           sl.slice(i, 4, name)   

def main(base, names, path):
    print('loading:')
    imag, filename = pp_load_image(base, names)
    print('lung finder:')
    #resize = resize_image(imag)   
    lungs_found = lung_finder(imag)
    print('creating path:')
    if not os.path.exists(path):
        os.makedirs(path)
        
    os.chdir(path)
    print('creating and saving slices...')
    slices_saved(lungs_found, filename)
    #save_img(resize, names, path)


  

trfalse_path = "E:\\PhD\\GAN_DT\\Train\\0.normal"
main(path_base, false[:92000], trfalse_path)


tefalse_path = "E:\\PhD\\GAN_DT\\Test\\0.normal"
main(path_base, false[92000:], tefalse_path)

tetrue_path = "E:\\PhD\\GAN_DT\\Test\\1.abnormal"
main(path_base, true, tetrue_path)


'''
base = path_base
names = false
path = trfalse_path
'''
