import mrcfile
import os
import numpy
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

def get_files_of_type_from_path(fpath, fext):
    '''
    Gets a list of all paths with a particular extension (fext) given a path (fpath)
    '''
    data_file_paths = []
    for file in os.listdir(fpath):
        if file.endswith(fext):
            data_file_paths.append(os.path.join(fpath, file))
    return data_file_paths

def get_all_imgs_from_mrcs(fname):
    '''
    Returns a list of 2D numpy arrays extracted from a given mrcs file (fname)
    '''
    if os.path.splitext(fname)[1]!='.mrcs':
        raise ValueError('File extension of {} is not .mrcs'.format(fname))
    
    with mrcfile.open(fname) as mrc:
        img_stack = mrc.data
        
    imgs = []
    if len(img_stack.shape)==3:
        for i in range(img_stack.shape[0]):
            imgs.append(img_stack[i,:,:])
    elif len(img_stack.shape)==2:
        imgs.append(img_stack)   

    return imgs

def get_all_imgs_from_paths(flist):
    '''
    Given a list of paths to .mrcs files (flist), produces a list of numpy arrays of images
    Note that this is creating an in-memory list of arrays, so don't use it with a large list of .mrcs files
    '''
    all_imgs = []
    
    for fname in flist:
        imgs = get_all_imgs_from_mrcs(fname)
        all_imgs.extend(imgs)
    
    return all_imgs

def create_dataset_from_mrcs(classes, paths_in, paths_out, fext='.mrcs'):
    '''
    Given a list of classes (classes),
    a list of paths (paths_in) to input folders corresponding to each class, each containing files with extension fext,
    and a list of paths (paths_out) to folders corresponding to each class, where .pkl files are to be stored,
    '''
    
    for i, cl in enumerate(classes):
        # Create output path and img counter
        img_cntr = 0
        if not os.path.exists(paths_out[i]):
            os.makedirs(paths_out[i])
        
        # Get paths to .mrcs files to be processed
        in_paths = get_files_of_type_from_path(paths_in[i], fext)
        
        # LoopExtract numpy arrays from
        for fpath in tqdm(in_paths):
            # Get list of images
            imgs = get_all_imgs_from_mrcs(fpath)
            
            # Write images to files
            for img in imgs:
                img.dump(os.path.join(paths_out[i], str(cl)+str(img_cntr)+'.pkl'))
                img_cntr += 1


# Data path
data_path = '../data/'

# Paths of side/top/bad .mrcs files
side_path_in = os.path.join(data_path, 'side/micrographs')
bad_path_in = os.path.join(data_path, 'bad/micrographs')

# Paths to store side/top/bad .pkl files
side_path_out = os.path.join(data_path, 'side/pkl')
bad_path_out = os.path.join(data_path, 'bad/pkl')
data_file_ext = '_lowpass.mrcs'

# Make datasets
create_dataset_from_mrcs(   ['side', 'bad'],
                            [side_path_in, bad_path_in],
                            [side_path_out, bad_path_out],
                            fext=data_file_ext
                            )
