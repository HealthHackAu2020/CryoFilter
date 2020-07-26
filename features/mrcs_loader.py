import mrcfile
import os
from tqdm import tqdm
import numpy as np

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
    Note that this is creating an in-memory list, so don't use it with a large list of images
    '''
    all_imgs = []
    
    for fname in flist:
        imgs = get_all_imgs_from_mrcs(fname)
        all_imgs.extend(imgs)
    
    return all_imgs


def create_dataset_from_mrcs(classes, paths_in, paths_out, fext='.mrcs'):
    '''
    Given class classes (classes),
    a list of paths to folders containing .mrcs files (paths_in)
    and a list of paths to folders where .pkl files are to be stored (paths_out)
    This reads .mrcs files and writes one by one  
    '''
    
    for i, cl in enumerate(classes):
        # Create output path and img counter
        img_cntr = 0
        if not os.path.exists(paths_out[i]):
            os.makedirs(paths_out[i])
        
        # Get paths to .mrcs files to be processed
        in_paths = get_files_of_type_from_path(paths_in[i], fext)
        
        # Loop
        for fpath in tqdm(in_paths):
            # Get list of images
            imgs = get_all_imgs_from_mrcs(fpath)
            
            # Write images to files
            for img in imgs:
                np.save(os.path.join(paths_out[i], cl+'_'+str(img_cntr)+'.npy'), img)
                img_cntr += 1
                
                
def load_paths_into_array(paths):
    '''
    Given a list of paths (paths) of .pkl files of numpy arrays,
    return a list of numpy arrays
    '''
    arrs = []
    for p in paths:
        arrs.append(np.load(p))
    return arrs
