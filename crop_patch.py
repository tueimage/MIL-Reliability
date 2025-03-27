import numpy as np
from openslide import OpenSlide
from glob import glob
import h5py
import os

def crop_patch(slide, att):
    slide_wsi = OpenSlide(os.path.join(data_dir,'images/images','{}.svs'.format(slide)))
    full_path = os.path.join(data_dir,'Tupac16_ostu_20x/feat/h5_files','{}.h5'.format(slide))

    with h5py.File(full_path,'r') as hdf5_file:
        coords = hdf5_file['coords'][:]

    att_slide= att[slide]
    ind= np.argpartition(att_slide[0], -20)[-20:]
    for j,i in enumerate(ind):
        x,y= coords[i][0], coords[i][1]
        patch= slide_wsi.read_region((x,y),1,(256,256))
        os.makedirs(os.path.join(data_dir,'top_patches',slide), exist_ok=True)
        patch.save(os.path.join(data_dir,'top_patches',slide, '{}.png'.format(j)))

data_dir= '/home/bme001/20215294/Data/TUPAC'
names= glob(os.path.join(data_dir,'Tupac16_ostu_20x/feat/h5_files','*.h5')) 
slides= [name.split('/')[-1].split('.')[0] for name in names]
att= np.load("att_abmil.npy", allow_pickle=True).item()

for slide in slides:
    try:
        print(slide)
        crop_patch(slide, att)
    except:
        pass    
