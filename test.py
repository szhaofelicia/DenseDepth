import os
import glob
import argparse
import matplotlib

# Keras / TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import predict, load_images, display_images
from matplotlib import pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable
import skimage
from skimage.transform import resize

import numpy as np
import cv2

from tqdm import tqdm

import pickle

def display_results(filenames,outputs, inputs=None ,gt=None, is_colormap=True, is_rescale=True,figsize=(24,16)):
    plasma = plt.get_cmap('plasma')

    # shape = (outputs[0].shape[0], outputs[0].shape[1], 3) # default
    shape = (inputs[0].shape[0], inputs[0].shape[1], 3) # H = 720, W=1280
    # all_images = []

    for i in tqdm(range(outputs.shape[0])):
        # imgs = []
        fig,ax=plt.subplots(2,2,figsize=figsize)

        ax[0,0].imshow(inputs[i])

        if is_colormap:
            rescaled = outputs[i][:,:,0]
            if is_rescale:
                rescaled = rescaled - np.min(rescaled)
                rescaled = rescaled / np.max(rescaled)
            depth=ax[0,1].imshow(cv2.resize(plasma(rescaled)[:,:,:3],dsize=(1280,720)  , interpolation = cv2.INTER_AREA),cmap='plasma')
        else:
            depth=ax[0,1].imshow(outputs[i],cmap='gray', vmin = 0, vmax = 255)
                
        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig.colorbar(depth,cax=cax, orientation='vertical')

        plt.colorbar(depth, ticks=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],orientation='vertical')

        ave=np.mean(rescaled,axis=(0,1))
        med=np.median(rescaled,axis=(0,1))
        ax[1,0].hist(rescaled.ravel(),100,[0,1])
        ax[1,0].set_title('ave='+str(ave)+' median='+str(med))
        
        plt.savefig('outputs/rescaled_histogram/'+filenames[i]+'.png')

    return outputs


# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default='pretrained/nyu.h5', type=str, help='Trained Keras model file.')
# parser.add_argument('--input', default='examples/*.png', type=str, help='Input filename or folder.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

print('Loading model...')

# Load model into GPU / CPU
model = load_model(args.model, custom_objects=custom_objects, compile=False)

print('\nModel loaded ({0}).'.format(args.model))

# Input images

PATH_TO_TEST_IMAGES_DIR = '/media/felicia/Data/mlb-youtube/frames_continuous/swing'
PICKLE_DIR='/media/felicia/Data/baseballplayers/pickles/'

# BATCH=19
# inputfiles=[]
# filenames=[]

videonames=pickle.load(open(PICKLE_DIR+'swing_videos.pkl','rb'))
ALL=len(videonames)

for b in tqdm(range(ALL)):
    inputfiles=[]
    filenames=[]
    outputs=list()

    v=videonames[b]
    for i in range(19):
        f=v+'{:04d}'.format(i)
        filenames.append(f)
        inputfiles.append(os.path.join(PATH_TO_TEST_IMAGES_DIR,f+'.jpg'))

    print('Loaded filenames.')

    inputs_original, inputs = load_images( inputfiles)

    print('\nLoaded ({0}) images of size {1}.'.format(inputs.shape[0], inputs.shape[1:]))
    print('\nLoaded ({0}) images of original size {1}.'.format(inputs_original.shape[0], inputs_original.shape[1:]))

    # Compute results
    outputs=predict(model, inputs) # B* 240 * 320 *1
    outputs=np.array(outputs)

    frame_depth={}

    for i in tqdm(range(19)):
        frame_depth[filenames[i]]=outputs[i]

    pkl_dict={}
    pkl_dict['filenames']=filenames
    pkl_dict['outputs']=outputs
    pkl_dict['frame_depth']=frame_depth
    # pkl_dict['inputs']=inputs_original

    pickle.dump(pkl_dict, open("/media/felicia/Data/baseballplayers/pickles/depth_outputs_{:03d}.pkl".format(b),"wb"))
    print('Saved in pickles.')
