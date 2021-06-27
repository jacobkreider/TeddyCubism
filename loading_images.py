"""Loading train and test images for cycleGAN and converting to numpy array"""
from datetime import datetime
from os import listdir

from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import asarray
from numpy import savez_compressed
from numpy import vstack


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


script_start = timer(None)


def load_images(path, size=(256,256)):
    data_list = list()
    # enumerate filenames in directory, assume all are images
    for filename in listdir(path):
        # load and resize image
        pixels = load_img(path + filename, target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)
        # store
        data_list.append(pixels)
    return asarray(data_list)


# dataset path
path = '/home/jacob/DataScience/Fun/Art/TeddyCubism/'

# load Teddy images
teddy1 = load_images(path + 'teddy/train/')
teddy2 = load_images(path + 'teddy/test/')
teddy = vstack((teddy1, teddy2))
print('Loaded Teddy images: {}'.format(teddy.shape))

# load art images
cubism1 = load_images(path + 'cubism/train/')
cubism2 = load_images(path + 'cubism/test/')
cubism = vstack((cubism1, cubism2))
print("Loaded art images: {}".format(cubism.shape))

# save as compressed numpy array
filename = 'teddyGAN.npz'
savez_compressed(filename, teddy, cubism)
print("Images saved as numpy array. Script complete.")

print(timer(script_start))
