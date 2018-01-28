import numpy as np
import scipy

from qtim_tools.qtim_utilities.nifti_util import save_numpy_2_nifti

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w: i * w + w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def inverse_transform(image):
    
    # Will have to rationalize how to transform images
    # back into MRI intensities here.
    return image


def save_images(images, size, image_path):
    data = inverse_transform(images)
    print data.shape
    return imsave(data, size, image_path)


def add_parameter(class_object, kwargs, parameter, default=None):
    if parameter in kwargs:
        setattr(class_object, parameter, kwargs.get(parameter))
    else:
        setattr(class_object, parameter, default)


if __name__ == '__main__':

    pass