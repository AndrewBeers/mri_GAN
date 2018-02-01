import glob
import os
import tables
import numpy as np

from subprocess import call
from scipy.misc import imresize
from scipy.ndimage import zoom
from scipy.ndimage.measurements import center_of_mass
from PIL import Image

from qtim_tools.qtim_utilities.format_util import convert_input_2_numpy
from qtim_tools.qtim_utilities.nifti_util import save_numpy_2_nifti
from qtim_tools.qtim_utilities.file_util import grab_files_recursive
from qtim_tools.qtim_preprocessing.orientation import orient

def extract_mri_cubes(data_directories, shape=(128,128,128)):

    for directory in data_directories:

        patients = glob.glob(os.path.join(directory, '*/'))

        for patient in patients:

            tumor_label = center_of_mass()

def preprocess_volumes(input_directory='E:/Pages_Images', output_directory='E:/Pages_Images_Preprocessed', resize_shape=(64, 64), verbose=True):

    # for directory in data_directories:

    #     patients = glob.glob(os.path.join(directory, '*/'))

    #     for patient in patients:

    #         output_filepath = os.path.join(output_directory, os.path.basename(filepath))
    #         if not os.path.exists(output_filepath):
    #             if verbose:
    #                 print 'Processing...', filepath

    #             img = Image.open(filepath)
    #             data = np.asarray(img, dtype='uint8')

    #             data = imresize(data, resize_shape)

    #             img = Image.fromarray(data)
    #             img.save(output_filepath)

    #     except KeyboardInterrupt:
    #         raise
    #     except:
    #         print 'ERROR converting', filepath

    return

def create_hdf5_file(output_filepath, num_cases, image_shape=(64, 64)):

    hdf5_file = tables.open_file(output_filepath, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')

    data_shape = (0,) + image_shape + (3,)

    hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=num_cases)
    hdf5_file.create_earray(hdf5_file.root, 'imagenames', tables.StringAtom(256), shape=(0,1), filters=filters, expectedrows=num_cases)

    return hdf5_file


def store_to_hdf5(data_directory, hdf5_file, image_shape, verbose=True):

    input_images = glob.glob(os.path.join(data_directory, '*.png'))

    hdf5_file = create_hdf5_file(hdf5_file, num_cases=len(input_images), image_shape=image_shape)

    for image in input_images:
        try:
            if verbose:
                print(image)
            img = Image.open(image)
            data = np.asarray(img)
            hdf5_file.root.data.append(data[np.newaxis])
            hdf5_file.root.imagenames.append(np.array(os.path.basename(image))[np.newaxis][np.newaxis])
        except:
            print 'ERROR WRITING TO HDF5', image

    return hdf5_file

def store_preloaded_hdf5_file(data_directories, output_filepath, modalities=['FLAIR_pp.nii.gz', 'T1post_pp.nii.gz'], label='full_edemamask_pp.nii.gz', verbose=True, levels=[4,8,16,32,64,128], boundary_padding=10, max_dimension=64, samples_per_patient=100, preload_levels=False, wholevolume=False):

    patient_vols = []
    for directory in data_directories:
        patients = glob.glob(os.path.join(directory, '*/'))
        for patient in patients:
            single_patient_vols = []
            for modality in modalities + [label]:
                if modality is None:
                    continue
                single_patient_vols += [glob.glob(os.path.join(patient, modality))[0]]
            patient_vols += [single_patient_vols]

    if wholevolume:
        num_cases = len(patient_vols)
    else:
        num_cases = len(modalities) * len(patient_vols)

    hdf5_file = tables.open_file(output_filepath, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    hdf5_file.create_earray(hdf5_file.root, 'imagenames', tables.StringAtom(256), shape=(0,1), filters=filters, expectedrows=num_cases)

    # If we want to pre-store different levels...
    if preload_levels:
        for dimension in levels:
            data_shape = (0, dimension + boundary_padding, dimension + boundary_padding, dimension + boundary_padding, 2)
            hdf5_file.create_earray(hdf5_file.root, 'data_' + str(dimension), tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=num_cases)
    else:
        # If we don't.
        if wholevolume:
            data_shape = (0, 200, 200, 200, len(modalities))
        else:
            data_shape = (0, max_dimension + boundary_padding, max_dimension + boundary_padding, max_dimension + boundary_padding, len(modalities))
        print data_shape
        hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=num_cases)

    for p_idx, single_patient_vols in enumerate(patient_vols):

        hdf5_file.root.imagenames.append(np.array(os.path.basename(os.path.dirname(single_patient_vols[0])))[np.newaxis][np.newaxis])
        print os.path.basename(os.path.dirname(single_patient_vols[0]))

        if label is not None:
            # Find tumor label center of mass
            label = single_patient_vols[-1]
            label_numpy = convert_input_2_numpy(label)
            label_center = [int(x) for x in center_of_mass(label_numpy)]

            # Load volumes, 
            volumes = np.stack([convert_input_2_numpy(vol) for vol in single_patient_vols[:-1]], axis=3)

            # pad if necessary, using black magic
            pad_dims = []
            radius = (max_dimension + boundary_padding)/2
            for idx, dim in enumerate(volumes.shape[:-1]):
                padding = (-1 * min(0, label_center[idx] - radius), -1 * min(0, dim - (label_center[idx] + radius)))
                pad_dims += [padding]
            pad_dims += [(0, 0)]
            print pad_dims
            volumes = np.pad(volumes, pad_dims, mode='constant')

            # and subsample, with more black magic ;)
            print label_center
            label_center = [x + pad_dims[i][0] for i, x in enumerate(label_center)]
            print label_center
            print volumes.shape
            patch = volumes[label_center[0]-radius:label_center[0]+radius, label_center[1]-radius:label_center[1]+radius, label_center[2]-radius:label_center[2]+radius, :]
            print patch.shape

            # Add to HDF5
            getattr(hdf5_file.root, 'data').append(patch[np.newaxis])

            save_numpy_2_nifti(patch[..., 1], single_patient_vols[0], os.path.join(os.path.dirname(single_patient_vols[0]), 'gan_patch.nii.gz'))

        elif wholevolume:

            # Load volumes, 
            volumes = np.stack([convert_input_2_numpy(vol) for vol in single_patient_vols], axis=3)

            # Crop volumes
            volumes = crop2(volumes)

            large = False
            
            # Skip strangely processed volumes
            for dim in volumes.shape:
                if dim > 200:
                    large = True

            if large:
                continue

            same_size_volume = np.zeros((200, 200, 200, len(modalities)))
            same_size_volume[0:volumes.shape[0], 0:volumes.shape[1], 0:volumes.shape[2], :] = volumes

            # Add to HDF5
            getattr(hdf5_file.root, 'data').append(same_size_volume[np.newaxis])

        else:

            # Generic MRI patching goes on here..

            continue

            if verbose:
                print 'Processed...', os.path.basename(os.path.dirname(single_patient_vols[0])), 'idx', p_idx

        # except KeyboardInterrupt:
        #     raise
        # except:
        #     print 'ERROR converting', filepath, 'at dimension', dimension

    hdf5_file.close()

    return


class PatchData(object):

    def __init__(self, shape=(64,64), hdf5=None, preloaded=False):

        self.shape = shape
        self.hdf5 = hdf5
        self.preloaded = preloaded

        self.image_num = getattr(self.hdf5.root, 'data').shape[0]
        self.indexes = np.arange(self.image_num)
        np.random.shuffle(self.indexes)

    def get_next_batch(self, batch_num=0, batch_size=64, zoom_level=1):

        total_batches = self.image_num / batch_size - 1

        if batch_num % total_batches == 0:
            np.random.shuffle(self.indexes)

        indexes = self.indexes[(batch_num % total_batches) * batch_size: (batch_num % total_batches + 1) * batch_size]
        
        data = np.array([getattr(self.hdf5.root, 'data')[idx] for idx in indexes])
        x, y, z = np.random.randint(low=0, high=10, size=(3))
        data = data[:, x:x+64, y:y+64, z:z+64, :]
        return data

    def close(self):

        self.hdf5.close()


def minimum_bounding_box(data_directories, modalities=['FLAIR_pp.nii.gz', 'T1post_pp.nii.gz']):

    max_dims = [0,0,0]

    patient_vols = []
    for directory in data_directories:
        patients = glob.glob(os.path.join(directory, '*/'))
        for patient in patients:
            single_patient_vols = []
            for modality in modalities:
                single_patient_vols += [glob.glob(os.path.join(patient, modality))[0]]
            patient_vols += [single_patient_vols]

    for p_idx, single_patient_vols in enumerate(patient_vols):

        for modality in single_patient_vols:
            
            array = convert_input_2_numpy(modality)
            # print array.shape
            cropped_array = crop2(array)
            # print cropped_array.shape

            for idx, dim in enumerate(max_dims):
                if cropped_array.shape[idx] > 200:
                    print idx, cropped_array.shape[idx]
                    print modality
                if cropped_array.shape[idx] > dim:
                    max_dims[idx] = cropped_array.shape[idx]

            # print max_dims

    print max_dims

def crop2(dat, clp=False):
    for i in range(dat.ndim):
        dat = np.swapaxes(dat, 0, i)  # send i-th axis to front
        while np.all( dat[0]==0 ):
            dat = dat[1:]
        while np.all( dat[-1]==0 ):
            dat = dat[:-1]
        dat = np.swapaxes(dat, 0, i)  # send i-th axis to its original position
    return dat

def download_slices(data_directories, output_filepath='mri_slice.hdf5', modalities=['FLAIR_pp.nii.gz', 'T1post_pp.nii.gz', 'T2_pp.nii.gz'], preload_levels=True, levels=[4,8,16,32,64,128,256], verbose=True):

    max_dims = [0,0,0]

    patient_vols = []
    for directory in data_directories:
        patients = glob.glob(os.path.join(directory, '*/'))
        for patient in patients:
            single_patient_vols = []
            for modality in modalities:
                single_patient_vols += [glob.glob(os.path.join(patient, modality))[0]]
            patient_vols += [single_patient_vols]

    num_cases = 120 * len(patient_vols)

    hdf5_file = tables.open_file(output_filepath, mode='w')
    filters = tables.Filters(complevel=5, complib='blosc')
    hdf5_file.create_earray(hdf5_file.root, 'imagenames', tables.StringAtom(256), shape=(0,1), filters=filters, expectedrows=num_cases)

    # If we want to pre-store different levels...
    if preload_levels:
        for dimension in levels:
            data_shape = (0, dimension, dimension, 3)
            hdf5_file.create_earray(hdf5_file.root, 'data_' + str(dimension), tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=num_cases)

    for p_idx, single_patient_vols in enumerate(patient_vols):

        filename = os.path.basename(os.path.dirname(single_patient_vols[0]))
        hdf5_file.root.imagenames.append(np.array(filename)[np.newaxis][np.newaxis])

        for dimension in levels:
            try:

                if verbose:
                    print 'Processing...', os.path.basename(filename), 'at', dimension, ', idx', p_idx

                volumes = np.stack([convert_input_2_numpy(vol) for vol in single_patient_vols], axis=3)

                for z in xrange(volumes.shape[2]):

                    data = volumes[:,:,z,:]
                    data = imresize(data, (dimension, dimension))

                    getattr(hdf5_file.root, 'data_' + str(dimension)).append(data[np.newaxis])

            except KeyboardInterrupt:
                raise
            except:
                raise
                # print 'ERROR converting', filename, 'at dimension', dimension

    hdf5_file.close()


def prepare_multi_institution_data(data_directories, output_directory, modalities=['FLAIR_norm2.nii.gz', 'T1post_norm2.nii.gz'], label='wholetumor_postprocessed-label.nii.gz',):

    patient_vols = []
    for directory in data_directories:
        patients = glob.glob(os.path.join(directory, '*/'))
        for patient in patients:
            single_patient_vols = []
            for modality in modalities + [label]:
                single_patient_vols += [glob.glob(os.path.join(patient, modality))[0]]
            patient_vols += [single_patient_vols]

    modality_dictionary = {'FLAIR_norm2.nii.gz': 'FLAIR_pp.nii.gz', 'T1post_norm2.nii.gz': 'T1post_pp.nii.gz', 'wholetumor_postprocessed-label.nii.gz': 'full_edemamask_pp.nii.gz'}

    for p_idx, single_patient_vols in enumerate(patient_vols):
        filename = os.path.basename(os.path.dirname(single_patient_vols[0]))
        print 'Processing...', os.path.basename(filename), 'at idx', p_idx

        patient_name = os.path.basename(os.path.dirname(single_patient_vols[0]))
        if not os.path.exists(os.path.join(output_directory, patient_name)):
            os.mkdir(os.path.join(output_directory, patient_name))

        for modality in single_patient_vols:

            output_modality_filename = os.path.join(output_directory, patient_name, modality_dictionary[os.path.basename(modality)])

            orient(modality, output_modality_filename)

if __name__ == '__main__':

    linux_media = '/media/anderff/My Passport/'
    windows_media = 'E:/'


    # minimum_bounding_box(['/home/local/PARTNERS/azb22/Local_Data/BRATS/preprocessed', preprocessed])
    # prepare_multi_institution_data(multi_institution, preprocessed)

    store_preloaded_hdf5_file([preprocessed], 'whole_volumes.hdf5', wholevolume=True, label=None)