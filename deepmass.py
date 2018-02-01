
import os
import tables

from download_data import preprocess_volumes, store_to_hdf5, PatchData, store_preloaded_hdf5_file
from util import add_parameter
from model import PGGAN

class DeepMass(object):

    def __init__(self, **kwargs):

        # General Parameters
        add_parameter(self, kwargs, 'verbose', True)
        add_parameter(self, kwargs, 'train', True)
        add_parameter(self, kwargs, 'test', True)

        # Train Data Parameters
        add_parameter(self, kwargs, 'train_data_directories', None)
        add_parameter(self, kwargs, 'train_preprocess_images', False)
        add_parameter(self, kwargs, 'train_hdf5', None)
        add_parameter(self, kwargs, 'train_overwrite', False)
        add_parameter(self, kwargs, 'train_preloaded', False)
        add_parameter(self, kwargs, 'train_max_dimension', 64)
        add_parameter(self, kwargs, 'train_boundary_padding', 10)

        # Training Classifier Parameters
        # TODO

        # Training GAN Parameters
        add_parameter(self, kwargs, 'gan_samples_dir', './samples')
        add_parameter(self, kwargs, 'gan_log_dir', './log')
        add_parameter(self, kwargs, 'gan_latent_size', 512)
        add_parameter(self, kwargs, 'gan_max_filter', 1024)

        return

    def execute(self):

        if self.train:

            # Data preparation.
            self.training_storage = self.get_data(data_directory=self.train_data_directories, preprocess_images=self.train_preprocess_images, hdf5=self.train_hdf5, overwrite=self.train_overwrite, preloaded=self.train_preloaded)

            if True:
            # try:
                self.train_gan()
            # except:
                # self.training_storage.close()

            self.training_storage.close()

        return

    def get_data(self, data_directory=None, preprocess_images=False, hdf5=None, overwrite=False, preloaded=False):

        if not os.path.exists(hdf5) or overwrite:

            # # Preprocess Images. TODO (different preprocessing methods)
            # if preprocess_images:
            #     preprocessed_directory = os.path.join(data_directory, 'converted_images')
            #     if not os.path.exists(converted_directory):
            #         os.mkdir(converted_directory)
            #     preprocess_volumes(data_directory, converted_directory)
            # else:
            #     preprocessed_directory = converted_directory

        # Convert to HDF5
            output_hdf5 = store_preloaded_hdf5_file(data_directory, hdf5)

        else:
            output_hdf5 = tables.open_file(hdf5, "r")

        # Convert to data-loading object. The logic is all messed up here for pre-loading images.
        return PatchData(hdf5=output_hdf5, preloaded=preloaded)

    def train_gan(self):

        # Create necessary directories
        for work_dir in [self.gan_samples_dir, self.gan_log_dir]:
            if not os.path.exists(work_dir):
                os.mkdir(work_dir)

        # Inherited this from other code, think of a programmatic way to do it.
        training_depths = [1,2,2,3,3,4,4,5,5,6,6]
        read_depths = [1,1,2,2,3,3,4,4,5,5,6]
        # training_depths = [9]
        # read_depths = [9]

        for i in range(len(training_depths)):

            if (i % 2 == 0):
                transition = False
            else:
                transition = True

            output_model_path = os.path.join(self.gan_log_dir, str(training_depths[i]), 'model.ckpt')
            if not os.path.exists(os.path.dirname(output_model_path)):
                os.makedirs(os.path.dirname(output_model_path))

            input_model_path = os.path.join(self.gan_log_dir, str(read_depths[i]), 'model.ckpt')

            sample_path = os.path.join(self.gan_samples_dir, 'sample_' + str(training_depths[i]) + '_' + str(transition))
            if not os.path.exists(sample_path):
                os.makedirs(sample_path)

            pggan = PGGAN(training_data = self.training_storage,
                            input_model_path=input_model_path, 
                            output_model_path=output_model_path,
                            samples_dir=sample_path, 
                            log_dir=self.gan_log_dir,
                            progressive_depth=training_depths[i],
                            transition=transition)

            pggan.build_model()
            pggan.train()

if __name__ == '__main__':

    pass