import os

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    from deepmass import DeepMass

    linux_media = '/media/anderff/My Passport/'
    windows_media = 'E:/'

    gan = DeepMass(train_data_directories=['~/Local_Data/BRATS/preprocessed'],
                    train_hdf5='whole_volumes.hdf5',
                    train_overwrite=False,
                    train_preloaded=False,
                    gan_samples_dir='./samples_wholebrain',
                    gan_log_dir='./log_wholebrain')

    gan.execute()