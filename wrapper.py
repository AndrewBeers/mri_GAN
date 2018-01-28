import os

if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    from deepmass import DeepMass

    linux_media = '/media/anderff/My Passport/'
    windows_media = 'E:/'

    gan = DeepZine(train_data_directory=None,
                    train_hdf5='patches.hdf5',
                    train_overwrite=False,
                    train_preloaded=True,
                    gan_samples_dir='./samples',
                    gan_log_dir='./log')

    gan.execute()