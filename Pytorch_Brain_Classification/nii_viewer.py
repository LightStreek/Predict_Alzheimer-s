import nibabel as nib
import matplotlib.pyplot as plt


def show_nii(path):
    """ 
    Taks an input file of type .nii and outputs an image/slice of the file. 
    """
    img = nib.load(path)
    data = img.get_fdata()
    print(data.dtype)
    plt.imshow(data[:,:, data.shape[2]//2].T, cmap="Greys_r" )
    plt.show()

