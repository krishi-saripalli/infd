from .datasets import register, make
from . import image_folder
from . import ffhq
from . import random_lr_hr
from . import wrapper_cae

# Explicitly import and register HDF5Dataset here
try:
    from OneNoise.noise_data import HDF5Dataset
    register('hdf5_dataset')(HDF5Dataset)
except ImportError:
    # Handle case where OneNoise might not be installed or in path
    print("Warning: Could not import or register HDF5Dataset from OneNoise.")
    # Optionally raise the error if HDF5Dataset is strictly required
    raise
except NameError:
    # Handle case where 'register' might not be defined yet (if import order is tricky)
    print("Warning: 'register' function not found for HDF5Dataset registration.")
    # raise
