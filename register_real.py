import matplotlib.pyplot as plt
from models.voxel_morph import SimpleRegistrationModel
from utils.data_utils import load_nifti_pair

fixed_img, moving_img = load_nifti_pair('data/real/ct.nii.gz', 'data/real/mri.nii.gz')
model = SimpleRegistrationModel()
registered_img = model.register(fixed_img, moving_img)

fig, axs = plt.subplots(1, 3)
axs[0].imshow(fixed_img[fixed_img.shape[0]//2], cmap='gray')
axs[1].imshow(moving_img[moving_img.shape[0]//2], cmap='gray')
axs[2].imshow(registered_img[registered_img.shape[0]//2], cmap='gray')
plt.show()
