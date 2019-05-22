import matplotlib.pyplot as plt
from inpainting import KSVDImageInpainting
from pursuits import MatchingPursuit
from dictionaries import DCTDictionary
from sklearn.feature_extraction.image import extract_patches_2d
import imageio
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--patch_size', type=int,
                        default=16, help='Patch Size')
parser.add_argument('-s', '--input_size', type=int,
                        default=64, help='Size of input image')
parser.add_argument('-i', '--image_dir', type=str,
                        default="./images/bridge.bmp", help='Image Directory')
parser.add_argument('-d', '--dict_dir', type=str,
                        default="./dictionary/dictionary.npy", help='Dictionary Directory')
parser.add_argument('-ds', '--dict_size', type=int,
                        default=24, help='Size of dictionary')
parser.add_argument('-sp', '--sparsity', type=int,
                        default=8, help='Sparsity constraint')
parser.add_argument('-n', '--n_iter', type=int,
                        default=50, help='Num of iterations')
parser.add_argument('-t', '--train', type=int,
                        default=0, help='Training mode')
parser.add_argument('-sd', '--save_dic', type=int,
                        default=0, help='Training mode')

args = parser.parse_args()

if not args.train:
    args.n_iter = 0

initx = 60
inity = 300
original_img = imageio.imread(args.image_dir).astype('float32')[initx:initx+args.input_size,inity:inity+args.input_size]
original_img = np.array(original_img, dtype=np.float32)
mask = np.ones((original_img.shape), dtype=np.float32)
noisy_img = np.zeros((original_img.shape), dtype=np.float32)
inpainted = np.zeros((original_img.shape), dtype=np.float32)
missing1 = 4
missing2 = 6
missing3 = 12
missing4 = 14
missing5 = 24
missing6 = 26
mask[missing1:missing2, missing1:missing2] = 0
mask[missing3:missing4, missing3:missing4] = 0
mask[missing5:missing6, missing5:missing6] = 0
# noisy_img = original_img
# noisy_img[4:6, 4:6] = 255
noisy_img = (original_img*mask)
noisy_img[missing1:missing2, missing1:missing2] = 255
noisy_img[missing3:missing4, missing3:missing4] = 255
noisy_img[missing5:missing6, missing5:missing6] = 255
image_size = original_img.shape[0]

# set patch size
patch_size = args.patch_size

plt.figure()
plt.imshow(original_img, cmap='gray')
plt.savefig("./results/original.png")

# initialize denoiser
initial_dictionary = DCTDictionary(patch_size, args.dict_size)
initial_dictionary.matrix = np.load(args.dict_dir)
print("Pre-trained dictionary is loaded.")
inpainting = KSVDImageInpainting(initial_dictionary, pursuit=MatchingPursuit)

# denoise image
z, d, a = inpainting.inpaint(original_img, noisy_img, n_iter=args.n_iter, patch_size=patch_size)

inpainted = original_img
inpainted[missing1:missing2, missing1:missing2] = z[missing1:missing2, missing1:missing2]
inpainted[missing3:missing4, missing3:missing4] = z[missing3:missing4, missing3:missing4]
inpainted[missing5:missing6, missing5:missing6] = z[missing5:missing6, missing5:missing6]
plt.figure()
plt.imshow(noisy_img, cmap='gray')
plt.savefig("./results/noisy.png")

plt.figure()
plt.imshow(inpainted, cmap='gray')
plt.savefig("./results/inpainted.png")
if args.save_dic:
    np.save(args.dict_dir, d.matrix)
    print("Updated dictionary is saved.")
