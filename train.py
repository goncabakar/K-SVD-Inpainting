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
                        default="./images/barbara.png", help='Image Directory')
parser.add_argument('-d', '--dict_dir', type=str,
                        default="./dictionary/dictionary.npy", help='Dictionary Directory')
parser.add_argument('-ds', '--dict_size', type=int,
                        default=24, help='Size of dictionary')
parser.add_argument('-sp', '--sparsity', type=int,
                        default=4, help='Sparsity constraint')
parser.add_argument('-n', '--n_iter', type=int,
                        default=15, help='Num of iterations')
parser.add_argument('-t', '--train', type=int,
                        default=1, help='Training mode')
parser.add_argument('-sd', '--save_dic', type=int,
                        default=1, help='Training mode')

args = parser.parse_args()

if not args.train:
    args.n_iter = 0

barbara = imageio.imread("./images/barbara.png").astype('float32')[0:args.input_size,0:args.input_size]
barbara = np.array(barbara, dtype=np.float32)

boat = imageio.imread("./images/boat.png").astype('float32')[0:args.input_size,0:args.input_size]
boat = np.array(boat, dtype=np.float32)

girlface = imageio.imread("./images/girlface.bmp").astype('float32')[0:args.input_size,0:args.input_size]
girlface = np.array(girlface, dtype=np.float32)

# set patch size
patch_size = args.patch_size


# initialize denoiser
initial_dictionary = DCTDictionary(patch_size, args.dict_size)
initial_dictionary.matrix = np.load(args.dict_dir)
inpainting = KSVDImageInpainting(initial_dictionary, pursuit=MatchingPursuit)

# denoise image
d = inpainting.train(barbara, n_iter=args.n_iter, patch_size=args.patch_size)
d = inpainting.train(boat, n_iter=args.n_iter, patch_size=args.patch_size)
d = inpainting.train(girlface, n_iter=args.n_iter, patch_size=args.patch_size)

if args.save_dic:
    np.save(args.dict_dir, d.matrix)
