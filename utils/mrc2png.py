# Script to process 3D MRC files and output PNG slices

import argparse
from glob import glob
import mrcfile
import numpy as np
import os
import skimage
from skimage import exposure
from skimage import io
from skimage import transform
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Convert MRC files to PNG')
parser.add_argument('--mrc_dir',
                    help='Directory containing MRC files',
                    type=str,
                    default='/Users/Holmes/Research/IDSAI/PROOF/data'
                            '/tomograms2D/original_mrc',
                    )
parser.add_argument('--png_dir', help='Directory in which to put PNGs',
                    type=str,
                    default=None,
                    )
parser.add_argument('--printout',
                    help='Print MRC header',
                    type=bool,
                    default=False,
                    )
parser.add_argument('--ahe',
                    help='Adaptive Histogram Equalisation',
                    type=bool,
                    default=False,
                    )
parser.add_argument('--he',
                    help='Histogram Equalisation',
                    type=bool,
                    default=False,
                    )
parser.add_argument('--clip',
                    help='Clip the 5% and 95% quartiles of image',
                    type=bool,
                    default=False,
                    )
parser.add_argument('--keep_dims',
                    help='Keep original image dimensions',
                    type=bool,
                    default=True,
                    )
parser.add_argument('--width', help='New PNG width', type=int, default=1024)
parser.add_argument('--height', help='New PNG height', type=int, default=1024)
parser.add_argument('--depth', help='New PNG height', type=int, default=None)
args = parser.parse_args()
if args.keep_dims:
    (args.depth, args.height, args.width) = (None, None, None)

# Check directories exists.
if not os.path.isdir(args.mrc_dir):
    raise IOError("Directory doesn't exist: {}".format(args.mrc_dir))
if not args.png_dir:
    parent_dir = os.path.abspath(os.path.join(args.mrc_dir, os.pardir))
    args.png_dir = parent_dir + '/processed_png'
    print(args.png_dir)
if not os.path.isdir(args.png_dir):
    os.makedirs(args.png_dir)
    print('Made new PNG dir: ', args.png_dir)

# List all MRC files in the directory.
mrcs = glob(args.mrc_dir + '/*.mrc')
if not mrcs:
    print('No MRC files here: ' + args.mrc_dir)

for idx, file in enumerate(mrcs):
    filename = os.path.basename(file)

    print("\nProcessing MRC #{num}: `{name}`".format(num=idx+1, name=filename))
    with mrcfile.open(file, permissive=True) as mrc:
        image = mrc.data
        if args.printout:
            mrc.print_header()

    print("Input image shape: ", image.shape)

    # Clip the 5% and 95% quartiles
    if args.clip:
        image = np.clip(
            image, np.percentile(image, 5), np.percentile(image, 95),
        )

    # Normalise image values over [0, 1]
    image = (image - image.min()) / (image.max() - image.min())

    # Resize output
    if any((args.depth, args.height, args.width)):
        print('Resizing image')
        output_shape = [args.height, args.width]
        if len(image.shape) == 3:
            output_shape = [args.depth] + output_shape
        for dim, length in enumerate(output_shape):
            if not length:
                output_shape[dim] = image.shape[dim]
        print("Output image shape: ", tuple(output_shape))
        image = transform.resize(image, output_shape, anti_aliasing=False)

    if args.ahe:
        print('Adaptive histogram equalisation')
        kernel_size = (image.shape[-2] // 5, image.shape[-1] // 5)
        if len(image.shape) == 3:
            kernel_size = (image.shape[0] // 2,) + kernel_size
        kernel_size = np.array(kernel_size)
        clip_limit = 0.9  # In [0, 1] where higher value increases contrast
        image = exposure.equalize_adapthist(image, kernel_size, clip_limit)
    elif args.he:
        print('Histogram equalisation')
        image = exposure.equalize_hist(image)

    # Renormalise to 8-bit unit for PNG save
    image = skimage.img_as_ubyte(image)

    print('Saving z-slices:')
    filename = filename.split('.')[0]
    if len(image.shape) == 2:
        io.imsave(args.png_dir + '/' + filename + '.png', image)
    elif len(image.shape) == 3:
        stackdir = args.png_dir + '/zstack_' + filename
        if not os.path.isdir(stackdir):
            os.makedirs(stackdir)
        for z in tqdm(range(image.shape[0])):
            io.imsave(stackdir + '/zslice{:04d}.png'.format(z), image[z])
