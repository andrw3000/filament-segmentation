import os
import sys
import pandas as pd
import starfile
import argparse
from pathlib import Path
from itertools import islice


def coords2star(line_ends, star_dir):
    """Function to convert the output of the instance segmentation to STAR.

    Args:
        `line_ends` containins coordinates obtainined via `get_line_instances`.
        `star_dir` is the location for the output STAR file.
    """

    # Generate blank data frame with the headers
    df = pd.DataFrame(columns=['Xcoords', 'Ycoords', 'ImageName'])

    # Check line ends
    for idx, end_pairs in enumerate(line_ends):
        print('\nLine endings on line {}:'.format(idx + 1))
        for ends in end_pairs:
            e1 = ends[0]
            e2 = ends[1]
            print('({e1y:d}, {e1x:d}) -> ({e2y:d}, {e2x:d})'.format(
                e1y=int(e1[0]), e1x=int(e1[1]), e2y=int(e2[0]), e2x=int(e2[1])
            ))

    i = 0

    # for each model file, convert to txt using model2point,
    # load into a pandas dataframe and append with tomogram name in a 4th column
    while i < len(modlist):
        print('model2point -sc -input ' + str(modlist[i]) + ' -ou ' + str(
            modlist[i]) + '.txt')
        os.system('model2point -sc -input ' + str(modlist[i]) + ' -ou ' + str(
            modlist[i]) + '.txt')
        modfile = pd.read_csv(modlist[i] + '.txt', sep=' ', header=None,
                              skipinitialspace=True)
        modfile.columns = ['rlnCoordinateX', 'rlnCoordinateY',
                           'rlnCoordinateZ']
        path = Path(edf[i])
        # .stem gives root file name
        file = Path(edf[i]).stem
        modfile.insert(3, 'rlnMicrographName',
                       './' + file + '/' + file + '.tomostar')
        star_file = star_file.append(modfile)
        i += 1

    starfile.write(star_file, star_dir)

    # Print first few lines of the STAR file for sanity
    with open(star_dir) as file:
        for line in islice(file, 15):
            print(line)




