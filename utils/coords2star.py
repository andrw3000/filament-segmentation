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
    df = pd.DataFrame(columns=['Xstart', 'Xstop',
                               'Ystart', 'Ystop',
                               'Line Number'])

    # Row labels
    filament = 'Filament {:02d}'

    # Extract line ends and append to df
    for idx, end_pairs in enumerate(line_ends):
        for line_num, ends in enumerate(end_pairs):
            e1 = ends[0]  # Start coords
            e2 = ends[1]  # Stop coords
            new_values = {'Xstart': int(e1[1]),
                          'Xstop': int(e2[1]),
                          'Ystart': int(e1[0]),
                          'Ystop': int(e2[0]),
                          'Line Number': line_num + 1,
                          }
            next_row = pd.Series(new_values, name=filament.format(idx + 1))
            df = df.append(next_row)

    starfile.write(df, star_dir)

    # Print first few lines of the STAR file for sanity
    with open(star_dir) as file:
        for line in islice(file, 15):
            print(line)
