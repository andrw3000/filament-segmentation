import pandas as pd
from itertools import islice


def coords2star(line_ends, star_dir):
    """Function to convert the output of the instance segmentation to STAR.

    Args:
        `line_ends` containins coordinates obtainined via `get_line_instances`.
        `star_dir` is the location for the output STAR file.
    """

    import starfile

    # Generate blank data frame with the headers
    df = pd.DataFrame(columns=['_rlnCoordinateX',
                               '_rlnCoordinateY',
                               '_rlnClassNumber',
                               '_rlnAnglePsi',
                               '_rlnAutopickFigureOfMerit'])

    # Row labels
    filament = 'Filament {:02d}'

    # Extract line ends and append to df
    for idx, end_pairs in enumerate(line_ends):
        for line_num, ends in enumerate(end_pairs):
            e1 = ends[0]  # Start coords
            e2 = ends[1]  # Stop coords

            new_values_start = {'_rlnCoordinateX': int(e1[1]),
                                '_rlnCoordinateY': int(e1[0]),
                                '_rlnClassNumber': line_num + 1,
                                '_rlnAnglePsi' : None,
                                '_rlnAutopickFigureOfMerit' : None,
                                }

            new_values_stop = {'_rlnCoordinateX': int(e2[1]),
                               '_rlnCoordinateY': int(e2[0]),
                               '_rlnClassNumber': line_num + 1,
                               '_rlnAnglePsi': None,
                               '_rlnAutopickFigureOfMerit': None,
                               }

            next_row_start = pd.Series(
                new_values_start, name=filament.format(idx + 1)
            )

            next_row_stop = pd.Series(
                new_values_stop, name=filament.format(idx + 1)
            )

            df = df.append(next_row_start)
            df = df.append(next_row_stop)

    starfile.write(df, star_dir, overwrite=True)

    # Print first few lines of the STAR file for sanity
    with open(star_dir) as file:
        for line in islice(file, 25):
            print(line)
