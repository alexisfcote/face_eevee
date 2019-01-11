def read_pts(filename):
    """A helper function to read the 68 ibug landmarks from a .pts file."""
    with open(filename) as f:
        lines = f.read().splitlines()
    lines = lines[3:71]

    shapes = []
    ibug_index = 1  # count from 1 to 68 for all ibug landmarks
    for l in lines:
        coords = l.split()
        shapes.append([float(coords[0]), float(coords[1])])

    return shapes