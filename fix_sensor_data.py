#!/usr/bin/env python
#
# This script fixes discrepancies between the boundary points of adjacent pixels
# in the sensor data files. It should work for any number of vertices per pixel,
# but assumes a 9x9 grid of pixels centered on (15,15) to (95,95).
#
# Usage:
#   fix_sensor_data.py <dat filename>
#
# The output filename is the same as the input but with ".fixed" appended.
#

import sys

# formats a float as a string to 4dp, padded with spaces to 15 characters
def formatFloat(val):
    result = "%.4f" % val
    result = result + (" " * (15 - len(result)))
    return result

if len(sys.argv) != 2:
    print("Usage: fix_sensor_data.py <dat filename>")
    sys.exit(1)

# read input file and work out size
f = open(sys.argv[1], 'r')
lines = f.readlines()
f.close()
print("Read", len(lines), "lines from", sys.argv[1])
nv = int((len(lines) - 1) / 81)
print("nv=", nv)
numVertices = int((nv - 4) / 4)
print("numVertices=", numVertices)
nv2 = int(numVertices / 2)

# parse it into a list of lists of data
points = []

# skip header
for line in lines[1:]:
    bits = line.split()
    if len(bits) != 5:
        print("Unexpected line in sensor file:", line)
        sys.exit(1)
    # X0 and Y0 (the first two items) are the pixel center co-ordinates
    # they run from (15,15) to (95,95) and are spaced 10 units apart
    points.append( [float(bits[0]), float(bits[1]), float(bits[2]), float(bits[3]), float(bits[4])] )

    
# apply correction
# correct top and bottom rows (including corners)
# loop over all columns in data
for x in range(0, 9):
    # loop down the rows of this column starting at the bottom
    for y in range(0, 8):
        is1 = ((x * 9) + y) * nv      # starting index of lower pixel
        is2 = ((x * 9) + y + 1) * nv  # starting index of upper pixel
        # loop over the vertices in between these pixels (+2 to include corners)
        for n in range(0, numVertices + 2):
            # calculate top index in lower pixel
            i1 = is1 + ((7 * nv2) + 3) - n
            # calculate bottom index in upper pixel
            i2 = is2 + nv2 + n
            # work out average position
            xavg = (points[i1][3] + points[i2][3]) * 0.5
            yavg = (points[i1][4] + points[i2][4]) * 0.5
            # store for both pixels
            points[i1][3] = xavg
            points[i2][3] = xavg
            points[i1][4] = yavg
            points[i2][4] = yavg

# correct left and right rows
# loop over all rows in data including corners
for y in range(0, 9):
    # loop across the columns of this row
    for x in range(0, 8):
        is1 = ((x * 9) + y) * nv        # starting index of left pixel
        is2 = (((x + 1) * 9) + y) * nv  # starting index of right pixel
        # loop over the vertices in between these pixels (+2 to include corners)
        for n in range(0, numVertices + 2):
            # calculate right index in left pixel
            i1 = is1 + ((3 * nv2) + 1) + n
            # calculate left index in right pixel
            if n < (nv2 + 1):
                # in bottom half
                i2 = is2 + nv2 - n
            else:
                # in top half
                i2 = is2 + ((8 * nv2) + 3) - (n - (nv2+1))
            # work out average position
            xavg = (points[i1][3] + points[i2][3]) * 0.5
            yavg = (points[i1][4] + points[i2][4]) * 0.5
            # store for both pixels
            points[i1][3] = xavg
            points[i2][3] = xavg
            points[i1][4] = yavg
            points[i2][4] = yavg

# write file back out
f = open(sys.argv[1] + ".fixed", "w")
f.write("X0             Y0             Theta          X              Y              \n")
for point in points:
    f.write(formatFloat(point[0]) +
            formatFloat(point[1]) +
            formatFloat(point[2]) +
            formatFloat(point[3]) +
            formatFloat(point[4]) + "\n")
f.close()
