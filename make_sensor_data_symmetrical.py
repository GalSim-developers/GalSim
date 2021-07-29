#!/usr/bin/env python
#
# This script makes the sensor data symmetrical, both vertically and horizontally.
#
# Usage:
#   make_sensor_data_symmetrical.py <dat filename>
#
# The output filename is the same as the input but with ".sym" appended.
#

import sys

# formats a float as a string to 4dp, padded with spaces to 15 characters
def formatFloat(val):
    result = "%.4f" % val
    result = result + (" " * (15 - len(result)))
    return result

# returns the horizontal mirror of the point index
#def mirrorNHorizontal(n, nv2):
#    return n

def mirrorNHorizontal(n, nv2):
    if n <= nv2:
        # lower left
        return (3*nv2 + 1) + (nv2 - n)
    elif n < (3*nv2 + 1):
        # bottom
        return (3*nv2) - (n-(nv2+1))
    elif n <= (4*nv2 + 1):
        # lower right
        return nv2 - (n-(3*nv2+1))
    elif n <= (5*nv2 + 2):
        # upper right
        return (8*nv2+3) - (n-(4*nv2+2))
    elif n < (7*nv2 + 3):
        # top
        return (7*nv2+2) - (n-(5*nv2+3))
    # upper left
    return (5*nv2+2) - (n-(7*nv2+3))

#def mirrorNVertical(n, nv2):
#    return n

# returns the vertical mirror of the point index
def mirrorNVertical(n, nv2):
    if n <= nv2:
        # lower left
        return (8*nv2+3) - n
    elif n < (3*nv2+1):
        # bottom
        return (7*nv2+2) - (n-(nv2+1))
    elif n <= (5*nv2+2):
        # right
        return (5*nv2+2) - (n-(3*nv2+1))
    elif n < (7*nv2+3):
        # top
        return (3*nv2) - (n-(5*nv2+3))
    # upper left
    return nv2 - (n-(7*nv2+3))

if len(sys.argv) != 2:
    print("Usage: make_sensor_data_symmetrical.py <dat filename>")
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

# create a result array and initialise it to the contents of the sensor file, normalised
result = []
for x in range(0, 9):
    for y in range(0, 9):
        for n in range(0, nv):
            idx = (x * 9 * nv) + (y * nv) + n
            result.append( points[idx][3] - points[idx][0] )
            result.append( points[idx][4] - points[idx][1] )

# add mirrored versions
for x in range(0, 9):
    for y in range(0, 9):
        for n in range(0, nv):
            idx = (x * 9 * nv) + (y * nv) + n
            
            # add data mirrored in x
            idxM = ((8 - x) * 9 * nv) + (y * nv) + mirrorNHorizontal(n, nv2)
            result[idx*2] = result[idx*2] - (points[idxM][3] - points[idxM][0])
            result[idx*2+1] = result[idx*2+1] + (points[idxM][4] - points[idxM][1])
    
            # add data mirrored in y
            idxM = (x * 9 * nv) + ((8 - y) * nv) + mirrorNVertical(n, nv2)
            result[idx*2] = result[idx*2] + (points[idxM][3] - points[idxM][0])
            result[idx*2+1] = result[idx*2+1] - (points[idxM][4] - points[idxM][1])

            # add data mirrored in both axes
            idxM = ((8 - x) * 9 * nv) + ((8 - y) * nv) + mirrorNVertical(mirrorNHorizontal(n, nv2), nv2)
            result[idx*2] = result[idx*2] - (points[idxM][3] - points[idxM][0])
            result[idx*2+1] = result[idx*2+1] - (points[idxM][4] - points[idxM][1])

# write output file including averaged data
f = open(sys.argv[1] + ".sym", "w")
f.write("X0             Y0             Theta          X              Y              \n")
for x in range(0, 9):
    for y in range(0, 9):
        for n in range(0, nv):
            idx = (x * 9 * nv) + (y * nv) + n
            X0 = points[idx][0]
            Y0 = points[idx][1]
            Theta = points[idx][2]
            X = (result[idx*2] * 0.25) + X0
            Y = (result[idx*2+1] * 0.25) + Y0
            f.write(formatFloat(X0) + formatFloat(Y0) + formatFloat(Theta) + formatFloat(X) + formatFloat(Y) + "\n")
f.close()
