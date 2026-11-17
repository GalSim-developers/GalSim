# Copyright (c) 2012-2026 by the GalSim developers team on GitHub
# https://github.com/GalSim-developers
#
# This file is part of GalSim: The modular galaxy image simulation toolkit.
# https://github.com/GalSim-developers/GalSim
#
# GalSim is free software: redistribution and use in source and binary forms,
# with or without modification, are permitted provided that the following
# conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions, and the disclaimer given in the accompanying LICENSE
#    file.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions, and the disclaimer given in the documentation
#    and/or other materials provided with the distribution.
#

"""Make a TAN-SIP WCS from the WCS spreadsheet supplied by the project.

Run from devel/roman with the xlsx spreadsheet in this directory.

Note that the spreadsheet uses FPA coordinates rather than the natural science coordinates
used for the final image. The science coordiates have the opposite direction for x and the
same direction for y.
"""

import pandas
import string
import numpy as np
import scipy
import galsim
import coord

xlsx_name = "RST PhaseC (Pre CDR) WIMWSM Zernike and Field Data_20210204_d2.xlsx"
dist_sheet = "WIM Distortion Full Data"
efl_sheet = "Effective Focal Length"
sip_output_file = "../../share/roman/sip_20210204.txt"
pos_output_file = "../../share/roman/sca_positions_20210204.txt"

last_col = 'W'
ncol = ord(last_col) - ord('A') + 1

df = pandas.read_excel(xlsx_name,
                       sheet_name=dist_sheet,
                       header=None,
                       usecols=f'A:{last_col}',
                       names=string.ascii_uppercase[:ncol]
                       )
data = df.to_records()

# Extract the fitted distortion function
# Note: the excel row numbers start with 1, not 0, so row numbers are 1 smaller
# than they are in Excel.
distortion_fit = data['W'][5:1:-1].astype(float)
print('distortion fit = ',distortion_fit)

efl_paraxial_fit = data['M'][5]
field_bias_deg = data['M'][6]
field_bias_mm = field_bias_deg * np.pi/180 * efl_paraxial_fit
print('field_bias_mm = ',field_bias_mm)

# Start by making a new sca_positions file

num_sca = 18
sca_data = {}

for isca in range(num_sca):
    nsca = isca + 1  # nsca is 1-based index.

    # These are the 0,0 values in the spreadsheet.
    # Outputs are
    #   nca   XAN     YAN     FPA-X   FPA-Y
    # However, in the spreadsheet, the field bias is not applied.
    # We apply it for this output file.

    row = 123 + 225 * isca
    # Sanity checks:
    assert data['B'][row] == nsca
    assert data['C'][row] == 8
    assert data['D'][row] == 8
    assert data['I'][row] == 0
    assert data['J'][row] == 0

    xan = data['E'][row]
    yan = data['F'][row]
    fpa_x = data['K'][row]
    fpa_y = data['L'][row]

    # Save these for below.
    sca_data[isca] = (nsca, xan, yan, fpa_x, fpa_y)

# We need the effective focal length values from a different sheet.
df = pandas.read_excel(xlsx_name,
                       sheet_name=efl_sheet,
                       header=None,
                       usecols=f'D:E',
                       names=['X','Y']
                       )
efl_data = df.to_records()[7:]

# We now make a new sip file in the same format as the one we got from Jeff Kruk,
# including the same conventions for signs and such.
# However, we don't want the image coordinates in terms of FPA x,y. We want them to use
# the regular science coordinates, which has science x = -FPA x and science y = FPA y.

def extract_params(params):
    cr0, cr1, cd00, cd01, cd10, cd11, *ab = params
    #print('cr = ',cr0,cr1)
    #print('cd = ',cd00,cd01,cd10,cd11)

    crval = np.array([cr0, cr1])
    cd = np.array([[cd00, cd01], [cd10, cd11]])

    a = np.zeros((5,5), dtype=float)
    b = np.zeros((5,5), dtype=float)
    a[1,0] = 1
    a[0,2:5] = ab[0:3]
    a[1,1:4] = ab[3:6]
    a[2,0:3] = ab[6:9]
    a[3,0:2] = ab[9:11]
    a[4,0:1] = ab[11:12]
    b[0,1] = 1
    b[0,2:5] = ab[12:15]
    b[1,1:4] = ab[15:18]
    b[2,0:3] = ab[18:21]
    b[3,0:2] = ab[21:23]
    b[4,0:1] = ab[23:24]

    return crval, cd, a, b

def sip_resid(params, x, y, u, v):
    crval, cd, a, b = extract_params(params)

    # This basically follows the code in GSFitsWCS for how to apply the SIP coefficients.
    #print('start: params = ',params)
    #print('x = ',x[:20])
    #print('y = ',y[:20])
    #print('u = ',u[:20])
    #print('v = ',v[:20])
    x1 = galsim.utilities.horner2d(x, y, a, triangle=True)
    y1 = galsim.utilities.horner2d(x, y, b, triangle=True)
    #print('x1 = ',x1[:20])
    #print('y1 = ',y1[:20])

    u1 = cd[0,0] * x1 + cd[0,1] * y1
    v1 = cd[1,0] * x1 + cd[1,1] * y1
    #print('u1 = ',u1[:20])
    #print('v1 = ',v1[:20])

    # Do the TAN projection
    u1 *= -np.pi / 180.  # Minus sign here is also in GSFitsWCS. Due to FITS definition of cd.
    v1 *= np.pi / 180.
    #print('fpa_center = ',fpa_center)
    sca_center = fpa_center.deproject(crval[0]*coord.degrees, crval[1]*coord.degrees)
    #print('sca_center = ',sca_center)
    u2, v2 = sca_center.deproject_rad(u1, v1, projection='gnomonic')
    u2 *= -180. / np.pi  # deproject returns ra, which is in -u direction
    v2 *= 180. / np.pi

    # Return the array of residuals.
    return np.concatenate((u2 - u, v2 - v))

for isca in range(num_sca):
    nsca, xan, yan, fpa_x, fpa_y = sca_data[isca]
    print('SCA ',nsca)
    print('xan,yan = ',xan, yan)
    print('fpa = ',fpa_x, fpa_y)

    # Get the effective focal lengths in each direction for this SCA.
    efl_u = efl_data['X'][isca]
    efl_v = efl_data['Y'][isca]
    print('EFL = ',efl_u,efl_v)

    # Calculate the conversion from mm to degrees for the CD matrix below.
    # The x,y values we'll have below will be in mm, so divide by FL to get radians.
    # Then unit conversions:
    # deg/pix = (1/FL) * (m/mm) * (mm/pix) * (deg/radian)
    deg_per_pix_u = 1./efl_u * (1/1000) * 0.01 * 180./np.pi
    deg_per_pix_v = 1./efl_v * (1/1000) * 0.01 * 180./np.pi

    # The SIP A and B matrices define a transformation
    #
    # u =   A00     + A01 y     + A02 y^2     + A03 y^3   + A04 y^4
    #     + A10 x   + A11 x y   + A12 x y^2   + A13 x y^3
    #     + A20 x^2 + A21 x^2 y + A22 x^2 y^2
    #     + A30 x^3 + A31 x^3 y
    #     + A40 x^4
    # v =   B00     + B01 y     + B02 y^2     + B03 y^3   + B04 y^4
    #     + B10 x   + B11 x y   + B12 x y^2   + B13 x y^3
    #     + B20 x^2 + B11 x^2 y + B22 x^2 y^2
    #     + B30 x^3 + B31 x^3 y
    #     + B40 x^4

    # A00 and B00 are definitionally 0 in this context.
    # Also, Jeff defined things in the file so that A01, A10, B01, B10 in the file are
    # really the elements of the CD matrix, rather than use these numbers in the SIP matrices.
    # So really, we have A01 = A10 = B01 = B10 = 0 for the SIP step, but we figure
    # out a separate next step that looks like
    #
    # u' = C00 u + C01 v
    # v' = C10 u + C11 v
    #
    # which gets applied after the above equation.

    # Rather than try to back all this out of the radial fit, we just do our own
    # 2d fit using the values in the spreadsheet.

    first_row = 11 + 225 * isca
    last_row = 236 + 225 * isca  # one past the end
    center_row = 123 + 225 * isca
    # The spreadsheet positions use FPA coordinates. To convert to science coordinates,
    # multiply x by -1.
    x = -data['I'][first_row:last_row].astype(float) * 100
    y = data['J'][first_row:last_row].astype(float) * 100
    u = data['E'][first_row:last_row].astype(float)
    v = data['F'][first_row:last_row].astype(float)

    fpa_center = coord.CelestialCoord(0*coord.degrees, field_bias_deg*coord.degrees)

    guess = np.zeros(30)
    guess[0:2] = xan, yan - field_bias_deg
    guess[2] = 0.11 / 3600.
    guess[5] = -0.11 / 3600.

    # Rescale the input x,y to be order unity.  We'll rescale the SIP coefficients
    # after the fact to compensate. Note: the 2048 here is a numerical normalization,
    # not the science-image center. It's just a convenient power of 2 to use.
    x /= 2048
    y /= 2048

    result = scipy.optimize.least_squares(sip_resid, guess, args=(x,y,u,v))

    print(result)
    assert result.success
    resid = sip_resid(result.x, x, y, u, v)
    print('Final residuals = ', resid)
    rms = np.sqrt(np.sum(resid**2)/len(u))
    print(isca,': rms = ',rms)
    # The fit should be good to better than 0.1 arcsec rms.
    assert rms*3600 < 0.1

    crval, cd, a, b = extract_params(result.x)

    # Correct for the x,y rescaling we did above.
    cd /= 2048
    powers = np.array([[0,1,2,3,4],
                       [1,2,3,4,0],
                       [2,3,4,0,0],
                       [3,4,0,0,0],
                       [4,0,0,0,0]])
    # CD already contributes a factor of 1/2048. A degree-n SIP term therefore
    # needs 2048**(1-n), so CD times the polynomial reproduces the fitted map.
    a *= 2048.0**(1-powers)
    b *= 2048.0**(1-powers)

    # The fitted crval is pretty close to our initial guess, but might as well save the
    # best fit value in our positions file, rather than do the calculation in code
    # like we used to do.
    print('fitted crval = ',crval)
    print('cf ',xan, yan - field_bias_deg)
    print('fitted cd = ',cd.ravel())
    print('cf ', 0.11/3600)
    print('fitted a = ',a)
    print('fitted b = ',b)

    sca_data[isca] = nsca, xan, yan, fpa_x, fpa_y, crval[0], crval[1], cd, a, b


with open(pos_output_file, 'w') as fout:
    for isca in range(num_sca):
        fout.write(("%3d" + 2*"\t%10.5f" + 2*"\t%10.2f" + 2*"\t%10.6f" + "\n")
                   %(sca_data[isca][:7]))


with open(sip_output_file, 'w') as fout:
    fout.write('# Generated by devel/roman/make_sip_file.py from the 20210204 spreadsheet.\n')
    for isca in range(num_sca):
        # Write A matrix along with C00, C01
        _, _, _, _, _, _, _, cd, a, b = sca_data[isca]

        # Jeff's original version of this packaged the CD matrix in 4 of the zero locations
        # of the A and B matrices.  Keep the same format here.
        a[1,0] = cd[0,0]
        a[0,1] = cd[0,1]
        b[1,0] = cd[1,0]
        b[0,1] = cd[1,1]

        for k in range(5):
            fout.write(("%3d a  %d" + 5*" %16.9e" + "\n")%(isca, k, *a[k]))
        for k in range(5):
            fout.write(("%3d b  %d" + 5*" %16.9e" + "\n")%(isca, k, *b[k]))
