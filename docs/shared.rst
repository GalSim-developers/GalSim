Shared Data
###########

GalSim includes some ancillary data along with the installed code.  They are installed in
the sub-directory ``share`` wherever your GalSim module is installed.

The location of these files are given by the variable

.. py:data:: galsim.meta_data.share_dir

    The installed location of your ``share`` directory.

.. note::

    Normally, the installation process will write a file called ``meta_data.py`` which sets
    the above variable automatically to the installation directory on your machine.

    However, if you install into a temporary location and then move the entire galsim directory to
    a different location, this will not have the correct location.  (It will instead be set to the
    temporary location.)  If you do this, you should define an environment variable,
    GALSIM_SHARE_DIR, to the correct location of the ``share`` directory.  Alternatively, you can
    set the value of `galsim.meta_data.share_dir` by hand in any Python programs that might need
    it.

Usually, you do not need to use the `galsim.meta_data.share_dir` variable directly.  Routines
that open files that might be in the ``share`` directory will automatically prepend this directory
name to the given file name when trying to open the file.

The following files are distributed in the ``share`` directory.  In each case, we provide the
command you would typically use to load the file with the appropriate GalSim class or function.

Shared SED files
================

vega.txt
    Use ``galsim.SED('vega.txt', wave_type='nm', flux_type='flam')``

    Specrum of the star Vega (aka Alpha Lyra), derived from HST CALSPEC data.
    File taken from http://www.stsci.edu/hst/observatory/crds/calspec.html
    Filename: alpha_lyr_mod_001.fits
    Clipped on the red side at 2200 nm
    Units converted to nm and erg/s/cm^2/nm.

CWW_E_ext.sed
    Use ``galsim.SED('CWW_E_ext.sed', wave_type='A', flux_type='flam')``

    E SED of Coleman, Wu, and Weedman (1980)
    Extended below 1400 A and beyond 10000 A by
    Bolzonella, Miralles, and Pello (2000) using evolutionary models
    of Bruzual and Charlot (1993)

    Obtained from ZPHOT code at

    http://webast.ast.obs-mip.fr/hyperz/zphot_src_1.1.tar.gz

    Truncated to wavelengths less than 22050 Angstroms, and thinned by
    galsim.utilities.thin_tabulated_values to a relative error of 1.e-5
    with fast_search=False.  See devel/modules/getSEDs.py for details.

CWW_E_ext_more.sed
    Use ``galsim.SED('CWW_E_ext_more.sed', wave_type='A', flux_type='flam')``

    Same as CWW_E_ext.sed, but thinned to a relative error of 1.e-3

CWW_Im_ext.sed
    Use ``galsim.SED('CWW_Im_ext.sed', wave_type='A', flux_type='flam')``

    Im SED of Coleman, Wu, and Weedman (1980)
    Extended below 1400 A and beyond 10000 A by
    Bolzonella, Miralles, and Pello (2000) using evolutionary models
    of Bruzual and Charlot (1993)

    Obtained from ZPHOT code at
    'http://webast.ast.obs-mip.fr/hyperz/zphot_src_1.1.tar.gz'

    Truncated to wavelengths less than 22050 Angstroms, and thinned by
    galsim.utilities.thin_tabulated_values to a relative error of 1.e-5
    with fast_search=False.  See devel/modules/getSEDs.py for details.

CWW_Im_ext_more.sed
    Use ``galsim.SED('CWW_Im_ext_more.sed', wave_type='A', flux_type='flam')``

    Same as CWW_Im_ext.sed, but thinned to a relative error of 1.e-3

CWW_Sbc_ext.sed
    Use ``galsim.SED('CWW_Sbc_ext.sed', wave_type='A', flux_type='flam')``

    Sbc SED of Coleman, Wu, and Weedman (1980)
    Extended below 1400 A and beyond 10000 A by
    Bolzonella, Miralles, and Pello (2000) using evolutionary models
    of Bruzual and Charlot (1993)

    Obtained from ZPHOT code at
    'http://webast.ast.obs-mip.fr/hyperz/zphot_src_1.1.tar.gz'

    Truncated to wavelengths less than 22050 Angstroms, and thinned by
    galsim.utilities.thin_tabulated_values to a relative error of 1.e-5
    with fast_search=False.  See devel/modules/getSEDs.py for details.

CWW_Sbc_ext_more.sed
    Use ``galsim.SED('CWW_Sbc_ext_more.sed', wave_type='A', flux_type='flam')``

    Same as CWW_Sbc_ext.sed, but thinned to a relative error of 1.e-3

For more details about how the above files were generated, see the script:

    GalSim/devel/getSEDs.py

Shared Bandpass files
=====================

ACS_wfc_F435W.dat
    Use ``galsim.Bandpass('ACS_wfc_F435W.dat', wave_type='nm')``

    ACS wfc_F435W total throughput
    File taken from http://www.stsci.edu/hst/acs/analysis/throughputs/tables/wfc_F435W.dat

    Thinned by galsim.utilities.thin_tabulated_values to a relative error of 1.e-3
    with fast_search=False.

ACS_wfc_F606W.dat
    Use ``galsim.Bandpass('ACS_wfc_F606W.dat', wave_type='nm')``

    ACS wfc_F606W total throughput
    File taken from http://www.stsci.edu/hst/acs/analysis/throughputs/tables/wfc_F606W.dat

    Thinned by galsim.utilities.thin_tabulated_values to a relative error of 1.e-3
    with fast_search=False.

ACS_wfc_F775W.dat
    Use ``galsim.Bandpass('ACS_wfc_F775W.dat', wave_type='nm')``

    ACS wfc_F775W total throughput
    File taken from http://www.stsci.edu/hst/acs/analysis/throughputs/tables/wfc_F775W.dat

    Thinned by galsim.utilities.thin_tabulated_values to a relative error of 1.e-3
    with fast_search=False.

ACS_wfc_F814W.dat
    Use ``galsim.Bandpass('ACS_wfc_F814W.dat', wave_type='nm')``

    ACS wfc_F814W total throughput
    File taken from http://www.stsci.edu/hst/acs/analysis/throughputs/tables/wfc_F814W.dat

    Thinned by galsim.utilities.thin_tabulated_values to a relative error of 1.e-3
    with fast_search=False.

ACS_wfc_F850LP.dat
    Use ``galsim.Bandpass('ACS_wfc_F850LP.dat', wave_type='nm')``

    ACS wfc_F850LP total throughput
    File taken from http://www.stsci.edu/hst/acs/analysis/throughputs/tables/wfc_F850LP.dat

    Thinned by galsim.utilities.thin_tabulated_values to a relative error of 1.e-3
    with fast_search=False.

LSST_u.dat
    Use ``galsim.Bandpass('LSST_u.dat', wave_type='nm')``

    LSST u-band total throughput at airmass 1.2
    File taken from https://raw.githubusercontent.com/lsst/throughputs/master/baseline/total_u.dat

    Thinned by galsim.utilities.thin_tabulated_values to a relative error of 1.e-3
    with fast_search=False.

LSST_g.dat
    Use ``galsim.Bandpass('LSST_g.dat', wave_type='nm')``

    LSST g-band total throughput at airmass 1.2
    File taken from https://raw.githubusercontent.com/lsst/throughputs/master/baseline/total_g.dat

    Thinned by galsim.utilities.thin_tabulated_values to a relative error of 1.e-3
    with fast_search=False.

LSST_r.dat
    Use ``galsim.Bandpass('LSST_r.dat', wave_type='nm')``

    LSST r-band total throughput at airmass 1.2
    File taken from https://raw.githubusercontent.com/lsst/throughputs/master/baseline/total_r.dat

    Thinned by galsim.utilities.thin_tabulated_values to a relative error of 1.e-3
    with fast_search=False.

LSST_i.dat
    Use ``galsim.Bandpass('LSST_i.dat', wave_type='nm')``

    LSST i-band total throughput at airmass 1.2
    File taken from https://raw.githubusercontent.com/lsst/throughputs/master/baseline/total_i.dat

    Thinned by galsim.utilities.thin_tabulated_values to a relative error of 1.e-3
    with fast_search=False.

LSST_z.dat
    Use ``galsim.Bandpass('LSST_z.dat', wave_type='nm')``

    LSST z-band total throughput at airmass 1.2
    File taken from https://raw.githubusercontent.com/lsst/throughputs/master/baseline/total_z.dat

    Thinned by galsim.utilities.thin_tabulated_values to a relative error of 1.e-3
    with fast_search=False.

LSST_y.dat
    Use ``galsim.Bandpass('LSST_y.dat', wave_type='nm')``

    LSST Y-band total throughput at airmass 1.2
    File taken from https://raw.githubusercontent.com/lsst/throughputs/master/baseline/total_y.dat

    Thinned by galsim.utilities.thin_tabulated_values to a relative error of 1.e-3
    with fast_search=False.

WFC3_uvis_F275W.dat
    Use ``galsim.Bandpass('WFC_uvis_F275W.dat', wave_type='nm')``

    WFC3 UVIS f275w total throughput
    Average of UVIS1 and UVIS2 throughputs, from files
    http://www.stsci.edu/hst/wfc3/ins_performance/throughputs/Throughput_Tables/f275w.UVIS1.tab
    http://www.stsci.edu/hst/wfc3/ins_performance/throughputs/Throughput_Tables/f275w.UVIS2.tab

    Thinned by galsim.utilities.thin_tabulated_values to a relative error of 1.e-3
    with fast_search=False.

WFC3_uvis_F336W.dat
    Use ``galsim.Bandpass('WFC_uvis_F336W.dat', wave_type='nm')``

    WFC3 UVIS f336w total throughput
    Average of UVIS1 and UVIS2 throughputs, from files
    http://www.stsci.edu/hst/wfc3/ins_performance/throughputs/Throughput_Tables/f336w.UVIS1.tab
    http://www.stsci.edu/hst/wfc3/ins_performance/throughputs/Throughput_Tables/f336w.UVIS2.tab

    Thinned by galsim.utilities.thin_tabulated_values to a relative error of 1.e-3
    with fast_search=False.

WFC3_ir_F105W.dat
    Use ``galsim.Bandpass('WFC_ir_F105W.dat', wave_type='nm')``

    WFC3 IR f105w total throughput
    File taken from http://www.stsci.edu/hst/wfc3/ins_performance/throughputs/Throughput_Tables/f105w.IR.tab

    Thinned by galsim.utilities.thin_tabulated_values to a relative error of 1.e-3
    with fast_search=False.

WFC3_ir_F125W.dat
    Use ``galsim.Bandpass('WFC_ir_F125W.dat', wave_type='nm')``

    WFC3 IR f125w total throughput
    File taken from http://www.stsci.edu/hst/wfc3/ins_performance/throughputs/Throughput_Tables/f125w.IR.tab

    Thinned by galsim.utilities.thin_tabulated_values to a relative error of 1.e-3
    with fast_search=False.

WFC3_ir_F160W.dat
    Use ``galsim.Bandpass('WFC_ir_F160W.dat', wave_type='nm')``

    WFC3 IR f160w total throughput
    File taken from http://www.stsci.edu/hst/wfc3/ins_performance/throughputs/Throughput_Tables/f160w.IR.tab

    Thinned by galsim.utilities.thin_tabulated_values to a relative error of 1.e-3
    with fast_search=False.

For more details about how the above files were generated, see the scripts:

    * GalSim/devel/getLSSTBandpass.py
    * GalSim/devel/getACSBandpass.py
    * GalSim/devel/getWFC3Bandpass.py


Shared Sensor models
=====================

lsst_itl_8
    Use ``galsim.SiliconSensor('lsst_itl_8')``

    The ITL sensor being used for LSST, using 8 points along each side of the
    pixel boundaries.

lsst_itl_32
    Use ``galsim.SiliconSensor('lsst_itl_32')``

    The ITL sensor being used for LSST, using 32 points along each side of the
    pixel boundaries.  (This is more accurate than the lsst_itl_8, but slower.)

lsst_etv_32
    Use ``galsim.SiliconSensor('lsst_etv_32')``

    The ETV sensor being used for LSST, using 32 points along each side of the
    pixel boundaries.  (This file is still somewhat preliminary and may be
    updated in the future.)


Shared HST noise model
======================

acs_I_unrot_sci_20_cf.fits
    Use ``galsim.getCOSMOSNoise()``


Shared Roman ST files
=====================

Roman_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727_01.txt
    Use ``galsim.roman.getPSF(1, bandpass)``

    Roman PSF information for SCA 1

Roman_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727_02.txt
    Use ``galsim.roman.getPSF(2, bandpass)``

    Roman PSF information for SCA 2

Roman_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727_03.txt
    Use ``galsim.roman.getPSF(3, bandpass)``

    Roman PSF information for SCA 3

Roman_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727_04.txt
    Use ``galsim.roman.getPSF(4, bandpass)``

    Roman PSF information for SCA 4

Roman_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727_05.txt
    Use ``galsim.roman.getPSF(5, bandpass)``

    Roman PSF information for SCA 5

Roman_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727_06.txt
    Use ``galsim.roman.getPSF(6, bandpass)``

    Roman PSF information for SCA 6

Roman_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727_07.txt
    Use ``galsim.roman.getPSF(7, bandpass)``

    Roman PSF information for SCA 7

Roman_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727_08.txt
    Use ``galsim.roman.getPSF(8, bandpass)``

    Roman PSF information for SCA 8

Roman_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727_09.txt
    Use ``galsim.roman.getPSF(9, bandpass)``

    Roman PSF information for SCA 9

Roman_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727_10.txt
    Use ``galsim.roman.getPSF(10, bandpass)``

    Roman PSF information for SCA 10

Roman_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727_11.txt
    Use ``galsim.roman.getPSF(11, bandpass)``

    Roman PSF information for SCA 11

Roman_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727_12.txt
    Use ``galsim.roman.getPSF(12, bandpass)``

    Roman PSF information for SCA 12

Roman_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727_13.txt
    Use ``galsim.roman.getPSF(13, bandpass)``

    Roman PSF information for SCA 13

Roman_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727_14.txt
    Use ``galsim.roman.getPSF(14, bandpass)``

    Roman PSF information for SCA 14

Roman_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727_15.txt
    Use ``galsim.roman.getPSF(15, bandpass)``

    Roman PSF information for SCA 15

Roman_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727_16.txt
    Use ``galsim.roman.getPSF(16, bandpass)``

    Roman PSF information for SCA 16

Roman_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727_17.txt
    Use ``galsim.roman.getPSF(17, bandpass)``

    Roman PSF information for SCA 17

Roman_Phase-A_SRR_WFC_Zernike_and_Field_Data_170727_18.txt
    Use ``galsim.roman.getPSF(18, bandpass)``

    Roman PSF information for SCA 18

Roman_SRR_WFC_Pupil_Mask_Shortwave_2048_reformatted.fits.gz
    Use ``galsim.roman.getPSF(sca, bandpass)``

    Roman Pupil Mask for the shorter wavelength bandpasses.
    Relevant for bands Z087, Y106, J129, and H158

Roman_SRR_WFC_Pupil_Mask_Longwave_2048_reformatted.fits.gz
    Use ``galsim.roman.getPSF(sca, bandpass)``

    Roman Pupil Mask for the longer wavelength bandpasses.
    Relevant for bands F184 and W149

afta_throughput.txt
    Use ``galsim.roman.getBandpasses()``

    Roman throughputs for all the Roman bands in a single file.

sip_7_6_8.txt
    Use ``galsim.roman.getWCS(world_pos)``

    Roman ST WCS information for all SCAs.

Shared COSMOS files
===================

These files are not shipped with GalSim, but can be installed into the ``share`` directory
by the executable ``galsim_download_cosmos``.  See `Downloading the COSMOS Catalog` for details.

COSMOS_25.2_training_sample
    | Use ``galsim.RealGalaxyCatalog(sample=25.2)``
    | Or ``galsim.COSMOSCatalog(sample=25.2)``

    Download with ``galsim_download_cosmos -s 25.2``

    A directory containing files for creating a `RealGalaxyCatalog` or a `COSMOSCatalog` using the
    F814W < 25.2 sample.

COSMOS_23.5_training_sample
    | Use ``galsim.RealGalaxyCatalog(sample=23.5)``
    | Or ``galsim.COSMOSCatalog(sample=23.5)``

    Download with ``galsim_download_cosmos -s 23.5``

    A directory containing files for creating a `RealGalaxyCatalog` or a `COSMOSCatalog` using the
    F814W < 23.5 sample.
