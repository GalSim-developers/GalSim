# Shell script used by Rachel to generate the standards for comparison, which used to
# live in tests/roman_files (then called wfirst_files).  We no longer have these tests, since
# the Roman wcs changed, and we don't have the corresponding truth files to do this test.
# The defunct test still exists as roman.py:skip_roman_wcs(), but it is skipped (per the name).

# Note: this script makes files for all SCAs, so it's a lot of files.
# We will just use SCAs 2, 13, 7, and 18 for tests 1, 2, 3, and 4 respectively.
/Users/rmandelb/great3/wfirst/wcs/new/wfi_wcs_gen_0.4 -f 127.0 -70.0 160.0 /great3/wfirst/wcs/sip_422.txt /git/GalSim/tests/wfirst_files/test1_
/Users/rmandelb/great3/wfirst/wcs/new/wfi_wcs_gen_0.4 307.4 50.0 79.0 /great3/wfirst/wcs/sip_422.txt /git/GalSim/tests/wfirst_files/test2_
/Users/rmandelb/great3/wfirst/wcs/new/wfi_wcs_gen_0.4 -f -61.52 22.7 23.4 /great3/wfirst/wcs/sip_422.txt /git/GalSim/tests/wfirst_files/test3_
/Users/rmandelb/great3/wfirst/wcs/new/wfi_wcs_gen_0.4 0.0 0.0 -3.1 /great3/wfirst/wcs/sip_422.txt /git/GalSim/tests/wfirst_files/test4_
