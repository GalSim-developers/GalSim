; Copyright (c) 2012-2018 by the GalSim developers team on GitHub
; https://github.com/GalSim-developers
;
; This file is part of GalSim: The modular galaxy image simulation toolkit.
; https://github.com/GalSim-developers/GalSim
;
; GalSim is free software: redistribution and use in source and binary forms,
; with or without modification, are permitted provided that the following
; conditions are met:
;
; 1. Redistributions of source code must retain the above copyright notice, this
;    list of conditions, and the disclaimer given in the accompanying LICENSE
;    file.
; 2. Redistributions in binary form must reproduce the above copyright notice,
;    this list of conditions, and the disclaimer given in the documentation
;    and/or other materials provided with the distribution.
;
PRO makestamps

;; IDL script used by Rachel to make the postage stamps for the first
;; 100 real galaxies.  This includes some approximations that are
;; undesirable in the long term (how much padding is needed, etc.) but
;; is good enough for a start.

; define filenames etc.
catfile = 'real_galaxy_catalog_23.5_example.fits'
listfile = '/u/rmandelb/sdss_systematics/cosmos/imgs.var.in'
outgalfile = 'real_galaxy_images.fits'
outpsffile = 'real_galaxy_PSF_images.fits'

; read in catalog file, others
cat = mrdfits(catfile,1)
ncat = n_elements(cat)
print,'Read in ',ncat,' from ',catfile

readcol,listfile,imgfilename, imgpref,ident,format='A,A,L'
nlist = n_elements(imgfilename)
print,'Read in ',nlist,' from file ',listfile

cosdata = mrdfits('/scr1/rmandelb/cosmos/Aug2010/acs_clean_only.fits',1,columns=['ident','background'])
ncosdata = n_elements(cosdata)
print,'Read in ',ncosdata,' COSMOS galaxies total from catalog'

if file_test(outgalfile) then print,'Error, output gal file already exists'
if file_test(outpsffile) then print,'Error, output PSF file already exists'

for i=0L,ncat-1 do begin
; find filenames for each ident
    wthis = where(ident eq cat[i].ident,c)

    if (c ne 1) then begin
        print,'Error: wrong number of matches made with ',cat[i].ident 
    endif else begin
        
; read in image, PSF, seg
        inimgfile = imgfilename[wthis[0]]
        inpsffile = imgpref[wthis[0]]+'.psf.fits'
        insegfile = imgpref[wthis[0]]+'_seg.fits'
        img = mrdfits(inimgfile,0)
        print,'Read from file ',inimgfile
        psf = mrdfits(inpsffile,0)
        print,'Read from file ',inpsffile
        seg = mrdfits(insegfile,0)
        print,'Read from file ',insegfile
        shapelets_read_sexcat, sextractorcat, imgpref[wthis[0]],/full_path

; subtract off background
        wback = where(cosdata.ident eq cat[i].ident,c)
        if (c ne 1) then begin
            print,'Error: wrong number of matches made with ACS for ',cat[i].ident
            backval = 0.
        endif else backval = cosdata[wback].background
        img -= backval

; mask other things
        ;; first identify the central object in catalog
        tmpsize = size(img)
        npix = float(tmpsize[1])
        centroidval = 0.5*npix
        distx = sextractorcat.x[*,0]-centroidval
        disty = sextractorcat.x[*,1]-centroidval
        dist = sqrt(distx^2 + disty^2)
        w = where(dist eq min(dist))
        myindx=1+sextractorcat.id[w[0]]
        wother = where(seg ne 0 AND seg ne myindx[0],countother)
        if (countother gt 0) then img[wother]=0

; do the trimming, compared to enlarged seg region
                ;; now find pixels belonging to it
        wimg = where(seg eq myindx[0],countin,complement=wnot)

        ;; find that region in x, y, and expand its size by factor of
        ;; 1.25 (unless that includes > the whole image)
        arrind = array_indices(seg, wimg)
        sortxind = sort(arrind[0,*])
        minx = arrind[0,sortxind[0.05*n_elements(sortxind)]]
        maxx = arrind[0,sortxind[0.95*n_elements(sortxind)]]
        xplussize = maxx-centroidval
        xminussize = centroidval-minx
        xsize = max(xminussize, xplussize)
        minx = round(centroidval - 1.25*xsize) > 0
        maxx = round(centroidval + 1.25*xsize) < tmpsize[1]-1

        sortyind = sort(arrind[1,*])
        miny = arrind[1,sortyind[0.05*n_elements(sortyind)]]
        maxy = arrind[1,sortyind[0.95*n_elements(sortyind)]]
        yplussize = maxy-centroidval
        yminussize = centroidval-miny
        ysize = max(yminussize, yplussize)
        miny = round(centroidval - 1.25*ysize) > 0
        maxy = round(centroidval + 1.25*ysize) < tmpsize[1]-1

; for PSF, cut it off where flux is <1000x peak
        tmpsize = size(psf)
        centroidval = 0.5*tmpsize[1]
        wkeep = where(psf ge 0.001*max(psf),nkeep)
        arrind = array_indices(psf,wkeep)
        minxp = min(arrind[0,*])
        maxxp = max(arrind[0,*])
        xplussize = maxxp-centroidval
        xminussize = centroidval-minxp
        xsize = max(xminussize, xplussize)
        minxp = round(centroidval - xsize) > 0
        maxxp = round(centroidval + xsize) < tmpsize[1]-1
        minyp = min(arrind[1,*])
        maxyp = max(arrind[1,*])
        yplussize = maxyp-centroidval
        yminussize = centroidval-minyp
        ysize = max(yminussize, yplussize)
        minyp = round(centroidval - ysize) > 0
        maxyp = round(centroidval + ysize) < tmpsize[1]-1
        

; write to file in different HDUs - this happens sequentially by not
;                                   using the /create keywrod
        print,'Writing to files!'
        mwrfits,img[minx:maxx,miny:maxy],outgalfile
        mwrfits,psf[minxp:maxxp,minyp:maxyp],outpsffile
        
    endelse
endfor

END
