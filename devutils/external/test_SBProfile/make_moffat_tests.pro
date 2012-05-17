FUNCTION sinc, x
q = where(abs(x) gt 1.d-14, COMPLEMENT=notq)
out = dblarr(size(x, /DIM))
out[q] = sin(!dpi * x[q]) / x[q] / !dpi
out[notq] = 1.d0
return, out
END

PRO rotatexy, x, y, theta_rad, xp, yp
c = cos(theta_rad)
s = sin(theta_rad)
xp =  x * c + y * s
yp = -x * s + y * c
END

PRO shearxy, x, y, g1, g2, xp, yp
xp = x * g1 + y * g2
yp = x * g2 - y * g1
END

PRO make_dblxarr, n, x1, x2, X0=x0
; Set reference point to the centre if not specified
if not keyword_set(x0) then x0 = .5d0 * double(n)
; Make x arrays
one = replicate(1.d0, n[1] > 1)
vec = dindgen(n[0] > 1) - x0[0] + .5d0  ; x(0) to centre and 0.5 to get into middle of pixel
x1  = vec#one
; Make y arrays
one = replicate(1.d0, n[0] > 1)
vec = dindgen(n[1] > 1) - x0[1] + .5d0
x2  = one#vec
END

PRO pix_convolve, image, imagepix, PIXSIZE=pixsize, NPAD=npad, THETA_RAD=theta_rad
if not keyword_set(pixsize) then message, "Set PIXSIZE keyword."
s = size(image, /DIM)
if not keyword_set(npad) then npad = 2L
impad = dblarr(s * npad)
impad[0L:s[0] - 1L, 0L:s[1] - 1L] = image
fftim = temporary(fft(impad, -1, /DOUBLE))
make_dblxarr, s * npad, ux, uy, X0=[0.5d0, 0.5d0]
ux = temporary(ux - double((s[0] * npad) / 2L))
uy = temporary(uy - double((s[1] * npad) / 2L))
ux = temporary(shift(ux, (s[0] * npad + 1L) / 2L) / double(s[0] * npad))
uy = temporary(shift(uy, 0L, (s[1] * npad + 1L) / 2L) / double(s[1] * npad))
rotatexy, ux, uy, theta_rad, uxp, uyp
sincim = temporary(dcomplex(sinc(uxp * double(pixsize)) * $
                            sinc(uyp * double(pixsize))))
inv_ft = temporary(fft(fftim * sincim, +1, /DOUBLE))
im_conv = temporary(abs(inv_ft))
imagepix = temporary(im_conv[0:s[0]-1L, 0:s[1]-1L] / $
                     total(im_conv[0:s[0]-1L, 0:s[1]-1L]))
END

FUNCTION moffat, x, y, fwhm, beta, trunc_nfwhm
rs2 = .25d0 * fwhm * fwhm / (1.d0 + 2.d0^(1.d0 / beta))
r2 = (x^2 + y^2)
im = (1.d0 + r2 / rs2)^(-beta) / !dpi / rs2 / (beta - 1.d0)
im[where(r2 gt (trunc_nfwhm * fwhm)^2)] = 0.d0
return, im / total(im)
END


PRO make_moffat_tests

nim=256  ; image size
dx = 0.2d0
fwhm = 1.d0 / dx
trunc_nfwhm = 4.d0
beta = 1.5d0

uratio = max([1L, ceil(7.d0 / fwhm)])
if (uratio mod 2) eq 0 then uratio += 1L
nup = uratio * nim ; ensure we get at least 7 samples per FWHM for 
                   ; reference image
print, uratio

make_dblxarr, [nup, nup], x, y, X0=[0.5d0, 0.5d0]
x = temporary(x - double(nup / 2L))  ; put x,y so that (0,0) is in BLHS pixel 
y = temporary(y - double(nup / 2L))
x = temporary(shift(x, (nup + 1L) / 2L))
y = temporary(shift(y, 0L, (nup + 1L) / 2L))
x /= double(uratio) ; put coords in pixel units
y /= double(uratio)
mofim = moffat(x, y, fwhm, beta, trunc_nfwhm)
pwin, 0
plt_image, alog10(mofim + 1.d-7), /COL,/FRAME
pix_convolve, mofim, mofimpix, PIXSIZE=double(uratio), NPAD=2L, THETA_RAD=0.d0
pwin, 1
plt_image, alog10(mofimpix + 1.d-7), /COL,/FRAME
print, total(mofimpix)

outim = mofim[0L:nup-1L:uratio, 0L:nup-1L:uratio] / dx^2
outim = shift(outim, size(outim, /DIM) / 2L)
pwin, 2
plt_image, outim, /COL, /FRAME
fits_write, 'out.fits', outim
END


