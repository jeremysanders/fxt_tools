# fxt_imgtool

## Jeremy Sanders

A tool for making images and exposure maps for Einstein Probe FXT data

Prerequisites:  Python 3, FXTSoft, forkqueue, cython, scipy, astropy, tqdm

Usage:

    fxt_imgtool --ra=RA --dec=DEC --pixsize=PIXSIZE --npix=NPIX --cores=CORES \
        --emin=MINENG --emax=MAXENG --energy=EXPMAPENG image evt.fits att.fits out_img.fits
    
or

    fxt_imgtool --ra=RA --dec=DEC --pixsize=PIXSIZE --npix=NPIX --cores=CORES \
        --emin=MINENG --emax=MAXENG --energy=EXPMAPENG expmap evt.fits att.fits out_exp.fits
  
Parameters:

 * `--ra=RA`: give RA of centre of image
 * `--dec=DEC`: give Dec of image centre
 * `--pixsize=PIXSIZE`: give size of pixels in arcsec
 * `--npix=NPIX`: dimensions of image in pixels (both dimensions)
 * `--cores=CORES`: number of cores to use for computation
 * `--emin=EMIN`: give minimum energy (in image mode)
 * `--emax=EMIN`: give maximum energy (in image mode)
 * `--energy=EXPMAPENG`: exposure map energy
 * `image` or `expmap`: mode to use
 * `evt.fits`: input event file
 * `att.fits`: input attitude
 * `out_img.fits` or `out_exp.fits`: output image or exposure map
