#!/usr/bin/env python3

import os
import logging
import math
import argparse
import sys

import numpy as np
from astropy.io import fits
import astropy.wcs
import tqdm
import scipy.interpolate

import get_vign
import attitude
import fxt_getcalf

import pyximport
pyximport.install()
import paint_rects

import forkqueue

def setup_logging():
    global logger
    logger = logging.getLogger('fxt_imgtool')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class TelDef:
    def __init__(self, evtfname):
        datapathfile, dataxno = self.get_teldef(evtfname)
        CALDBpath = os.getenv('CALDB')
        filename = CALDBpath + "/" + datapathfile

        header = fits.getheader(filename)

        self.mij = mij = np.zeros((3, 3))
        mij[0, 0] = header['ALIGNM11']
        mij[0, 1] = header['ALIGNM12']
        mij[0, 2] = header['ALIGNM13']
        mij[1, 0] = header['ALIGNM21']
        mij[1, 1] = header['ALIGNM22']
        mij[1, 2] = header['ALIGNM23']
        mij[2, 0] = header['ALIGNM31']
        mij[2, 1] = header['ALIGNM32']
        mij[2, 2] = header['ALIGNM33']
        self.qtel = attitude.max_to_quat(mij)

        self.coeij = coeij = np.zeros((2,3))
        coeij[0, 0] = header['COE_X1_A']
        coeij[0, 1] = header['COE_X1_B']
        coeij[0, 2] = header['COE_X1_C']
        coeij[1, 0] = header['COE_Y1_A']
        coeij[1, 1] = header['COE_Y1_B']
        coeij[1, 2] = header['COE_Y1_C']

        self.det = {
            'xsiz': header['DET_XSIZ'],
            'xpix1': header['DETXPIX1'],
            'xscl': header['DET_XSCL'],
            'ysiz': header['DET_YSIZ'],
            'ypix1': header['DETYPIX1'],
            'yscl': header['DET_YSCL'],
        }

        self.sky = {
            'xsiz': header['SKY_XSIZ'],
            'xpix1': header['SKYXPIX1'],
            'ysiz': header['SKY_YSIZ'],
            'ypix1': header['SKYYPIX1'],
        }
        self.align = {
            'roll_sign': header['ROLLSIGN'],
            'roll_offset': header['ROLLOFF']
        }
        self.focallen = header['FOCALLEN']
        self.pixel_size = header["DET_XSCL"]
        self.optaxis= {
            'x': header['OPTAXISX'],
            'y': header['OPTAXISY']
        }

        self.imxscl = 206265 / self.focallen * self.det['xscl'] / 3600
        self.imyscl = 206265 / self.focallen * self.det['yscl'] / 3600

    def get_teldef(self, evtfname):
        fin = fits.open(evtfname)
        hdr = fin['EVENTS'].header

        te = hdr["TELESCOP"]
        ins = hdr["INSTRUME"]
        det = hdr["DETNAM"]
        filtemp = hdr["FILTER"]
        if type(filtemp)==str:
            fil = filtemp[-1]
        if type(filtemp)==int:
            fil = filtemp
        logger.debug(fil)
        if int(fil)>3:
            logger.debug("Warning:FILTER is >3!,the data is for radioactive source or all blocked.")
        fil = "None"
        date_obs = hdr["DATE-OBS"]
        date_end = hdr["DATE-END"]
        logger.debug(f"{date_obs,date_end}")
        startdate, starttime = date_obs.split("T")
        stopdate, stoptime = date_end.split("T")
        grade = 4
        cod = "TELDEF"
        try:
            grade = hdr['GRADE']
        except:
            logger.debug("No Grade keyword, use 4 default")
            pass
        datamode = hdr["DATAMODE"]
        logger.debug(f"{te,ins,det,fil,datamode}")
        fin.close()

        expr = None
        logger.debug(expr)
        datapathfile,dataxno = fxt_getcalf.getcalf(
            te,ins,det,fil,cod,startdate,starttime,stopdate,stoptime,
            expr,1,1,1,1,1,1,1,1)
        return datapathfile,dataxno


class BadPix:
    """Bad pixel mask from event file."""

    def __init__(self, evt_fname):
        self.mask = np.ones((384, 384), dtype=np.int32)

        # remove edges
        self.mask[0,:] = 0
        self.mask[383,:] = 0
        self.mask[:,383] = 0
        self.mask[:,0] = 0

        fin = fits.open(evt_fname)
        if 'BADPIX' not in fin:
            fin.close()
            logger.info(f"No badpix extension in {evt_fname}.")
            return

        data = fin['BADPIX'].data
        for badx, bady, badtype, badext in zip(
                data['RAWX'], data['RAWY'], data['TYPE'], data['YEXTENT']):
            if badtype == 1:
                for igroupx in range(3):
                    for igroupy in range(3):
                        tx = np.clip(badx+igroupx-1,0,383)
                        ty = np.clip(bady+igroupy-1,0,383)
                        self.mask[ty,tx] = 0
            elif badtype == 2:
                for bade in range(badext):
                    for ix in range(-1, 2):
                        for iy in range(-1, 2):
                            tx = np.clip(badx+ix,0,383)
                            ty = np.clip(bady+bade+iy,0,383)
                            self.mask[ty,tx] = 0

        # fixme: below might be wrong - need swapping X and Y
        datamode = fin['EVENTS'].header['DATAMODE']
        if datamode == "PW":
            logger.debug("datamode: %s"%datamode)
            self.mask[:,0:161] = 0
            self.mask[:,222:] = 0
            self.mask[0:128,:] = 0
            self.mask[256:,:] = 0
        elif datamode == "TM":
            logger.debug("datamode: %s"%datamode)
            self.mask[0:128,:] = 0
            self.mask[256:,:] = 0

class Vignetting:
    def __init__(self, evt_fname, teldef, energy_keV):
        """Get vignetting map."""

        logger.info("calculating vignetting rate map")
        idxfile, xno = get_vign.get_vign(evt_fname)
        logger.debug("{idxfile,xno}")
        vignpar = get_vign.get_vignpar(idxfile, xno)
        e0 = vignpar["ENERGY"]
        p0 = vignpar["COEF0"]
        p1 = vignpar["COEF1"]
        #p2 = vignpar["COEF2"][0]
        #logger.debug(p0,p1,p2)
        detimage = np.ones((384,384))
        detidx = np.argwhere(detimage>-11)
        det_center = [teldef.optaxis["x"]-1, teldef.optaxis["y"]-1]
        ralist  = ((detidx[:,1]-det_center[1]+1)**2+(detidx[:,0]-det_center[0]+1)**2)**0.5
        ramatrix = ralist.reshape(384,384)
        theta = ramatrix * teldef.imxscl * 60.
        thetaline = theta.flatten()
        vignrateflatten = np.zeros(len(thetaline))
        for i, te in enumerate(thetaline):
            vignratelist = []
            vignnum = 0
            for ij in range(p0.shape[0]):
                vignratetemp = self.vignfunc(te, p0[ij], p1[ij])
                vignratelist.append(vignratetemp)
                vignnum =vignnum+1
            if vignnum == 2:
                funcinter = scipy.interpolate.UnivariateSpline(
                    e0,np.array(vignratelist),k=1,s=0) #s = 0 : go through all the data point, s>0
            else:
                funcinter = scipy.interpolate.UnivariateSpline(
                    e0,np.array(vignratelist),k=2,s=0) #s = 0 : go through all the data point, s>0
            vignrateflatten[i] = funcinter(energy_keV)

        vignrate = vignrateflatten.reshape(384,384)
        vignrate[vignrate>1] = 1
        vignrate[vignrate<0] = 0
        self.vign = vignrate.T # note transpose!

    def vignfunc(self, x,c0,beta):
        return ((1+(x/c0)**2)**(-beta))

class Attitude:
    def __init__(self, atfname, teldef, evtfname, corrfname=None):
        fin = fits.open(atfname)
        logger.debug(f'Opening attitude: {atfname}')
        data = fin[1].data
        self.times = data["TIME"]
        self.ra_pnt = data["RA_PNT"]
        self.dec_pnt = data["DEC_PNT"]
        self.pa_pnt = data["PA_PNT"]
        fin.close()
        self.teldef = teldef

        # correct positions based on fits
        if corrfname:
            logger.debug(f'Opening correction {corrfname}')
            self.corr = np.loadtxt(corrfname)
            fin = fits.open(evtfname)
            ra = fin['events'].header['ra_pnt']
            dec = fin['events'].header['dec_pnt']
            fin.close()

            self.corr_wcs = astropy.wcs.WCS(header={
                'NAXIS': 2,
                'CRPIX1': 1, 'CRPIX2': 1,
                'CRVAL1': ra, 'CRVAL2': dec,
                'CDELT1': -1/3600, 'CDELT2': 1/3600,
                'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
                'CUNIT1': 'deg', 'CUNIT2': 'deg',
            })

        else:
            self.corr = self.corr_wcs = None

    def getQuat(self, t):
        ra = np.interp(t, self.times, self.ra_pnt)
        dec = np.interp(t, self.times, self.dec_pnt)
        pa = np.interp(t, self.times, self.pa_pnt)
        quat = attitude.eq_to_quat(ra, dec, pa)
        return quat

    def det2RaDec(self, t, row, column):
        axis_column = self.teldef.optaxis['x']-1
        axis_row = self.teldef.optaxis['y']-1

        ax = (axis_column-column) * self.teldef.pixel_size
        ay = (axis_row-row) * self.teldef.pixel_size
        az = np.ones(ax.shape)*(-self.teldef.focallen)
        q = self.getQuat(t)
        R = np.array(attitude.quaternion_to_matrix(q[0], q[1], q[2], q[3])).reshape(3, 3)
        b = np.matmul(R, np.matmul(self.teldef.mij, np.array([ax, ay, az])))

        ra, dec = attitude.cal_ra_dec(b)

        if self.corr is not None:
            # inverse of transformation of regions to events in fit
            n_x, n_y = self.corr_wcs.all_world2pix(ra, dec, 0)
            r_x, r_y, r_theta, delx, dely = self.corr
            s = np.sin(-r_theta/180*np.pi)
            c = np.cos(-r_theta/180*np.pi)
            dx = n_x + delx + r_x
            dy = n_y + dely + r_y
            rx = dx*c - dy*s - r_x
            ry = dx*s + dy*c - r_y
            ra, dec = self.corr_wcs.all_pix2world(rx, ry, 0)

        return ra, dec

class GTI:
    """Get GTIs from event filename."""

    def __init__(self, evtfname):
        fin = fits.open(evtfname)
        logger.debug(f'Opening events to read GTIs: {evtfname}')
        gti = fin['GTI']
        self.start = gti.data['START']
        self.stop = gti.data['STOP']
        fin.close()

    def getSteps(self, tstep=1.0):
        """Produce set of steps with interval given between start and stop times."""
        out = []
        for start, stop in zip(self.start, self.stop):
            lo = int(math.ceil(start/tstep))
            hi = int(math.floor(stop/tstep))
            assert hi >= lo
            tsteps = (np.arange(hi-lo+1) + lo)*tstep
            out.append(tsteps)
        out = np.concatenate(out)
        return out 

class ImgWCS:
    def __init__(self, ra0, dec0, pixsize_as=2.5, npix=2048, proj='SIN'):
        self.npix = npix
        self.wcs = w = astropy.wcs.WCS(naxis=2)
        w.wcs.crpix = np.array([npix/2, npix/2])
        w.wcs.cdelt = np.array([-pixsize_as/3600, pixsize_as/3600])
        w.wcs.crval = np.array([ra0, dec0])
        w.wcs.ctype = [f"RA---{proj}", f"DEC--{proj}"]
        w.wcs.cunit = ["deg", "deg"]

    def world2pix(self, ra, dec):
        return self.wcs.wcs_world2pix(ra, dec, 0)

def chunk_range(n, nchunk):
    out = []
    for i in range(n):
        out.append(i)
        if len(out) == nchunk:
            yield out
            out = []
    if out:
        yield out

def main_expmap(args, teldef, att, gti, badpix, wcs):

    vign = Vignetting(args.events, teldef, args.energy)

    size_e = (teldef.det['ysiz']+1,teldef.det['xsiz']+1)
    row_e, col_e = np.indices(size_e)+0.5
    row_er = np.ravel(row_e)
    col_er = np.ravel(col_e)

    ccd_img = badpix.mask * args.tstep * vign.vign
    tsteps = gti.getSteps(tstep=args.tstep)

    def inner(its):
        out = np.zeros((wcs.npix, wcs.npix))
        for it in its:
            ra_er, dec_er = att.det2RaDec(tsteps[it], row_er, col_er)
            x_er, y_er = wcs.world2pix(ra_er, dec_er)
            x_e = np.ascontiguousarray(x_er.reshape(size_e))
            y_e = np.ascontiguousarray(y_er.reshape(size_e))
            paint_rects.paint_rects(y_e, x_e, ccd_img, out)
        return out

    logger.info('Projecting exposure')
    outimg = np.zeros((wcs.npix, wcs.npix))
    innargs = [(x,) for x in chunk_range(len(tsteps), 16)]
    with forkqueue.ForkQueue(numforks=args.cores, ordered=False, env=locals()) as q:
        for res in tqdm.tqdm(q.process(inner, innargs), total=len(innargs)):
            outimg += res

    logger.info(f'Writing exposure map {args.outimage}')
    fout = fits.HDUList([fits.PrimaryHDU(
        data=outimg.astype(np.float32),
        header=wcs.wcs.to_header())])
    fout.writeto(args.outimage, overwrite=True)

def main_image(args, teldef, att, gti, badpix, wcs):

    logger.info(f'Reading events {args.events}')
    fevt = fits.open(args.events)
    evt = fevt['EVENTS']
    ccdy = evt.data['DETY']
    ccdx = evt.data['DETX']
    time = evt.data['TIME']
    energy = evt.data['ENG']

    # construct event mask
    mask = np.zeros(len(time), dtype=bool)
    for start, stop in zip(gti.start, gti.stop):
        mask |= ((time >= start) & (time <= stop))

    mask = (
        mask &
        (energy >= args.emin) & (energy < args.emax) &
        (badpix.mask[ccdy-1,ccdx-1] != 0)
        )

    time_m = time[mask]
    num = len(time_m)

    ccdy_m = ccdy[mask]
    ccdx_m = ccdx[mask]

    # randomize location of events within pixels
    rand_gen = np.random.default_rng(42)
    ccdy_m = ccdy_m + rand_gen.uniform(-0.499999, 0.499999, size=num)
    ccdx_m = ccdx_m + rand_gen.uniform(-0.499999, 0.499999, size=num)

    chunksize = 256
    nchunk = int(math.ceil(num / chunksize))
    innargs = ((i,) for i in range(nchunk))

    def inner(i):
        out_ra = []
        out_dec = []
        for j in range(i*chunksize, min(num, (i+1)*chunksize)):
            ra, dec = att.det2RaDec(time_m[j], ccdy_m[j:j+1], ccdx_m[j:j+1])
            out_ra.append(ra[0])
            out_dec.append(dec[0])
        return np.array(out_ra), np.array(out_dec)

    logger.info('Projecting events')
    ras = []
    decs = []
    with forkqueue.ForkQueue(numforks=args.cores, env=locals()) as q:
        for ra, dec in tqdm.tqdm(q.process(inner, innargs), smoothing=0, total=nchunk):
            ras.append(ra)
            decs.append(dec)
    ras = np.concatenate(ras)
    decs = np.concatenate(decs)

    ix, iy = wcs.world2pix(ras, decs)
    hist, e1, e2 = np.histogram2d(iy, ix, bins=np.arange(wcs.npix+1)-0.5)

    logger.info(f'Writing output image {args.outimage}')
    fout = fits.HDUList([fits.PrimaryHDU(data=hist, header=wcs.wcs.to_header())])
    fout.writeto(args.outimage, overwrite=True)

    if args.writeevt:
        evt.data = evt.data[mask]
        colx, coly = evt.data.columns['X'], evt.data.columns['Y']
        wcsout = astropy.wcs.WCS(header={
            'NAXIS': 2,
            'CRPIX1': colx.coord_ref_point, 'CRPIX2': coly.coord_ref_point,
            'CRVAL1': colx.coord_ref_value, 'CRVAL2': coly.coord_ref_value,
            'CDELT1': colx.coord_inc, 'CDELT2': coly.coord_inc,
            'CTYPE1': colx.coord_type, 'CTYPE2': coly.coord_type,
            'CUNIT1': 'deg', 'CUNIT2': 'deg',
        })
        xout, yout = wcsout.all_world2pix(ras, decs, 1)
        evt.data['X'][:] = xout
        evt.data['Y'][:] = yout
        logger.info(f'Writing output events {args.writeevt}')
        evt.header['HISTORY'] = 'Filtered and new coordinates created by fxt_imgtool.py'
        fevt.writeto(args.writeevt, overwrite=True)

    fevt.close()

def main():
    setup_logging()

    parser = argparse.ArgumentParser(
        prog='fxt_imgtool',
        description='Make images and exposure maps for EP FXT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--ra', type=float, help='central image R.A. (deg)', default=0.)
    parser.add_argument('--dec', type=float, help='central image Dec. (deg)', default=0.)
    parser.add_argument('--corr', help='apply correction file')
    parser.add_argument('--pixsize', type=float, help='pixel size (arcsec)', default=5.0)
    parser.add_argument('--npix', type=int, help='number of pixels', default=1024)
    parser.add_argument('--cores', type=int, help='parallel cores to use', default=8)
    parser.add_argument('--emin', type=float, help='minimum energy (for image)', default=0.2)
    parser.add_argument('--emax', type=float, help='maximum energy (for image)', default=7.0)
    parser.add_argument('--writeevt', help='output filtered event file with new coordinates (for image)')
    parser.add_argument('--energy', type=float, help='energy (for expmap)', default=1.0)
    parser.add_argument('--tstep', type=float, help='time step (for expmap)', default=5.0)

    parser.add_argument('mode', choices=('image', 'expmap'), help='program mode to use')
    parser.add_argument('events', help='input event file')
    parser.add_argument('attitude', help='input attitude file')
    parser.add_argument('outimage', help='output image file')

    args = parser.parse_args()

    logger.info('Setting up telescope definiton')
    teldef = TelDef(args.events)
    logger.info(f'Reading attitude {args.attitude}')
    att = Attitude(args.attitude, teldef, args.events, corrfname=args.corr)
    logger.info('Reading GTIs')
    gti = GTI(args.events)
    logger.info('Reading bad pixels')
    badpix = BadPix(args.events)

    wcs = ImgWCS(args.ra, args.dec, npix=args.npix, pixsize_as=args.pixsize)

    if args.mode == 'image':
        main_image(args, teldef, att, gti, badpix, wcs)
    elif args.mode == 'expmap':
        main_expmap(args, teldef, att, gti, badpix, wcs)


if __name__ == '__main__':
    main()
