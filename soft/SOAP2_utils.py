import numpy as np
import astropy.io.fits as pf
from glob import glob

def read_ccfs(prefix, root = '../data/HARPN_Sun'):
    fls = np.sort(glob('{:s}/{:s}/CCF*.fits'.format(root, prefix)))
    print('{:s}/{:s}/CCF*.fits'.format(root, prefix))
    print(fls)
    nobs = len(fls)
    hdul = pf.open(fls[0])
    hdr = hdul[0].header
    nvel = int(hdr['NAXIS1'])
    vel = np.arange(nvel) * float(hdr['STEP']) 
    vel -= max(vel)/2.
    ccf = np.zeros((nobs,nvel))
    for i in range(nobs):
        hdul = pf.open(fls[i])
        ccf[i,:] = hdul[0].data.flatten()
        hdul.close()
    return vel, ccf
