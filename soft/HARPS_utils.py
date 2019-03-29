from __future__ import division, absolute_import, print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from glob import glob
from astropy.io import fits as pf
sns.set()
sns.set_style('white')

def find_2d(spec_dir, obj = None, blaze_dir = None, inst = 'HARPS'):
    '''Locate all the 2-d (order by order) HARPS or HARPS-N spectra in a
    directory, optionally for a specific object
    '''
    e2ds_all = np.sort(glob('{}/{}*_e2ds_A.fits'.format(spec_dir, inst)))
    print('Found {} e2ds files in directory {}'.format(len(e2ds_all),spec_dir))
    if obj:
        print('Will look for object name {}'.format(obj))
    if blaze_dir is None:
        blaze_dir = np.copy(spec_dir)
    print('Will look for blaze files in directory {}'.format(blaze_dir))    
    e2ds_sel = []
    blaze_sel = []
    for efn in e2ds_all:
        print('Processing e2ds file {}.'.format(efn))
        hdul = pf.open(efn)
        hdr = hdul[0].header
        if obj:
            objn = hdr['OBJECT']
            if objn != obj:
                print('Object name is {}, skipping.'.format(objn))
                continue
            else:
                print('Object name is {}, keeping.'.format(objn))
        if inst == 'HARPS':
            blf = hdr['ESO DRS BLAZE FILE']
        else:
            blf = hdr['TNG DRS BLAZE FILE']
        print('Blaze file name in headeris {}'.format(blf))
        bfn = '{}/{}'.format(blaze_dir, blf.replace(':','?'))
        bf = glob(bfn)
        if len(bf) == 0:
            print("Can't find blaze file {}, skipping.".format(bfn))
            continue
        else:
            print('Found blaze file {}.'.format(bfn))
        e2ds_sel.append(efn)
        blaze_sel.append(bf[0])
        hdul.close()
    K = len(e2ds_sel)
    if obj:
        print('Found {} 2-d spectra for object {} with blaze files'.format(K, obj))
    else:
        print('Found {} 2-d spectra with blaze files'.format(K))
    return e2ds_sel, blaze_sel

def read_2d(e2ds_files, blaze_files, inst = 'HARPS'):
    ''' Read spectra from a list of files and save in a dictionary'''
    if inst == 'HARPN':
        pref = 'TNG'
    else:
        pref = 'ESO'
    K = len(e2ds_files)
    baryvel = np.zeros(K)
    bjd = np.zeros(K)
    for k in range(K):
        print('Reading spectrum {} from file {}'.format(k+1, e2ds_files[k]))
        hdul = pf.open(e2ds_files[k])
        s = hdul[0].data
        if k == 0:
            M,N = s.shape
            wav_earth = np.zeros((K,M,N))
            flux = np.zeros((K,M,N))
            blaze = np.zeros((K,M,N))
        flux[k,:,:] = s
        hdr = hdul[0].header
        deg = int(hdr['{} DRS CAL TH DEG LL'.format(pref)])
        nc = deg+1
        x = np.arange(N)
        for j in range(M):
            for l in range(deg+1):
                A = float(hdr['{:} DRS CAL TH COEFF LL{:d}'.format(pref, j*(deg+1)+l)])
                wav_earth[k,j,:] += A * x**l    
        bjd[k] = float(hdr['{} DRS BJD'.format(pref)])
        baryvel[k] = float(hdr['{} DRS BERV'.format(pref)])
        hdul.close()
        hdul = pf.open(blaze_files[k])
        blaze[k,:,:] = hdul[0].data
        hdul.close()
    # compute photon noise errors
    flux[flux<=0] = np.nan
    flux_err = np.sqrt(flux)
    # divide by blaze function
    flux_bl = flux / blaze
    flux_bl_err = flux_err / blaze
    # apply barycentric RV correction to wavelengths
    lwav = np.log(wav_earth * 1e-10) # input wavelengths are in Angstrom
    dlw = baryvel / 2.99796458e5 # in km/s
    wav_bary = np.zeros((K,M,N))
    for k in np.arange(K):
        wav_bary[k,:,:] = np.exp(lwav[k,:,:] + dlw[k]) * 1e10
    # normalise spectra
    med = np.zeros((K,M))
    snr_order = np.zeros((K,M))
    flux_norm = np.zeros((K,M,N))
    flux_err_norm = np.zeros((K,M,N))
    for k in np.arange(K):
        for m in range(M):
            flx = flux_bl[k,m,:].flatten()
            flxe = flux_bl_err[k,m,:].flatten()
            l = np.isfinite(flx)
            flx = flx[l]
            flxe = flxe[l]
            s = np.argsort(flx/flxe)
            n = len(flx)
            med[k,m] = flx[s[int(0.98*n)]]
            snr_order[k,m] = med[k,m] / flxe[s[int(0.98*n)]]
            flux_norm[k,m,:] = flux_bl[k,m,:] / med[k,m]
            flux_err_norm[k,m,:] = flux_bl_err[k,m,:] / med[k,m]
    snr_av = np.nanmax(snr_order, axis = 0)
    m = np.argmax(snr_av)
    snr = snr_order[:,m].flatten()
    spec_dict = {'bjd': bjd, \
                     'baryvel': baryvel, \
                     'wav_earth': wav_earth, \
                     'wav_bary': wav_bary, \
                     'flux': flux, \
                     'flux_err': flux_err, \
                     'blaze': blaze, \
                     'flux_norm': flux_norm, \
                     'flux_err_norm': flux_err_norm, \
                     'order_med': med, \
                     'snr': snr}
    return spec_dict

def plot_2d(spec_dict, offset = 0.1, \
                wav_type = 'bary', norm = True, \
                epochs = None, epoch_range = None, \
                bjd_range = None, baryvel_range = None, \
                snr_range = [30, 10000], wav_range = None, \
                orders = None, order_range = None, \
                sort_by = 'bjd', col_by = 'bjd', \
                region = None, renorm = True, alpha = 1):

    # Specific regions    
    if region is 'Ca2H':
        orders = [5]
        wav_range = [3929,3939]
    elif region is 'Ca2K':
        orders = [7]
        wav_range = [3964,3974]
        renorm = False
    elif region is 'MgI4571':
        orders = [27]
        wav_range = [4570.5,4572.5]
    elif region is 'MgIb':
        orders = [43]
        wav_range = [5160,5190]
    elif region is 'NaID':
        orders = [56]
        wav_range = [5882,5903]
    elif region is 'Halpha':
        orders = [67]
        wav_range = [6559,6567]
        
    # Extract variables to plot
    wav = spec_dict["wav_{}".format(wav_type)]
    if norm:
        flux =  spec_dict["flux_norm"]
    else:
        flux =  spec_dict["flux"]
    snr = spec_dict['snr']
    bjd = spec_dict['bjd']
    baryvel = spec_dict['baryvel']
    
    K, M, N = wav.shape
    # Select by epoch, a.k.a. spectrum no (need to do this before sorting)
    if epochs:
        # list of epochs 
        bjd = bjd[epochs]
        baryvel = baryvel[epochs]
        snr = snr[epochs]
        wav = wav[epochs, :, :]
        flux = flux[epochs, :, :]
    else:
        epochs = np.arange(K).astype(int)
        if epoch_range: # range of epochs
            epochs = epochs[epoch_range[0]:epoch_range[1]+1]
            bjd = bjd[epoch_range[0]:epoch_range[1]+1]
            baryvel = baryvel[epoch_range[0]:epoch_range[1]+1]
            snr = snr[epoch_range[0]:epoch_range[1]+1]
            wav = wav[epoch_range[0]:epoch_range[1]+1, :, :]
            flux = flux[epoch_range[0]:epoch_range[1]+1, :, :]
    # Select by order no
    if orders:
        # list of orders 
        wav = wav[:, orders, :]
        flux = flux[:, orders, :]
    else:
        orders = np.arange(M).astype(int)
        if order_range: # range of orders
            orders = orders[order_range[0]:order_range[1]+1]
            wav = wav[:, order_range[0]:order_range[1]+1, :]
            flux = flux[:, order_range[0]:order_range[1]+1, :]
    
    # Sort as required
    s = np.arange(len(epochs)).astype(int)
    if sort_by is 'bjd':
        s = s[np.argsort(bjd)]
    elif sort_by is 'epoch':
        s = s[np.argsort(epochs)]
    elif sort_by is 'baryvel':
        s = s[np.argsort(baryvel)]
    elif sort_by is 'snr':
        s = s[np.argsort(snr)]
    bjd = bjd[s]
    epochs = epochs[s]
    baryvel = baryvel[s]
    snr = snr[s]
    wav = wav[s, :, :]
    flux = flux[s, :, :]

    # Select by bjd, baryvel and/or SNR
    l = np.ones(len(epochs), bool)
    if not(bjd_range is None):
        l *= (bjd >= bjd_range[0]) * (bjd < bjd_range[1])
    if not(baryvel_range is None):
        l *= (baryvel >= baryvel_range[0]) * (baryvel < baryvel_range[1])
    if not(snr_range is None):
        l *= (snr >= snr_range[0]) * (snr < snr_range[1])
    bjd = bjd[l]
    epochs = epochs[l]
    baryvel = baryvel[l]
    snr = snr[l]
    wav = wav[l, :, :]
    flux = flux[l, :, :]

    cmap = matplotlib.cm.get_cmap('viridis')
    if col_by is None:
        c = np.zeros_like(bjd)
    else:
        if col_by is 'bjd':
            c = bjd.copy()
        elif col_by is 'epoch':
            c = epochs.copy()
        elif col_by is 'baryvel':
            c = baryvel.copy()
        elif col_by is 'snr':
            c = snr.copy()
        c = (c - c.min()) / (c.max() - c.min())

    plt.figure(figsize = (10,5))
    wmin = 1e20
    wmax = 0
    for k, epoch in enumerate(epochs):
        off = offset * k
        col = cmap(c[k])
        for m, order in enumerate(orders):
            w = wav[k,m].flatten()
            f = flux[k,m].flatten()
            if wav_range:
                l = (w >= wav_range[0]) * (w < wav_range[1])
                w = w[l]
                f = f[l]
            wmin = min(w.min(), wmin)
            wmax = max(w.max(), wmax)
            if renorm:
                l = np.isfinite(f)
                p = np.polyfit(w[l],f[l],1)
                v = np.polyval(p,w)
                f /= v
            plt.plot(w, f - off, c = col, lw = 0.5,alpha=alpha)
    plt.xlim(wmin,wmax)
    if wav_type == 'bary':
        plt.xlabel(r'wavelength ($\AA$) [barycentric]')
    else:
        plt.xlabel(r'wavelength ($\AA$) [observatory]')
    if norm or renorm:
        plt.ylabel('Norm. flux')
    else:
        plt.ylabel('Flux')
    return 

def test_2d():
    spec_dir = '/Data/HARPS_Grenoble/HD215641'
    spec_dir = '/Data/HARPS_Grenoble/AU_Mic/DRS_spectra'
    
    obj = None
    blaze_dir = None
    inst = 'HARPS'
    e2ds, bl = find_2d(spec_dir, obj, blaze_dir, inst)
    spec_dict = read_2d(e2ds, bl, inst)
    return spec_dict

def read_1d(spec_dir, obj = None, inst = 'HARPS'):
    '''Locate and read the 1-d (orders merged) HARPS or HARPS-N spectra in a
    directory, optionally for a specific object, and save in a dictionary
    '''
    if inst == 'HARPN':
        pref = 'TNG'
    else:
        pref = 'ESO'
    print('{}/{}*_s1d_A.fits'.format(spec_dir, inst))
    s1d = glob('{}/{}*_s1d_A.fits'.format(spec_dir, inst))
    print('Found {} s1d files in directory {}'.format(len(s1d),spec_dir))
    if obj:
        print('Looking for object name {}'.format(obj))
        s1d_sel = []
        for sfn in s1d:
            print('Processing s1d file {}.'.format(sfn))
            hdul = pf.open(sfn)
            hdr = hdul[0].header
            objn = hdr['OBJECT']
            if objn == obj:
                print('Object name is {}, keeping.'.format(objn))
                s1d_sel.append(sfn)
            else:
                print('Object name is {}, skipping.'.format(objn))
            hdul.close()
        s1d = s1d_sel
        K = len(s1d)
        print('Found {} 1-d spectra for object {}'.format(K, obj))
    else:
        K = len(s1d)
        print('Found {} 1-d spectra'.format(K))
    baryvel = np.zeros(K)
    bjd = np.zeros(K)
    snr = np.zeros(K)
    for k in range(K):
        print('Reading spectrum {} from file {}'.format(k+1, s1d[k]))
        hdul = pf.open(s1d[k])
        hdr = hdul[0].header
        bjd[k] = float(hdr['{} DRS BJD'.format(pref)])
        baryvel[k] = float(hdr['{} DRS BERV'.format(pref)])
        snr[k] = float(hdr['{} DRS SPE EXT SN55'.format(pref)])
        if k == 0:
            N = int(hdr['NAXIS1'])
            flux = np.zeros((K,N))
            wav_earth = float(hdr['CRVAL1']) + \
              np.arange(N) * float(hdr['CDELT1'])
        flux[k,:] = hdul[0].data
        hdul.close()
    # apply barycentric RV correction to wavelengths
    lwav = np.log(wav_earth * 1e-10) # input wavelengths are in Angstrom
    dlw = baryvel / 2.99796458e5 # in km/s
    wav_bary = np.zeros((K,N))
    for k in np.arange(K):
        wav_bary[k,:] = np.exp(lwav + dlw[k]) * 1e10
    # use region around 550nm to normalise spectra
    l = abs(wav_earth-5500) <= 50
    med = np.median(flux[:,l], axis=1)
    flux_norm = flux / med[:,None]
    spec_dict = {'bjd': bjd, \
                     'baryvel': baryvel, \
                     'wav_earth': wav_earth, \
                     'wav_bary': wav_bary, \
                     'flux': flux, \
                     'flux_norm': flux_norm, \
                     'snr': snr}
    return spec_dict

def plot_1d(spec_dict, offset = 0.1, \
                wav_type = 'bary', norm = True, \
                epochs = None, epoch_range = None, \
                bjd_range = None, baryvel_range = None, \
                snr_range = [30, 10000], wav_range = None, \
                sort_by = 'bjd', col_by = 'bjd', \
                region = None, renorm = True, alpha = 1):

    # Specific regions    
    if region is 'Ca2H':
        wav_range = [3929,3939]
    elif region is 'Ca2K':
        wav_range = [3964,3974]
    elif region is 'MgI4571':
        wav_range = [4570.5,4572.5]
    elif region is 'MgIb':
        wav_range = [5160,5190]
    elif region is 'NaID':
        wav_range = [5882,5903]
    elif region is 'Halpha':
        wav_range = [6559,6567]
        
    # Extract variables to plot
    if norm:
        flux =  spec_dict["flux_norm"]
    else:
        flux =  spec_dict["flux"]
    K, N = flux.shape
    if wav_type == 'earth':
        wav = np.zeros((K,N))
        for k in range(K):
            wav[k,:] = spec_dict["wav_earth"]
    else:
        wav = spec_dict['wav_bary']
    snr = spec_dict['snr']
    bjd = spec_dict['bjd']
    baryvel = spec_dict['baryvel']
    
    # Select by epoch, a.k.a. spectrum no (need to do this before sorting)
    if epochs:
        # list of epochs 
        bjd = bjd[epochs]
        baryvel = baryvel[epochs]
        snr = snr[epochs]
        wav = wav[epochs, :]
        flux = flux[epochs, :]
    else:
        epochs = np.arange(K).astype(int)
        if epoch_range: # range of epochs
            epochs = epochs[epoch_range[0]:epoch_range[1]+1]
            bjd = bjd[epoch_range[0]:epoch_range[1]+1]
            baryvel = baryvel[epoch_range[0]:epoch_range[1]+1]
            snr = snr[epoch_range[0]:epoch_range[1]+1]
            wav = wav[epoch_range[0]:epoch_range[1]+1, :]
            flux = flux[epoch_range[0]:epoch_range[1]+1, :]
    
    # Sort as required
    s = np.arange(len(epochs)).astype(int)
    if sort_by is 'bjd':
        s = s[np.argsort(bjd)]
    elif sort_by is 'epoch':
        s = s[np.argsort(epochs)]
    elif sort_by is 'baryvel':
        s = s[np.argsort(baryvel)]
    elif sort_by is 'snr':
        s = s[np.argsort(snr)]
    bjd = bjd[s]
    epochs = epochs[s]
    baryvel = baryvel[s]
    snr = snr[s]
    wav = wav[s, :]
    flux = flux[s, :]

    # Select by bjd, baryvel and/or SNR
    l = np.ones(len(epochs), bool)
    if not(bjd_range is None):
        l *= (bjd >= bjd_range[0]) * (bjd < bjd_range[1])
    if not(baryvel_range is None):
        l *= (baryvel >= baryvel_range[0]) * (baryvel < baryvel_range[1])
    if not(snr_range is None):
        l *= (snr >= snr_range[0]) * (snr < snr_range[1])
    bjd = bjd[l]
    epochs = epochs[l]
    baryvel = baryvel[l]
    snr = snr[l]
    wav = wav[l, :]
    flux = flux[l, :]

    cmap = matplotlib.cm.get_cmap('viridis')
    if col_by is None:
        c = np.zeros_like(bjd)
    else:
        if col_by is 'bjd':
            c = bjd.copy()
        elif col_by is 'epoch':
            c = epochs.copy()
        elif col_by is 'baryvel':
            c = baryvel.copy()
        elif col_by is 'snr':
            c = snr.copy()
        c = (c - c.min()) / (c.max() - c.min())

    plt.figure(figsize = (10,5))
    wmin = 1e20
    wmax = 0
    for k, epoch in enumerate(epochs):
        off = offset * k
        col = cmap(c[k])
        w = wav[k,:].flatten()
        f = flux[k,:].flatten()
        if wav_range:
            l = (w >= wav_range[0]) * (w < wav_range[1])
            w = w[l]
            f = f[l]
        wmin = min(w.min(), wmin)
        wmax = max(w.max(), wmax)
        if renorm:
            l = np.isfinite(f)
            p = np.polyfit(w[l],f[l],1)
            v = np.polyval(p,w)
            f /= v
        plt.plot(w, f - off, c = col, lw = 0.5,alpha=alpha)
    plt.xlim(wmin,wmax)
    if wav_type == 'bary':
        plt.xlabel(r'wavelength ($\AA$) [barycentric]')
    else:
        plt.xlabel(r'wavelength ($\AA$) [observatory]')
    if norm or renorm:
        plt.ylabel('Norm. flux')
    else:
        plt.ylabel('Flux')
    return 

def test_1d():
    spec_dir = '/Data/HARPSN_Kelt9'
    
    obj = None
    inst = 'HARPN'
    spec_dict = read_1d(spec_dir, obj, inst)
    return spec_dict

