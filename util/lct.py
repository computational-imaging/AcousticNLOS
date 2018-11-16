import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, ifftshift, fftn, ifftn

def getPSF(X, Y, Z, S, slope_x, slope_y):
    x = np.linspace(-1, 1, 2*X)
    y = np.linspace(-1, 1, 2*Y)
    z = np.linspace(0,2,2*S*Z)
    grid_z, grid_y, grid_x = np.meshgrid(z, y, x, indexing='ij')
    psf = np.abs(slope_x**2 * grid_x**2 + slope_y**2 * grid_y**2 - grid_z)

    psf = psf == np.tile(np.min(psf, axis=0, keepdims=True), (2*S*Z, 1, 1)) 
    psf = psf.astype(np.float32)
    psf = psf / np.sum(psf)

    psf = np.roll(psf, X, axis=2)
    psf = np.roll(psf, Y, axis=1)
    fpsf = fftn(psf)

    return psf, fpsf

def interpMtx(M, N, start, stop):
    x = np.linspace(start, stop, M)# - start
    x = x ** 2
    val = np.eye(M)
    xq = np.linspace(start**2, stop**2, M*N)# - start**2

    mtx = np.zeros((M*N, M))
    for i in range(mtx.shape[1]):
        mtx[:, i] = np.interp(xq, x, val[:, i])

    mtx = mtx / np.tile(np.sum(mtx, axis=0, keepdims=True), (mtx.shape[0], 1))
    mtx[np.isnan(mtx)] = 0
    mtxi = mtx.T
    mtxi = mtxi / np.tile(np.sum(mtxi, axis=0, keepdims=True), (mtxi.shape[0], 1))
    mtxi[np.isnan(mtxi)] = 0

    return mtx, mtxi

def lct(x1, y1, t1, v, vol, snr):
    X = len(x1)
    Y = len(y1)
    Z = len(t1)
    S = 2
    slope = np.max(x1) / (np.max(t1) * v/2)
    slope = np.max(y1) / (np.max(t1) * v/2)
    psf, fpsf = getPSF(X, Y, Z, S, slope)
    mtx, mtxi = interpMtx(Z, S, 0, np.max(t1)*v)

    def pad_array(x, S, Z, X):
        return np.pad(x, ((S*Z//2, S*Z//2), (X//2, X//2), (Y//2, Y//2)), 'constant')

    def trim_array(x, S, Z, X):
        return x[S*(Z//2)+1:-S*(Z//2)+1, X//2+1:-X//2+1, Y//2+1:-Y//2+1]

    invpsf = np.conj(fpsf) / (abs(fpsf)**2 + 1 / snr)
    tdata = np.matmul(mtx, vol)
    fdata = fftn(pad_array(tdata, S, Z, X))
    tvol = abs(trim_array(ifftn(fdata * invpsf), S, Z, X))
    vol = np.matmul(mtxi, tvol)
    return vol
