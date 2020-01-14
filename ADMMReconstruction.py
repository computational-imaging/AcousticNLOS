'''
MIT License

Copyright (c) 2018 Stanford Computational Imaging Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import util.lct as lct
import os
from util.pypher import psf2otf
from numpy.fft import ifftn, fftn
from types import SimpleNamespace
from time import time
import sys
matplotlib.rcParams.update({'font.size': 6})

plot_every = 5 
do_plots = True

def usage():
    print('Usage: python3 ADMMReconstruction.py [scene1] [scene2] ...')
    print('       valid scenes are')
    print('         \'double\'')
    print('         \'letter_H\'')
    print('         \'corner_reflectors\'')
    print('         \'letters_LT\'')
    return

def gaussian_kernel(size, sigma, size_y=None, size_z=None):
    size = int(size)
    if not size_y:
        size_y = size
    else:
        size_y = int(size_y)
    if not size_z:
       pass 
    else:
        size_z = int(size_z)

    if size_z is not None:
        x, y, z = np.mgrid[-size:size+1, -size_y:size_y+1, -size_z:size_z+1]
        g = np.exp(-(x**2/float(size)+y**2/float(size_y)+z**2/float(size_z)))
    else:
        x, y = np.mgrid[-size:size+1, -size_y:size_y+1]
        g = np.exp(-(x**2/float(sigma)+y**2/float(sigma)))

    return g / g.sum()

def shrinkage(vy, vx, vz, kappa):
        vv = np.sqrt(vy**2 + vx**2 + vz**2)
        zy = np.maximum(1 - kappa / abs(vv), 0) * vy
        zx = np.maximum(1 - kappa / abs(vv), 0) * vx
        zz = np.maximum(1 - kappa / abs(vv), 0) * vz
        return zy, zx, zz

def fconv(x, otf):
    return np.real(ifftn(fftn(x) * otf))

def run(scene):
    global plot_every 

    # set parameters
    finished = False
    S = 2
    x_max = 1 
    y_max = 1 
    snr = 8e3
    dx = 0.9 
    dy = 0.95 
    eps_abs = 1e-3
    eps_rel = 1e-3

    ckpt = np.load('data/' + scene + '_out.npy', allow_pickle=True)[()]    
    out_fname = scene + '_admm_out.npy'

    if scene == 'letter_H':
        rho = 1e-1
        lambda_l1 = 9e-1 
        lambda_tv = 2
        use_l1 = True 
        use_tv = True 
        use_lct = False 
        plot_every = 20 
 
    elif scene == 'corner_reflectors':
        use_l1 = True 
        use_tv = True 
        use_lct = True 
        rho = 1e-1
        lambda_l1 = 0.1 
        lambda_tv = 0.1 

    elif scene == 'double':
        use_l1 = True 
        use_tv = True 
        use_lct = True 
        rho = 1e-1
        lambda_l1 = 3e-1
        lambda_tv = 3e-1 

    elif scene == 'letters_LT':
        lambda_l1 = 0.9 
        lambda_tv = 2.0 
        use_l1 = True 
        use_tv = True 
        use_lct = False 
        rho = 1e-1
        plot_every = 30

    # set params
    dat = SimpleNamespace(**ckpt)
    Nx = dat.meas.shape[2] 
    Ny = dat.meas.shape[1] 
    Nt = dat.meas.shape[0] 

    max_dist = int(dat.T * dat.v/2 / 16)
    slope_x = dx * x_max / (dat.fend/dat.B * max_dist) * (1 + ((dat.fstart/dat.B) / (dat.fend/dat.B))**2)
    slope_y = dy * y_max / (dat.fend/dat.B * max_dist) * (1 + ((dat.fstart/dat.B) / (dat.fend/dat.B))**2)

    # gradient kernels
    d2 = np.array([[0, 0, 0],[0, 1, -1], [0, 0, 0]])
    d2 = np.pad(d2[:, :, None], ((0, 0), (0, 0), (1, 1)), 'constant')
    d1 = np.array([[0, 0, 0],[0, 1, 0], [0, -1, 0]])
    d1 = np.pad(d1[:, :, None], ((0, 0), (0, 0), (1, 1)), 'constant')
    d3 = np.zeros((3, 3, 3))
    d3[1, 1, 1] = 1
    d3[1, 1, 2] = -1

    # lct params
    def pad_array(x, S, Nt, Nx, Ny):
        return np.pad(x, ((S*Nt//2, S*Nt//2), (Ny//2, Ny//2), (Nx//2, Nx//2)), 'constant')

    def trim_array(x, S, Nt, Nx, Ny):
        return x[S*int(np.floor(Nt/2))+1:-S*int(np.ceil(Nt/2))+1, Ny//2+1:-Ny//2+1, Nx//2+1:-Nx//2+1]

    def p2o(x):
        return psf2otf(x, b.shape)

    b = dat.meas.copy()
    if use_lct:
        # get LCT kernel and calculate initial reconstruction
        A, AFT = lct.getPSF(Nx, Ny, Nt, S, slope_x, slope_y)
        mtx, mtxi = lct.interpMtx(Nt, S, dat.fstart/dat.B * max_dist, dat.fend/dat.B * max_dist)
        invpsf = np.conj(AFT) / (abs(AFT)**2 + 1 / snr)
        tdata = np.matmul(mtx, b.reshape((Nt, -1))).reshape((-1, Ny, Nx))

        fdata = fftn(pad_array(tdata, S, Nt, Nx, Ny))
        out = abs(ifftn(fdata * invpsf))
        x = out.copy()
        x_init = out.copy()
        x_init = trim_array(x_init, S, Nt, Nx, Ny)
        x_init = np.matmul(mtxi, x_init.reshape((S*Nt, -1))).reshape((-1, Ny, Nx))
        b = np.matmul(mtx, b.reshape((Nt, -1))).reshape((-1, Ny, Nx))
        b = pad_array(b, S, Nt, Nx, Ny)

        # add spatial blur as well
        A_blur = np.load('data/psf_gaussian.npy')
        A_blur = np.pad(A_blur[:, :, None], ((0, 0), (0, 0), (A_blur.shape[0]//2, A_blur.shape[0]//2)), 'constant')
        A_blur = np.transpose(A_blur, (2, 0, 1))
        AFT_blur = p2o(A_blur)
        AFT *= AFT_blur

    else:
        # load fitted gaussian kernel
        A = np.load('data/psf_gaussian.npy')
        A = np.pad(A[:, :, None], ((0, 0), (0, 0), (A.shape[0]//2, A.shape[0]//2)), 'constant')
        A = np.transpose(A, (2, 0, 1))
        AFT = p2o(A)
        x = b.copy()
        x_init = b.copy()

    # operator functions
    d1FT = p2o(d1)
    d2FT = p2o(d2)
    d3FT = p2o(d3)
    bFT = fftn(b)
    
    # run admm
    max_iters = 100000
    z1 = np.zeros((b.shape))
    z2 = np.zeros((b.shape))
    z3 = np.zeros((b.shape))
    z4 = np.zeros((b.shape))

    z1_prev = np.zeros((b.shape))
    z2_prev = np.zeros((b.shape))
    z3_prev = np.zeros((b.shape))
    z4_prev = np.zeros((b.shape))

    k1 = np.zeros((b.shape))
    k2 = np.zeros((b.shape))
    k3 = np.zeros((b.shape))
    k4 = np.zeros((b.shape))

    u1 = np.zeros((b.shape))
    u2 = np.zeros((b.shape))
    u3 = np.zeros((b.shape))
    u4 = np.zeros((b.shape))
    primal_residual = []
    dual_residual = []
    timing = []

    assert(use_l1 or use_tv)
    
    # precompute for x update
    denom = (np.conj(AFT) * AFT).astype(np.complex128)
    if use_l1:
        denom += rho

    if use_tv:
        denom += rho * (np.conj(d1FT) * d1FT + np.conj(d2FT) * d2FT + np.conj(d3FT) * d3FT)

    # run admm iterations
    for i in range(max_iters):
        
        start = time()
        # z1 update (sparsity)
        # save z_prev
        if use_l1:
            k1 = x.copy()
            v = x + u1;
            kappa = lambda_l1 / rho;
            z1 = np.maximum(v - kappa, 0) - np.maximum(-v - kappa, 0)   
            z1_prev = z1.copy()

        # z2-z4 update (tv)
        if use_tv:
            z2_prev = z2.copy()
            z3_prev = z3.copy()
            z4_prev = z4.copy()

            k2 = fconv(x, d1FT)
            k3 = fconv(x, d2FT)
            k4 = fconv(x, d3FT)

            vy = k2 + u2
            vx = k3 + u3
            vz = k4 + u4

            kappa = lambda_tv / rho;
            z2, z3, z4 = shrinkage(vy, vx, vz, kappa);

        # u update
        if use_l1:
            u1 = u1 + x - z1
        if use_tv:
            u2 = u2 + k2 - z2
            u3 = u3 + k3 - z3
            u4 = u4 + k4 - z4

        # calculate stopping criterion
        N_scale = 0
        kx_norm = 0
        z_norm = 0
        ktu = np.zeros(u1.shape)
        ktu_norm = 0
        resid_pri = 0
        resid_dual = 0
        if use_l1:
            N_scale += 1
            kx_norm += np.sum(k1.ravel()**2) 
            z_norm += np.sum(z1.ravel()**2)
            ktu += u1
            resid_pri += np.sum((k1.ravel() - z1.ravel())**2)
            resid_dual += z1 - z1_prev
        if use_tv:
            N_scale += 3
            kx_norm += np.sum(k2.ravel()**2 + k3.ravel()**2 + k4.ravel()**2)
            z_norm += np.sum(z2.ravel()**2 + z3.ravel()**2 + z4.ravel()**2)
            ktu += fconv(u2, np.conj(d1FT)) + fconv(u3, np.conj(d2FT)) + fconv(u4, np.conj(d3FT)) 
            resid_pri += np.sum((k2.ravel() - z2.ravel())**2
                                 + (k3.ravel() - z3.ravel())**2
                                 + (k4.ravel()- z4.ravel())**2)
            resid_dual += fconv(z2-z2_prev, np.conj(d1FT)) \
                          + fconv(z3-z3_prev, np.conj(d2FT)) \
                          + fconv(z4-z4_prev, np.conj(d3FT))

        resid_pri = np.sqrt(resid_pri)
        resid_dual *= rho
        resid_dual = np.sqrt(np.sum(resid_dual.ravel()**2))
        ktu_norm = np.sqrt(np.sum(ktu.ravel()**2))
        kx_norm = np.sqrt(kx_norm)
        z_norm = np.sqrt(z_norm)
        eps_pri = eps_abs * np.sqrt(N_scale*Nx*Ny*Nt) + eps_rel * np.maximum(kx_norm, z_norm)
        eps_dual = eps_abs * np.sqrt(Nx*Ny*Nt)  + eps_rel * ktu_norm

        print('Iter: {}'.format(i))
        print('===Primal/Dual Thresholds===')
        print('{:.02f}, {:.02f}'.format(eps_pri, eps_dual))
        print('===Primal/Dual Residuals===')
        print('{:.02f}, {:.02f}'.format(resid_pri, resid_dual))
        print()

        # check stopping criterion
        if resid_pri < eps_pri and resid_dual < eps_dual:
            if use_lct:
                x_out = trim_array(x, S, Nt, Nx, Ny)
                x_out = np.matmul(mtxi, x_out.reshape((S*Nt, -1))).reshape((-1, Ny, Nx))
            else:
                x_out = x.copy()

            out_dict = {'out': x_out, 'iter': i, 'primal_residual': primal_residual, 'timing': timing, 'dual_residual': dual_residual}
            np.save('chimera/raw/' + out_fname, out_dict)
            finished = True

        if not finished:
            # x update
            num = (np.conj(AFT) * bFT).astype(np.complex128)
            if use_l1:
                num += rho * fftn(z1 - u1)

            if use_tv:
                num += rho * (np.conj(d1FT) * fftn(z2 - u2) +
                              np.conj(d2FT) * fftn(z3 - u3) +
                              np.conj(d3FT) * fftn(z4 - u4))

            x = np.real(ifftn(num / denom))

            # record timing
            stop = time()
            timing.append(stop - start)
            primal_residual.append(resid_pri)
            dual_residual.append(resid_dual)

            if use_lct:
                x_out = trim_array(x, S, Nt, Nx, Ny)
                x_out = np.matmul(mtxi, x_out.reshape((S*Nt, -1))).reshape((-1, Ny, Nx))
            else:
                x_out = x.copy() 

        # plots
        if do_plots and i % plot_every == 0:
            plt.clf()
            plt.subplot(3,3,7)
            plt.imshow(np.max(x_out, axis=1).squeeze(), aspect='auto')
            plt.xlabel('x')
            plt.ylabel('t')
            plt.xticks([], [])
            plt.yticks([], [])
             
            plt.subplot(3,3,8)
            plt.title('Reconstruction')
            plt.imshow(np.max(x_out, axis=2).squeeze(), aspect='auto')
            plt.xlabel('y')
            plt.ylabel('t')
            plt.xticks([], [])
            plt.yticks([], [])

            plt.subplot(3,3,9)
            plt.imshow(np.max(x_out, axis=0).squeeze(), aspect='auto')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xticks([], [])
            plt.yticks([], [])
            plt.colorbar()

            plt.subplot(3,3,1)
            plt.title('Residuals')
            plt.plot(primal_residual)
            plt.plot(dual_residual)

            plt.subplot(3,3,2)
            plt.title('Initial Reconstruction')
            plt.imshow(np.max(x_init, axis=0).squeeze(), aspect='auto')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.xticks([], [])
            plt.yticks([], [])
            plt.colorbar()
             
            if finished:
                plt.pause(1)
                break
            else:
                plt.pause(0.01)

if __name__ == '__main__':
    valid_scenes = ['double', 'corner_reflectors',  'letter_H', 'letters_LT']
    scene = sys.argv[1:] 
    if len(scene) == 0:
        usage()

    if scene == ['all']:
        scene = valid_scenes

    for s in scene:
        if s not in valid_scenes:
            usage()
            break
        print('Processing scene: {}'.format(s))
        run(s)
