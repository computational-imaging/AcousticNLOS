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
import matplotlib.pyplot as plt
import scipy.optimize as opt

def gaussian2d(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple                                                        
    xo = float(xo)                                                              
    yo = float(yo)                                                              
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)   
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)    
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)   
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo)         
                        + c*((y-yo)**2)))                                   
    return g.ravel()

def fit_gaussian():
    # load data
    meas = np.load('./data/psf_out.npy', allow_pickle=True)[()]
    out = np.max(meas['out'], axis=0)
    x_len = out.shape[1]
    y_len = out.shape[0]
    out = out.ravel()
    x = np.linspace(0, 1, x_len)
    y = np.linspace(0, 1, y_len)
    x,y = np.meshgrid(x, y)

    initial_guess = (50,0.5,0.5,0.3,0.3,0,0.1)
    gaussian2d((x, y), *initial_guess)
    popt, pcov = opt.curve_fit(gaussian2d, (x, y), out, p0=initial_guess)

    data_fitted = gaussian2d((x, y), *popt)

    plt.style.use('dark_background')
    plt.subplot(121)
    plt.title('PSF')
    plt.imshow(out.reshape(y_len, x_len), origin='bottom',
        extent=(x.min(), x.max(), y.min(), y.max()))
    plt.contour(x, y, data_fitted.reshape(y_len, x_len), 4, colors='w')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('x')
    plt.ylabel('y')
 
    plt.subplot(122)    
    plt.title('Fitted PSF')
    popt[6] = 0 # offset
    popt[0] = 1 # amplitude
    popt[1] = np.max(x) / 2. # x0
    popt[2] = np.max(y) / 2. # y0

    data_fitted = gaussian2d((x, y), *popt).reshape(y_len, x_len)
    plt.imshow(data_fitted)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    
    np.save('data/psf_gaussian_params.npy', popt)
    np.save('data/psf_gaussian.npy', data_fitted)

if __name__ == '__main__':
    fit_gaussian()
