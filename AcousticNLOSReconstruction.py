import numpy as np
from numpy.fft import ifftn, fftn
import matplotlib.pyplot as plt
import util.lct as lct
import pickle
from tqdm import tqdm
from scipy.signal import firwin, lfilter
import scipy.signal
import csv
import os
from util.pickle_util import *
import sys
import time
plt.style.use('dark_background')

class AcousticNLOSReconstruction:
    
    def __init__(self):
        # enable plotting functions
        self.enable_plotting = 1
        self.T = 1.0  # chirp length = 1 s
        self.v = 340  # speed of sound
        self.fs = 48000 # sample frequency = 48 kHz
        self.f0 = 2000 # chirp start frequency = 2 kHz 
        self.f1 = 20000 # chirp end frequency = 20 kHz
        self.channels = [16, 16] # 16 transmit/receive channels
        self.samples_per_chirp = int(self.fs*self.T / 16) 
        self.B = self.f1 - self.f0
        self.f = np.linspace(0, self.fs, self.samples_per_chirp);
        self.t = np.linspace(0, self.T, int(self.fs*self.T))

        self.bg_offset = 0 # dark image calibration parameter
        self.snr = 8e3  # LCT calibration parameters
        self.dx =  0.9 
        self.dy =  .95 

        self.hp, self.lp = self.getFilterParameters(1.5e3, self.B) # high pass and low pass filters 
        self.calibration = self.getCalibrationArray()
        self.chirp = self.getChirpSignal()

    # retrieve calibration information from text files for the measurement microphones
    def getCalibrationArray(self):
        mic_ids = [15454, 15715, 15558, 15466, 15555,
                   15556, 15496, 15492, 15720,
                   15499, 15449, 15491, 15713,
                   15711, 15500, 15553] 
        calibration_freq = np.zeros((256, self.channels[0]))
        calibration_val = np.zeros((256, self.channels[0]))

        for idx, mic_id in enumerate(mic_ids):
            if not mic_id:
                continue
            with open('calibration_files/' + str(mic_id) + '.txt', 'r') as fid:
                reader = csv.reader(fid, dialect="excel-tab")
                reader = list(reader)[2:]
                calibration_freq[:, idx] = np.array([np.float32(reader[i][0]) for i in range(256)])
                calibration_val[:, idx] = np.array([np.float32(reader[i][1]) for i in range(256)])

        # interpolate to our frequencies
        fx = np.linspace(-self.fs/2, self.fs/2, self.samples_per_chirp)
        x = np.abs(np.fft.fftshift(fx))
        calibration = np.zeros((self.samples_per_chirp, self.channels[0]))
        for idx in range(self.channels[0]):
            calibration[:, idx] = np.interp(x, calibration_freq[:, idx], calibration_val[:, idx])

        # convert from log space to scale factor
        calibration = 10**(calibration / 20)
        return calibration

    # get the low-pass/high-pass filter parameters used for processing the raw measurements
    def getFilterParameters(self, fc_lp, fc_hp, n=511):
        # get lowpass and highpass filters
        n = 511
        fc = 1.5e3
        hp = firwin(n, fc_lp, fs=self.fs, pass_zero=False)
        lp = firwin(n, fc_hp, fs=self.fs, pass_zero=True)
        return hp, lp

    # define the transmit chirp signal sent over the speakers
    def getChirpSignal(self):
        chirp = scipy.signal.chirp(self.t[:int(self.samples_per_chirp)], 
                                       self.f0, self.T/self.channels[1], self.f1)
        return chirp

    # demodulate the recording of the scene response to the FMCW transmit signal  
    def demodulateFMCW(self, input_file, output_file, overwrite_raw=False):

        if not os.path.isfile(output_file) or overwrite_raw:
            ckpt = pickle_load(input_file)
            raw_scan = ckpt['raw_scan']
            
            num_horz_samples = raw_scan.shape[0]
            raw_data = np.zeros((self.channels[0], self.channels[1], num_horz_samples, self.samples_per_chirp))
            for i in range(self.channels[0]):
                for j in range(self.channels[1]):
                    for k in range(num_horz_samples):
                        raw_data[i, j, k, :] = np.mean(raw_scan[k, :, j*self.samples_per_chirp:(j+1)*self.samples_per_chirp, i], axis=0)

            confocal_data = np.zeros((self.channels[0], num_horz_samples, self.samples_per_chirp))
            for i in range(self.channels[0]):
                for j in range(num_horz_samples):
                    confocal_data[i, j, :] = raw_data[i, i, j, :]

            processed_data = np.zeros((self.channels[0], self.channels[1], num_horz_samples, self.samples_per_chirp))
            for i in tqdm(range(self.channels[0])):
                for j in range(self.channels[1]):
                    for k in range(num_horz_samples):

                        # high pass filter the input data
                        data = lfilter(self.hp, 1, raw_data[i, j, k, :], axis=0) 

                        # take the fourier transform and apply calibration 
                        data_ft = np.fft.fft(data, axis=0)
                        data_ft_cal = data_ft / self.calibration[:, i]
                        data = np.fft.ifft(data_ft_cal, axis=0)

                        # mix with chirp signal
                        data *= self.chirp        

                        # low pass filter the output
                        data = lfilter(self.lp, 1, data, axis=0)

                        # take the fourier transform and display
                        data_ft = np.fft.fft(data, axis=0)
                        data_ft = np.abs(data_ft)**2

                        processed_data[i, j, k, :] = data_ft 

            # save this intermediary output so we don't have to calculate it every time
            np.save(output_file, processed_data)

    def run_lct(self, meas, X, Y, Z, S, x_max, y_max):
        self.max_dist = int(self.T * self.v/2 / self.channels[0])
        slope_x = self.dx * x_max / (self.fend/self.B * self.max_dist) * (1 + ((self.fstart/self.B)/(self.fend/self.B))**2)
        slope_y = self.dy * y_max / (self.fend/self.B * self.max_dist) * (1 + ((self.fstart/self.B)/(self.fend/self.B))**2)
        
        # params
        psf, fpsf = lct.getPSF(X, Y, Z, S, slope_x, slope_y)
        mtx, mtxi = lct.interpMtx(Z, S, self.fstart/self.B * self.max_dist, self.fend/self.B * self.max_dist)

        def pad_array(x, S, Z, X, Y):
            return np.pad(x, ((S*Z//2, S*Z//2), (Y//2, Y//2), (X//2, X//2)), 'constant')

        def trim_array(x, S, Z, X, Y):
            return x[S*int(np.floor(Z/2))+1:-S*int(np.ceil(Z/2))+1, Y//2+1:-Y//2+1, X//2+1:-X//2+1]

        invpsf = np.conj(fpsf) / (abs(fpsf)**2 + 1 / self.snr)
        tdata = np.matmul(mtx, meas.reshape((Z, -1))).reshape((-1, Y, X))

        fdata = fftn(pad_array(tdata, S, Z, X, Y))
        tvol = abs(trim_array(ifftn(fdata * invpsf), S, Z, X, Y))
        out = np.matmul(mtxi, tvol.reshape((S*Z, -1))).reshape((-1, Y, X))

        return out

    def plot_vol(self, out, title, savefile=None):
        # display
        plt.subplot(1,3,1)
        plt.imshow(np.max(out, axis=1).squeeze(), aspect='auto', cmap='gray')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.xticks([], [])
        plt.yticks([], [])

        plt.subplot(1,3,2)
        plt.imshow(np.max(out, axis=2).squeeze(), aspect='auto', cmap='gray')
        plt.xlabel('y')
        plt.ylabel('t')
        plt.xticks([], [])
        plt.yticks([], [])
        plt.title(title)

        plt.subplot(1,3,3)
        plt.imshow(np.max(out, axis=0).squeeze(), aspect='auto', cmap='gray')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xticks([], [])
        plt.yticks([], [])

        if savefile is not None:
            plt.savefig(savefile, bbox_inches='tight', dpi=300)
        plt.pause(1)

    def confocalReconstruction(self, data_file, background_file=None, save_npy=False):
        processed_data = np.load(data_file)
        if background_file is not None:
            processed_bg = np.load(background_file)
            processed_data -= processed_bg

        num_horz_samples = processed_data.shape[2]
        confocal_data = np.zeros((self.channels[0], num_horz_samples, self.samples_per_chirp))
        for i in range(self.channels[0]):
            for j in range(num_horz_samples):
                confocal_data[i, j, :] = processed_data[i, i, j, :]

        confocal_data = confocal_data[:, :, self.fstart_idx:self.fend_idx]
        meas = confocal_data
        meas = np.transpose(meas, axes=(2, 0, 1))
        meas = np.flip(meas, axis=1)
        out = meas

        #run lct
        if self.use_lct:
            Nx = meas.shape[2] 
            Ny = meas.shape[1] 
            Nt = meas.shape[0] 
            x_max = 1 
            y_max = 1 
            X = Nx 
            Y = Ny
            Z = Nt
            S = 2
            out = self.run_lct(meas, X, Y, Z, S, x_max, y_max)

        title = 'Confocal reconstruction'
        if self.enable_plotting:
            self.plot_vol(out, title, savefile=data_file.replace('.npy', '.png'))
            if save_npy: 
                out_dict = {'out': out, 'meas': meas, 'fstart_idx': self.fstart_idx,
                            'fend_idx': self.fend_idx, 'f': self.f,
                            'f0': self.f0, 'f1': self.f1, 'B': self.B,
                            'fstart': self.fstart, 'fend': self.fend, 'T': self.T, 'v': self.v, 
                            'processed_data': processed_data_to_save}
                np.save(data_file.replace('.npy', '_out.npy'), out_dict)

    def DMO(self, cmp_stack, mp, offset):
        # log stretch over time
        # cmp_stack formatted as [midpoint, offset, horz_sample, t]
        # get midpoint frequencies
        dmp = mp[1] - mp[0]
        ky = np.fft.fftfreq(len(mp), d=dmp)

        # get log-stretch output frequencies 
        dt = self.t[1] - self.t[0]
        w = np.fft.fftfreq(len(self.t), d=dt)
        w[w == 0] = np.nan # ignore zero frequencies
        tau = np.exp(np.linspace(np.log(self.t[0]), np.log(self.t[-1]), len(self.t)))
        dmo = np.zeros(cmp_stack.shape, dtype=complex)
        for i in range(cmp_stack.shape[0]):
            for j in range(cmp_stack.shape[1]):
                for k in range(cmp_stack.shape[2]):
                    dmo[i, j, k, :] = np.interp(tau, self.t, cmp_stack[i, j, k, :])
                      
        for i in range(dmo.shape[1]): # offset
            for j in range(dmo.shape[2]): # horz sample
                # this is the correction from zhou 1996
                # http://www.ahay.org/RSF/book/sep/fkamo/paper.pdf
                dmo[:, i, j, :] = np.flip(dmo[:, i, j, :], axis=1)
                dmo[:, i, j, :] = np.fft.fft2(dmo[:, i, j, :])
                phase = w / 2 * ( np.sqrt(1 + (2 * ky[:, None] * offset[i])**2 / w**2) - 1 - 
                                  np.log((np.sqrt(1 + (2 * ky[:, None] * offset[i])**2 / w**2) + 1) / 2 ) )
                phase[np.isnan(phase)] = 0
                dmo[:, i, j, :] = dmo[:, i, j, :] * np.exp(1j * phase)
                dmo[:, i, j, :] = np.fft.ifft2(dmo[:, i, j, :])
                dmo[:, i, j, :] = np.flip(dmo[:, i, j, :], axis=1)

                for k in range(cmp_stack.shape[0]):
                    dmo[k, i, j, :] = np.interp(self.t, tau, dmo[k, i, j, :]) 

        return np.real(dmo)

    def NMO(self, offset_y, offset_x, meas):
        tq = np.sqrt(self.t**2 + (offset_x**2 / self.v**2 + offset_y**2 / self.v**2))
        tq[np.isnan(tq)] = 0
        meas_2 = np.interp(tq, self.t, meas)
        return meas_2

    def NMOReconstruction(self, data_file, background_file=None, save_npy=True):
        processed_data = np.load(data_file)[:, :, :, self.fstart_idx:self.fend_idx]
        if background_file is not None:
            processed_bg = np.load(background_file)[:, :, :, self.fstart_idx+self.bg_offset:self.fend_idx+self.bg_offset]
            processed_data = np.maximum(0, processed_data - 1*processed_bg)
        processed_data_to_save = processed_data.copy()

        # scale processed data for radial falloff
        z = self.t * self.v
        z2 = z**2 / z[0]**2
        processed_data *= z2[None, None, None, :]

        # rearrange by midpoint and offset location
        mic_pos_y = np.linspace(0, 1, self.channels[0])
        mic_pos_x = np.linspace(0.08, 0.08, self.channels[0])
        speak_pos_y = np.linspace(0, 1, self.channels[1])
        speak_pos_x = np.linspace(0.0, 0.0, self.channels[1])

        mp_x = np.zeros(self.channels)
        mp_y = np.zeros(self.channels)
        for i in range(self.channels[0]): # mic
            for j in range(self.channels[1]): # speaker
                mp_x[i, j] = (mic_pos_x[i] + speak_pos_x[j]) / 2.
                mp_y[i, j] = (mic_pos_y[i] + speak_pos_y[j]) / 2.

        offset_x = np.zeros(self.channels)
        offset_y = np.zeros(self.channels)
        for i in range(self.channels[0]): # mic
            for j in range(self.channels[1]): # speaker
                offset_x[i, j] = mic_pos_x[i] - speak_pos_x[j]
                offset_y[i, j] = mic_pos_y[i] - speak_pos_y[j]

        unique_mp_x = np.unique(mp_x.ravel().round(decimals=4))
        unique_mp_y = np.unique(mp_y.ravel().round(decimals=4))
        unique_offset_x = np.unique(offset_x.ravel().round(decimals=4))
        unique_offset_y = np.unique(offset_y.ravel().round(decimals=4))

        num_horz_samples = processed_data.shape[2]
        processed_cmp_nmo = np.zeros((len(unique_mp_y), len(unique_offset_y), num_horz_samples, processed_data.shape[-1]))
        processed_cmp = np.zeros((len(unique_mp_y), len(unique_offset_y), num_horz_samples, processed_data.shape[-1]))
        num_offset = np.zeros(len(unique_mp_y))
        for i in range(self.channels[0]): # mic
            for j in range(self.channels[1]): # speaker
                for k in range(num_horz_samples):
                    mp_idx = np.argmin(abs(unique_mp_y - mp_y[i, j]))
                    offset_idx = np.argmin(abs(unique_offset_y - offset_y[i, j]))
                    processed_cmp_nmo[mp_idx, offset_idx, k, :] = self.NMO(offset_y[i, j], offset_x[i, j], processed_data[i, j, k, :])
                    processed_cmp[mp_idx, offset_idx, k, :] = processed_data[i, j, k, :]
                    num_offset[mp_idx] += 1

        ## do DMO correction
        processed_cmp_nmo = self.DMO(processed_cmp, unique_mp_y, unique_offset_y / 2)
        num_offset /= num_horz_samples 
        processed_cmp = np.sum(processed_cmp, axis=1) / num_offset[:, None, None]

        processed_cmp_nmo = np.sum(processed_cmp_nmo, axis=1) / (num_offset[:, None, None])
        processed_cmp_nmo -= np.min(processed_cmp_nmo)

        processed_cmp_nmo = processed_cmp_nmo[:-1, :, :]
        meas = processed_cmp_nmo 
        meas = np.transpose(meas, axes=(2, 0, 1))
        meas = np.flip(meas, axis=1)
        out = meas

        # run lct
        if self.use_lct:
            Nx = meas.shape[2] 
            Ny = meas.shape[1] 
            Nt = meas.shape[0] 
            x_max = 1 
            y_max = 1 
            X = Nx 
            Y = Ny
            Z = Nt
            S = 2
            out = self.run_lct(meas, X, Y, Z, S, x_max, y_max)

        title = 'Confocal + non-confocal reconstruction'
        if self.enable_plotting:
            self.plot_vol(out, title, savefile=data_file.replace('.npy', '.png'))
            if save_npy: 
                out_dict = {'out': out, 'meas': meas, 'fstart_idx': self.fstart_idx,
                            'fend_idx': self.fend_idx, 'f': self.f,
                            'f0': self.f0, 'f1': self.f1, 'B': self.B,
                            'fstart': self.fstart, 'fend': self.fend, 'T': self.T, 'v': self.v, 
                            'processed_data': processed_data_to_save}
                np.save(data_file.replace('.npy', '_out.npy'), out_dict)

    def run(self, scene):
        ignore_bg = False
        use_confocal = False
        overwrite_raw = False 
        self.use_lct = False

        if scene == 'letter_H':
            self.fstart = 3500
            self.fend = 4500

        elif scene == 'corner_reflectors':
            self.fstart = 4200
            self.fend = 6000
            self.use_lct = True
        
        elif scene == 'resolution_corner1m':
            self.fstart = 1000
            self.fend = 3800
            self.dx =  0.85
            self.dy =  .9 
            self.use_lct = True

        elif scene == 'resolution_corner2m':
            self.fstart = 1000
            self.fend = 3800
            self.use_lct = True

        elif scene == 'double':
            self.fstart = 5800
            self.fend = 6800
            self.use_lct = True

        elif scene == 'psf':
            # note this is los measurement at ~ 1m with 5x5x5 cm corner reflector
            self.fstart = 1200
            self.fend = 2200
            self.dx = 0.90
            self.dy = 0.85
            self.use_lct = True
            self.snr = 1e5

        elif scene == 'resolution_plane1m':
            self.fstart = 1000
            self.fend = 3800

        elif scene == 'resolution_plane2m':
            self.fstart = 1000
            self.fend = 3800

        elif scene == 'letters_LT':
            self.fstart = 4200
            self.fend = 5500
     
        # additional params
        fname = './data/' + scene
        self.fstart_idx = np.argmin(abs(self.f - self.fstart));
        self.fend_idx = np.argmin(abs(self.f - self.fend));
        self.max_dist = int(self.T * self.v/2 / self.channels[0])
        self.t = np.linspace(self.fstart/self.B * self.max_dist * 2 / self.v, 
                             self.fend/self.B * self.max_dist * 2 / self.v, 
                             self.fend_idx-self.fstart_idx) 
       
        print('=> Reconstructing scene {}'.format(scene))
        print('=> Volume distance {:.02f} m to {:.02f} m'.format(self.fstart/self.B * self.max_dist, self.fend/self.B * self.max_dist))

        # processing on raw data
        # - range compression
        # - microphone calibration
        # - output 4D volume of [n_mic, n_speaker, n_horz_samples, n_time_samples] 
        print('=> Raw data processing')
        self.demodulateFMCW(fname + '.pkl', fname + '.npy', overwrite_raw=overwrite_raw) 
        if os.path.isfile(fname + '_bg.pkl'):
            self.demodulateFMCW(fname + '_bg.pkl', fname + '_bg.npy', overwrite_raw=overwrite_raw) 
        # optionally process confocal volume
        start = time.time()
        if use_confocal:
            self.confocalReconstruction(fname + '.npy')
            return

        # process confocal + nonconfocal measurements with NMO & DMO correction
        print('=> NMO + DMO Processing and post-stack migration')
        if not os.path.isfile(fname + '_bg.npy') or ignore_bg:
            self.NMOReconstruction(fname + '.npy')
        else:
            print('=> Using background subtraction')
            self.NMOReconstruction(fname + '.npy', fname + '_bg.npy')
        stop = time.time()
        print('=> Elapsed Time: {:.02f}'.format(stop - start))
        print()
        return

    def usage(self):
        print('Usage: python3 AcousticNLOSReconstruction.py [scene1] [scene2] ...')
        print('       valid scenes are')
        print('         \'double\'')
        print('         \'letter_H\'')
        print('         \'corner_reflectors\'')
        print('         \'psf\'')
        print('         \'resolution_corner1m\'')
        print('         \'resolution_corner2m\'')
        print('         \'resolution_plane1m\'')
        print('         \'resolution_plane2m\'')
        print('         \'letters_LT\'')
        return

if __name__ == '__main__':
    reconstruction = AcousticNLOSReconstruction()
    valid_scenes = ['double', 'letter_H', 'corner_reflectors',
                  'psf', 'resolution_corner1m', 'resolution_corner2m',
                  'resolution_plane1m', 'resolution_plane2m', 'letters_LT']

    scene = sys.argv[1:] 
    if len(scene) == 0:
        reconstruction.usage()

    if scene == ['all']:
        scenes = valid_scenes

    for s in scene:
        if s not in valid_scenes:
            reconstruction.usage()
            break
        reconstruction.run(s)
