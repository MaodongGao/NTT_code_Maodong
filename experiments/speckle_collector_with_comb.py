
from hardware.camera import NITCam
from hardware.toptica_laser import TopticaLaser
from hardware.waveshaper import WaveShaper
from hardware.device import Device
from typing import Optional

import numpy as np
import time
import os
import pandas as pd

from hardware.utils.filename_handler import DirnameHandler, FilenameHandler

C_CONST = 299792458

class SpeckleCollectorWithComb:
    def __init__(self, camera: Optional[NITCam],
                       ws: Optional[WaveShaper]):
        # Create dummy device for logging
        self.logger = Device(addr='', name='SpeckleCollectorWithComb', isVISA=False)
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical
        self.devicename = self.logger.devicename

        self.camera = camera
        self.ws = ws

        self.bpf_bw_thz = 0.02 # Bandwidth of the bandpass filter in THz
        self.comb_fsr_ghz = 25 #25 # 50 # Comb FSR in GHz
        self.comb_reference_wl_nm = 1557.547 # 1557.605 # Comb reference wavelength in nm
        self.comb_min_wl_nm = 1527 # 1548.122 # Comb minimum wavelength in nm
        self.comb_max_wl_nm = 1563 # 1575.000 # Comb maximum wavelength in nm

        # TODO: dynamic atten not implemented yet
        self.comb_spectrum = [0 for _ in self.comb_center_freq_thz] # Power(dBm) of every comb line
        self.max_input_power_dB = 0 # Maximum input power in dBm

    @property
    def comb_reference_freq_thz(self):
        return self.ws.nm2thz(self.comb_reference_wl_nm)
    @property
    def comb_min_freq_thz(self):
        return self.ws.nm2thz(self.comb_max_wl_nm)
    @property
    def comb_max_freq_thz(self):
        return self.ws.nm2thz(self.comb_min_wl_nm)

    @property
    def comb_center_freq_thz(self):
        thz_center_list_1 = np.arange(self.comb_reference_freq_thz + self.comb_fsr_ghz/1e3, 
                                      self.comb_max_freq_thz, 
                                      self.comb_fsr_ghz/1e3)
        thz_center_list_2 = np.arange(self.comb_reference_freq_thz,
                                      self.comb_min_freq_thz,
                                      -self.comb_fsr_ghz/1e3)
        thz_center_list = np.concatenate([thz_center_list_2, thz_center_list_1]).flatten()
        return np.sort(thz_center_list)
    
    def determin_within_passband(self, thz, centers=None, bw=None):
        if centers is None:
            centers = self.comb_center_freq_thz
        if bw is None:
            bw = self.bpf_bw_thz
        for center in centers:
            if np.abs(thz - center) < bw/2:
                return True
        return False
    
    def filter_comb_with_new_fsr(self, centers=None, bw=None):
        if centers is None:
            centers = self.comb_center_freq_thz
        if bw is None:
            bw = self.bpf_bw_thz
        wsAtten = np.array([0 if self.determin_within_passband(f, centers=centers, bw=bw)
                            else self.ws.MAX_ATTEN for f in self.ws.wsFreq])
        wsPhase = np.zeros_like(self.ws.wsFreq)
        wsPort = np.array([1 for _ in self.ws.wsFreq])

        self.ws.upload_profile(wsAtten, wsPhase, wsPort, plot=False)

    def pass_one_comb_line(self, num: int, passband_atten: float = 0):
        self.ws.set_bandpass(center=self.comb_center_freq_thz[num], span=self.bpf_bw_thz, unit='THz', passband_atten=passband_atten)

    def collect_thz_pattern(self, freq_thz: float, save_log: bool = False, passband_atten: float = 0):
        self.ws.set_bandpass(center=freq_thz, span=self.bpf_bw_thz, unit='THz', passband_atten=passband_atten)
        import time
        time.sleep(0.5)

        # Capture the image
        self.camera.clearFrames()
        img = self.camera.capture_single_frame(save_log=save_log)
        self.info(f"Captured image centered at {freq_thz} THz, with bandwidth {self.bpf_bw_thz} THz")

        return img
    
    def scan_pattern(self, 
                     thz_list= None, # list of THz frequencies to scan
                     filename=None,
                     callback=None, # callback function to be called after each image is captured
                     get_bandpass_atten_func=None # function to get the bandpass attenuation
                    ):
        if thz_list is None:
            thz_list = self.comb_center_freq_thz
            
        if filename is not None:
            # First need to prepare the file directory which is one level up
            dirname_handler_2 = DirnameHandler(os.path.dirname(filename)).prepare()
            self.info(dirname_handler_2)

            dirname_handler = DirnameHandler(filename).prepare()
            self.info(dirname_handler)


            extensions = ['.csv', '.json']
            for ext in extensions: self.info(FilenameHandler(filename).prepare(ext))

            import json
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return json.JSONEncoder.default(self, obj)
            # save images to json
            with open(filename+'.json', 'w') as file:
                json.dump(self.camera.config, file, cls=NumpyEncoder)

        # Prepare the array of images
        images = {}
        temp_tosave = {}
        callback_tosave = {}

        failed_freqs = []
        failed_callback_freqs = []

        from tqdm import tqdm
        for ii, freq_thz in enumerate(tqdm(thz_list)):
            time.sleep(0.5)
            try:
                # atten = np.max([0, -self.max_input_power_dB + self.comb_spectrum[ii]])
                if get_bandpass_atten_func is not None:
                    atten = get_bandpass_atten_func(freq_thz)
                else:
                    atten = 0
                images[freq_thz] = self.collect_thz_pattern(freq_thz, passband_atten=atten).flatten().tolist()
            except TimeoutError:
                self.warning(f"Failed to collect image centered at {freq_thz} THz")
                failed_freqs.append(freq_thz)

            temp_tosave[freq_thz] = images[freq_thz]

            if callback is not None:
                try:
                    callback_tosave[freq_thz] = callback()
                except Exception as e:
                    self.warning(f"Failed to execute callback at {freq_thz} THz")
                    failed_callback_freqs.append(freq_thz)

            # Save the images to file every 10 images
            if filename is not None and ((ii+1) % 10 == 0):
                df = pd.DataFrame(temp_tosave)
                df.to_csv(os.path.join(filename, f'{filename}_{(ii+1)//10}.csv'), index=False)
                temp_tosave = {}
                if callback is not None:
                    df_callback = pd.DataFrame(callback_tosave)
                    df_callback.to_csv(os.path.join(filename, f'{filename}_{(ii+1)//10}_callback.csv'), index=False)
                    callback_tosave = {}

        # handle remaining images
        if filename is not None and len(temp_tosave) > 0:
            df = pd.DataFrame(temp_tosave)
            df.to_csv(os.path.join(filename, f'{filename}_{(ii+1)//10+1}.csv'), index=False)
            if callback is not None:
                df_callback = pd.DataFrame(callback_tosave)
                df_callback.to_csv(os.path.join(filename, f'{filename}_{(ii+1)//10+1}_callback.csv'), index=False)

        # save failed frequencies
        if len(failed_freqs) > 0 and filename is not None:
            failed_freqs = np.array(failed_freqs)
            np.savetxt(os.path.join(filename, f'{filename}_failed_freqs.csv'), failed_freqs, delimiter=',', fmt='%f')
        if len(failed_callback_freqs) > 0 and filename is not None and callback is not None:
            failed_callback_freqs = np.array(failed_callback_freqs)
            np.savetxt(os.path.join(filename, f'{filename}_failed_callback_freqs.csv'), failed_callback_freqs, delimiter=',', fmt='%f')