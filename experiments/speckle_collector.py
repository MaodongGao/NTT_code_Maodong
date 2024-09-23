
from hardware.camera import NITCam
from hardware.toptica_laser import TopticaLaser
from hardware.device import Device
from typing import Optional

import numpy as np
import time

C_CONST = 299792458

class SpeckleCollector:
    def __init__(self, camera: Optional[NITCam]
                     , laser: Optional[TopticaLaser]):
        # Create dummy device for logging
        self.logger = Device(addr='', name='SpeckleCollector', isVISA=False)
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical
        self.devicename = self.logger.devicename

        self.camera = camera
        self.laser = laser
        self.wl_tol_MHz = 10 # tolerance of laser wavelength uncertainty in MHz



    @property
    def wl_tol_nm(self):
        fc_THz = self.__nm2thz(1550)
        fc2_THz = fc_THz - self.wl_tol_MHz / 1e6
        return np.abs(self.__thz2nm(fc_THz) - self.__thz2nm(fc2_THz))

    def collect_wl_pattern(self, wl_nm: float, save_log: bool = False):
        self.laser.laser1.ctl.wavelength_set.set(wl_nm)

        # Wait for the laser to stabilize
        tic = time.time()
        while not np.isclose(self.laser.laser1.ctl.wavelength_act.get(), wl_nm, atol=self.wl_tol_nm):
            if time.time() - tic > 10:
                self.laser.error(f"Timeout waiting for laser to stabilize at {wl_nm} nm")
                raise TimeoutError("Timeout waiting for laser to stabilize")
        self.laser.info(f"Set wavelength to {self.laser.laser1.ctl.wavelength_act.get()} nm")

        # Capture the image
        img = self.camera.capture_single_frame(save_log=save_log)
        self.info(f"Captured image at {wl_nm} nm")

        return img

    def scan_wl_pattern(self, wl_start_nm: float, wl_stop_nm: float, step_MHz: float = 10, filename=None, extensions=None):
        wl_start_THz = self.__nm2thz(wl_start_nm)
        wl_stop_THz = self.__nm2thz(wl_stop_nm)
        step_THz = step_MHz / 1e6

        if wl_start_THz > wl_stop_THz:
            step_THz = -step_THz

        # Prepare the array of wavelengths
        wl_THz = np.arange(wl_start_THz, wl_stop_THz, step_THz)

        if filename is not None:
            if extensions is None:
                extensions = ['.csv','.json']
            self._prepare_file(filename, extensions)

        if '.csv' in extensions:
            import csv
            with open(filename+'.csv', mode='w', newline='') as file:
                writer = csv.writer(file)
                # Headers record camera settings
                writer.writerow(['bitDepth', self.camera.bitDepth])
                writer.writerow(['exposureTime', self.camera.exposure_time])
                writer.writerow(['AnalogGain', self.camera.analog_gain])
                writer.writerow(['offset_x', self.camera.offset_x])
                writer.writerow(['offset_y', self.camera.offset_y])
                writer.writerow(['width', self.camera.frame_width])
                writer.writerow(['height', self.camera.frame_height])
                writer.writerow([])
                writer.writerow(['Wavelength(nm)', 'Freq(THz)', 'Image'])

        # Prepare the array of images
        images = {}
        for freq_thz in wl_THz:
            wl_nm = self.__thz2nm(freq_thz)
            images[wl_nm] = self.collect_wl_pattern(wl_nm)

            if '.csv' in extensions:
                with open(filename+'.csv', mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([wl_nm, freq_thz] + images[wl_nm].flatten().tolist())

        if '.json' in extensions:
            import json
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return json.JSONEncoder.default(self, obj)
            # save images to json
            with open(filename+'.json', 'w') as file:
                json.dump(images, file, cls=NumpyEncoder)

        return images


    
    def __nm2thz(self, nm):
        return C_CONST / nm / 1000

    def __thz2nm(self, thz):
        return C_CONST / thz / 1000
    
    def _prepare_file(self, filename, extensions=None):
        import os
        """Prepare the file by creating directories and handling existing files."""
        filedir, single_filename = os.path.split(filename)
        if not os.path.isdir(filedir) and filedir != '':
            self.warning(self.devicename + ": Directory " + filedir + " does not exist. Creating new directory.")
            os.makedirs(filedir)

        if extensions is None:
            # get extension from single_filename
            filename, ext = os.path.splitext(filename)
            extensions = [ext]

        for ext in extensions:
            if os.path.isfile(filename + ext):
                self.warning(self.devicename + ": Filename " + filename + ext + " already exists. Previous file renamed.")
                now = time.strftime("%Y%m%d_%H%M%S")  # Current date and time
                new_name = f"{filename}_{now}_bak{ext}"
                os.rename(filename + ext, new_name)

        return filename