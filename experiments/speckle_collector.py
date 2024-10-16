
from hardware.camera import NITCam
from hardware.toptica_laser import TopticaLaser
from hardware.device import Device
from typing import Optional

import numpy as np
import time
import os
import pandas as pd

from hardware.utils.filename_handler import DirnameHandler, FilenameHandler

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
        self.wl_tol_MHz = 125 # tolerance of laser wavelength uncertainty in MHz



    @property
    def wl_tol_nm(self):
        fc_THz = self.__nm2thz(1550)
        fc2_THz = fc_THz - self.wl_tol_MHz / 1e6
        return np.abs(self.__thz2nm(fc_THz) - self.__thz2nm(fc2_THz))

    def collect_wl_pattern(self, wl_nm: float, save_log: bool = False):
        self.laser.laser1.ctl.wavelength_set.set(wl_nm)
        # time.sleep(1)

        # Wait for the laser to stabilize
        tic = time.time()
        # while not np.isclose(self.laser.laser1.ctl.wavelength_act.get(), wl_nm, atol=self.wl_tol_nm):
        wl_now = self.laser.get_time_avged_value(get_func=self.laser.laser1.ctl.wavelength_act.get, 
                                                             avg_time=1, interval=0)
        while not np.isclose(wl_now,
                             wl_nm, atol=self.wl_tol_nm, rtol=0.0):
            if time.time() - tic > 20:
                self.laser.error(f"Timeout waiting for laser to stabilize at {wl_nm} nm")
                raise TimeoutError("Timeout waiting for laser to stabilize")
            
            self.info(f"Comparing last second avg wavelength {wl_now} nm to target {wl_nm} nm, diff = {np.abs(wl_now - wl_nm)} nm out of tolerance {self.wl_tol_nm} nm ({self.wl_tol_MHz} MHz)")
            self.laser.laser1.ctl.wavelength_set.set(wl_nm)
            time.sleep(0.5)
            wl_now = self.laser.get_time_avged_value(get_func=self.laser.laser1.ctl.wavelength_act.get,
                                                        avg_time=1, interval=0)
        
        self.info(f"Last second avg wavelength \t{wl_now} nm ({self.__nm2thz(wl_now)*1e6} MHz)\n is close to target \t\t{wl_nm} nm ({self.__nm2thz(wl_nm)*1e6} MHz).\n Difference \t\t\t{np.abs(wl_now - wl_nm)} nm ({np.abs(self.__nm2thz(wl_now) - self.__nm2thz(wl_nm))*1e6} MHz)\n is within tolerance \t\t{self.wl_tol_nm} nm ({self.wl_tol_MHz} MHz)")

        ww = self.laser.laser1.ctl.wavelength_act.get()
        self.laser.info(f"Last acquired laser wavelength is {ww} nm, {self.__nm2thz(ww)*1e6} MHz")

        # Capture the image
        self.camera.clearFrames()
        img = self.camera.capture_single_frame(save_log=save_log)
        self.info(f"Captured image at {wl_nm} nm, {self.__nm2thz(wl_nm)*1e6} MHz")

        return img

    def scan_wl_pattern(self, wl_start_nm: float, wl_stop_nm: float, step_MHz: float = 10, filename=None):
        wl_start_THz = self.__nm2thz(wl_start_nm)
        wl_stop_THz = self.__nm2thz(wl_stop_nm)
        step_THz = step_MHz / 1e6

        if wl_start_THz > wl_stop_THz:
            step_THz = -step_THz

        # Prepare the array of wavelengths
        wl_THz = np.arange(wl_start_THz, wl_stop_THz, step_THz)

        # Prepare the array of images
        images = {}
        temp_tosave = {}

        dirname_handler = DirnameHandler(filename).prepare()
        self.info(dirname_handler)

        for ii, freq_thz in enumerate(wl_THz):
            wl_nm = self.__thz2nm(freq_thz)
            time.sleep(0.5)
            images[wl_nm] = self.collect_wl_pattern(wl_nm).flatten().tolist()

            temp_tosave[wl_nm] = images[wl_nm]

            # with open(filename+'.csv', mode='a', newline='') as file:
            #     writer = csv.writer(file)
            #     writer.writerow([wl_nm] + images[wl_nm].flatten().tolist())

            # Save the images to a file every 10 images
            if filename is not None and ((ii+1) % 10 == 0):
                df = pd.DataFrame(temp_tosave)
                df.to_csv(os.path.join(filename, f'{filename}_{(ii+1)//10}.csv'), index=False)
                temp_tosave = {}
        
        # handle the remaining images
        if filename is not None and len(temp_tosave) > 0:
            df = pd.DataFrame(temp_tosave)
            df.to_csv(os.path.join(filename, f'{filename}_{((ii+1)//10) +1}.csv'), index=False)

        if filename is not None:
            extensions = ['.json', '.csv']
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


        #     df = pd.DataFrame(images)
        #     df.to_csv(filename+'.csv', index=False)

        # return images


    
    def __nm2thz(self, nm):
        return C_CONST / nm / 1000

    def __thz2nm(self, thz):
        return C_CONST / thz / 1000
    
    def _prepare_file(self, filename, extensions=None):
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

    def _prepare_directory(self, directory):
        """Prepare the directory by creating directories and handling existing files."""
        if not os.path.isdir(directory):
            self.warning(self.devicename + ": Directory " + directory + " does not exist. Creating new directory.")
            os.makedirs(directory)
        else:
            self.warning(self.devicename + ": Directory " + directory + " already exists. Previous directory renamed.")
            now = time.strftime("%Y%m%d_%H%M%S")
            new_name = f"{directory}_{now}_bak"
            os.rename(directory, new_name)

        return directory