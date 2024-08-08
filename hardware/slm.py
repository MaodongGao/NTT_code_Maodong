import numpy as np
import ctypes
import time
import matplotlib.pyplot as plt
from .santec import _slm_win as slm # Import the vendor-specific SLM functions
import os
from scipy.interpolate import interp1d
import warnings
from .device import Device

from .utils.grid_config import GridConfig

SLM_OK = 0 # SLM return code for success

class SLM(Device):
    def __init__(self, SLM_number, name = None, isVISA=False):
        """
        Initialize the SLM device.
        """
        if name is None:
            name = f"SLM{SLM_number}"

        super().__init__(addr=SLM_number, name=name, isVISA=isVISA)
        # self.config = config
        self.SLM_number = self.addr
        self.csv_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'slm_csv_files')
        self.csv_filename = "csvfile" + str(self.SLM_number) + ".csv"
        self.slm_height = 1200
        self.slm_width = 1920
        self.lookuptable_csv_path = os.path.join(self.csv_directory, "r_vs_phase_slm" + str(self.SLM_number) + ".csv")
        
        # self.connect()
        # self.set_wavelength_nm()

        self._grid_config = GridConfig() # dictionary like object for grid configuration

    def update_grid_config(self, params):
        self._grid_config.update(params)
        self.info(self.devicename + ': Grid configuration updated to:\n' + str(self.grid_config))

    @property
    def grid_config(self):
        return self._grid_config

    def set_wavelength_nm(self, wavelength=1500):
        """
        Set the wavelength for the SLM.
        """
        # Set wavelength
        ret = slm.SLM_Ctrl_WriteWL(self.SLM_number, wavelength, 200)
        # Save wavelength
        ret = slm.SLM_Ctrl_WriteAW(self.SLM_number)
        if ret == SLM_OK:
            self.info(self.devicename + f": Wavelength set to {wavelength} nm.")
        else:
            self.error(self.devicename + f": Error setting wavelength to {wavelength} nm.")
        return ret
    
    def connect(self):
        slm.SLM_Ctrl_Close(self.SLM_number) # TODO: is this line necessary?
        ret = slm.SLM_Ctrl_Open(self.SLM_number)
        if ret == SLM_OK:
            self.connected = True
            self.info(self.devicename +': SLM' + str(self.SLM_number) + ' successfully initialized.')
            return 1
        else:
            self.error(self.devicename +': Error connecting to SLM' + str(self.SLM_number))
            return -1

    def disconnect(self):
        slm.SLM_Ctrl_Close(self.SLM_number)
        self.connected = False
        self.info(self.devicename +': SLM' + str(self.SLM_number) + ' disconnected.')

    @property
    def lookup_table(self):
        '''
        Returns the lookup table for the SLM
        Assumes the lookup table is in the format of a CSV file with two columns: phase and reflection
        '''
        lookup_table = np.genfromtxt(self.lookuptable_csv_path, delimiter=',', skip_header=1)
        return lookup_table
    
    @property
    def LUT_phase(self):
        '''
        Returns the phase values from the lookup table
        Assumes the first column of the lookup table is the phase values
        '''
        return self.lookup_table[:, 0]
    
    @property
    def LUT_reflection(self):
        '''
        Returns the reflection values from the lookup table
        Assumes the second column of the lookup table is the reflection values
        '''
        return self.lookup_table[:, 1]
    
    @property
    def phase_interp_func(self):
        '''
        Returns an interpolation function for the lookup table
        Return:
            interp1d: An interpolation function that maps reflection values to phase values
        '''
        return interp1d(self.LUT_reflection, self.LUT_phase, kind='linear', bounds_error=False, fill_value="extrapolate")

    @property
    def zero_reflection_phase(self):
        '''
        Returns the phase value corresponding to zero reflection
        '''
        return self.LUT_phase[0]

##########################################################
#############      Basic functions
##########################################################
    
        
    def uniform_phase(self, phase):
        if phase > 1023 or phase < 0:
            raise Exception("Phase is out of expected range (0,1023).")
        
        # Check if the phase input is not an integer and issue a warning
        if not isinstance(phase, int):
            warnings.warn(f"Input phase {phase} is not an integer and will be converted. Original type: {type(phase)}")
            phase = int(phase)  # Convert to integer
        
        slm.SLM_Ctrl_Close(self.SLM_number)
        ret = slm.SLM_Ctrl_Open(self.SLM_number)
        ret = slm.SLM_Ctrl_WriteGS(self.SLM_number, phase)
        slm.SLM_Ctrl_Close(self.SLM_number)
        return ret
    
    def upload_csv_memory_mode(self, CSVfilepath):
        """
        Upload a CSV file to the SLM in memory mode.

        :param CSVfilename: The path to the CSV file to be uploaded.
        """
        slm.SLM_Ctrl_Close(self.SLM_number)
        ret = slm.SLM_Ctrl_Open(self.SLM_number)
        MemoryNumber = 1
        ret = SLM_OK # we initialize the SLM in __init__
        if ret == SLM_OK:
            for i in range(60):# TODO: is this line necessary?
                ret = slm.SLM_Ctrl_ReadSU(self.SLM_number)
                if ret == SLM_OK:
                    break
                else:
                    time.sleep(1)

            slm.SLM_Ctrl_WriteVI(self.SLM_number, 0)  # MemoryMode mode = 0
            
            # TODO: is this line necessary?
            slm.SLM_Ctrl_WriteGS(self.SLM_number, 480)
            
            # send data
            str1 = ctypes.c_char_p(CSVfilepath.encode('utf-8'))
            ret = slm.SLM_Ctrl_WriteMI_CSV_A(self.SLM_number, MemoryNumber, 0, str1)
            slm.SLM_Ctrl_WriteDS(self.SLM_number, MemoryNumber) #Display this MemoryNumber
        slm.SLM_Ctrl_Close(self.SLM_number)
        if ret == 0:
            result = True
        else:
            result = False
        return result

##########################################################
#############      Additional functions
##########################################################
    def update_zero_reflection_phase(self,new_phase):
        self.zero_reflection_phase = new_phase
        self.info(self.devicename + ': Zero-reflection phase updated to ' + str(new_phase))
    
    def generate_csv_from_phase_mask(self, phase_mask):
        """
        Generate a CSV file from a phase matrix for the SLM.

        :param filename: Name of the file to be created.
        :param fullpanel_array: Phase matrix to be converted into a CSV file.
        """
        # TODO: check if array size correct
        # TODO: make the CSV directory path relative to main
        #base_path = os.path.dirname(os.path.abspath(__file__))
        #csv_path = os.path.join(base_path, 'config', 'config.json')
        filepath = f"{self.csv_directory}\\{self.csv_filename}"
        # self.info(self.devicename+filepath)
        with open(filepath, 'w', encoding='cp1252') as file:
            file.write("Y/X," + ','.join(map(str, range(1920))) + '\n')
            for idx, row in enumerate(phase_mask):
                line = str(idx) + ',' + ','.join(map(str, row.astype(int))) + '\n'
                file.write(line)
        #self.info(self.devicename+": CSV file saved in " + filepath)
        return filepath


    def upload_phase_mask(self, phase_mask):
        """
        Load a full-panel phase array onto the SLM.

        :param phase_mask: The phase array to be loaded onto the SLM.
        """
        csvfilepath = self.generate_csv_from_phase_mask(phase_mask)
        self.upload_csv_memory_mode(csvfilepath)


    def embed_matrix(self, matrix, params=None, plot=False):
        """
    Generates a display of Nx by Ny pixels to show a matrix of size (sx,sy) on this display.
    Each matrix element is represented by a rectangular region of nx by ny pixels, separated by guard pixels.
    Pixels outside these regions are set to the value of "background".
    
    :param matrix: Input matrix to be displayed.
    :param params: Dictionary containing parameters for display generation.
    :param plot: Boolean indicating whether to plot the display matrix as a heatmap.
    :return: The display matrix as a numpy array.
        """
        self.info(self.devicename + ": Calculating Embedding matrix on SLM" + str(self.SLM_number) + ".")
        if isinstance(params, dict):
            self.update_grid_config(params)
        elif params is None:
            self.info(self.devicename + ": Current grid configuration is:\n" + str(self.grid_config))
            pass
        else:
            raise TypeError("Parameters must be a dictionary or None.")
        
        params = self.grid_config
        if params.has_missing_keys():
            raise Exception(f"Missing grid configuration parameters {self.grid_config.missing_keys()}. Please provide all necessary parameters.")

        # Extract parameters
        slm_width = self.slm_width
        slm_height = self.slm_height
        elem_width = params["elem_width"]
        elem_height = params["elem_height"]
        gap_x = params["gap_x"]
        gap_y = params["gap_y"]
        topleft_x = params["topleft_x"]
        topleft_y = params["topleft_y"]
        background = self.zero_reflection_phase

    # Initialize the display matrix with the background value
        display = np.full((slm_height, slm_width), fill_value=background, dtype=np.float64)
        sx, sy = matrix.shape
    # Loop over each element in the input matrix to populate the display matrix
        for i in range(sx):
            for j in range(sy):
            # Calculate the top left corner of the current block
                block_top_left_y = topleft_y + i * (elem_height + gap_y)
                block_top_left_x = topleft_x + j * (elem_width + gap_x)
            
            # Ensure the block does not exceed display dimensions
                if block_top_left_y + elem_height <= slm_height and block_top_left_x + elem_width <= slm_width:
                # Set the pixels for the current block to the matrix entry value
                    display[block_top_left_y:block_top_left_y+elem_height, block_top_left_x:block_top_left_x+elem_width] = matrix[i, j]


    # Plot the display matrix as a heatmap if requested
        if plot:
            plt.figure(figsize=(8, 8))
            plt.imshow(display, cmap='hot', interpolation='nearest')
            plt.colorbar()
            plt.title("SLM" + str(self.SLM_number) + "Matrix Heatmap")
            plt.show()
        
        self.upload_phase_mask(display)
        return display
    

    def get_phase_for_attenuation(self, target_reflection):
        """
        Get the phase corresponding to a target attenuation level by interpolating a look-up table.
        
        :param target_attenuation: The target attenuation level or a numpy array of attenuation levels.
        :return: The interpolated phase (integer from 0 to 1023) corresponding to the target attenuation.
        """
        min_phase, max_phase = 0, 1023
        
        # Interpolate phase for the given target attenuation
        interpolated_phase = self.phase_interp_func(target_reflection)
        
        # Check for extrapolation by comparing against the bounds of the lookup table's attenuation values
        if np.any(target_reflection < np.min(self.LUT_reflection)) or np.any(target_reflection > np.max(self.LUT_reflection)):
            warnings.warn("Some target attenuation values are outside the lookup table's range and will be extrapolated.")

        # Check for any interpolated values outside the valid phase range and issue warnings
        if np.any(interpolated_phase < min_phase) or np.any(interpolated_phase > max_phase):
            warnings.warn("Some interpolated phase values are outside the valid range (0 to 1023) and will be clipped.")

        # Ensure the interpolated phase is within bounds and round to nearest integer
        target_phase = np.round(np.clip(interpolated_phase, min_phase, max_phase)).astype(int)

        return target_phase
    

    def plot_LUT(self):
        plt.figure(figsize=(8, 5))
        reflections = np.linspace(self.LUT_reflection[0], self.LUT_reflection[-1],100)
        phases_interpolated = self.phase_interp_func(reflections)
        plt.plot(phases_interpolated, reflections, linestyle=':')
        plt.plot(self.LUT_phase, self.LUT_reflection, marker='o', color = 'red')
        plt.xlabel('Phase (index)')
        plt.ylabel('Reflection')
        plt.grid(True)
        plt.title('SLM Lookup Table')
        plt.show()


##############################
        ####### Script for obtaining the LUT
        #######  place the camera somewhere after the first slm (here, SLM2) and run
        #######  COpy the CSV file in the csv_files folder
##############################
'''
        phases = np.round(np.linspace(0,1020,103))
        reflection = np.full([103,1], np.nan)

        for idx, phase in enumerate(phases):
            SC.slm2.uniform_phase(int(phase))
            time.sleep(0.2)
            single_frame = SC.camera.captureFrames(2)[-1]
            reflection[idx] = np.mean(single_frame)


        reflection_offset = reflection - np.min(reflection)
        reflection_norm   = reflection_offset/np.max(reflection_offset)
        # Plot powers vector vs phases
        plt.figure(figsize=(10, 6))
        plt.plot(phases, powers_norm)
        plt.title('Reflection vs Phases')
        plt.xlabel('Phase (radians)')
        plt.ylabel('Reflection')
        plt.grid(True)
        plt.show()

        min_power_index = np.argmin(reflection_norm)
        reflection_monotone = reflection_norm[min_power_index:]

        # Corresponding phases vector for the second powers vector
        phases_second = phases[min_power_index:]

        # Creating a DataFrame for the second truncated vectors
        df = pd.DataFrame({
            'Phase': phases_second.flatten(),
            'Reflection': reflection_monotone.flatten() # Normalizing the second powers vector
        })

        # Adding header
        df.columns = ['Phase', 'Reflection']

        # Saving to CSV file
        df.to_csv('RvsPhi_slm1.csv', index=False)
'''