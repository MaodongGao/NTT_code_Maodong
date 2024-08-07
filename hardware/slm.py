import numpy as np
import ctypes
import time
import matplotlib.pyplot as plt
from .santec import _slm_win as slm # Import the vendor-specific SLM functions
import os
from scipy.interpolate import interp1d
import warnings

SLM_OK = 0

class SLM:
    def __init__(self, config, SLM_number):
        """
        Initialize the SLM object with configuration settings.
        :param config: A dictionary containing configuration settings for all SLMs.
        :param SLM_number: The number of the specific SLM device to configure.
        """
        self.config = config
        self.SLM_number = SLM_number
        self.csv_directory = self.config['csv_directory']
        self.csv_filename = self.config["csv_filename"]
        self.slm_height = self.config['slm_height']
        self.slm_width = self.config['slm_width']

        self.lookuptable_csv_path = f"{self.csv_directory}\\{self.config['lookuptable_csv_path']}"
        
        lookup_table = np.genfromtxt(self.lookuptable_csv_path, delimiter=',', skip_header=1)
        # Extract phases and attenuations
        self.LUT_phase = lookup_table[:, 0]  # Assuming phase is in the first column
        self.LUT_reflection = lookup_table[:, 1]  # Assuming attenuation is in the second column
        # Create an interpolation function from attenuation to phase
        # We're flipping slm_phase and slm_attn because we want to find phase for a given attenuation
        self.phase_interp_func = interp1d(self.LUT_reflection, self.LUT_phase, kind='linear', bounds_error=False, fill_value="extrapolate")

        self.zero_reflection_phase = self.LUT_phase[0]

        slm.SLM_Ctrl_Close(self.SLM_number)
        ret = slm.SLM_Ctrl_Open(self.SLM_number)
        if ret == SLM_OK:
            print('SLM' + str(self.SLM_number) + ' successfully initialized.')

        setwl = 0
        if setwl == 1:
            # set wavelength
            ret = slm.SLM_Ctrl_WriteWL(self.SLM_number,1500,200)
            
            #save wavelength
            ret = slm.SLM_Ctrl_WriteAW(self.SLM_number)
    



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
        print('Zero-reflection phase on SLM' + str(self.SLM_number + 'updated to' + str(new_phase)))
    
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
        # print(filepath)
        with open(filepath, 'w', encoding='cp1252') as file:
            file.write("Y/X," + ','.join(map(str, range(1920))) + '\n')
            for idx, row in enumerate(phase_mask):
                line = str(idx) + ',' + ','.join(map(str, row.astype(int))) + '\n'
                file.write(line)
        #print("CSV file saved in " + filepath)
        return filepath


    def upload_phase_mask(self, phase_mask):
        """
        Load a full-panel phase array onto the SLM.

        :param phase_mask: The phase array to be loaded onto the SLM.
        """
        csvfilepath = self.generate_csv_from_phase_mask(phase_mask)
        self.upload_csv_memory_mode(csvfilepath)


    def embed_matrix(self, matrix, params, plot=False):
        """
    Generates a display of Nx by Ny pixels to show a matrix of size (sx,sy) on this display.
    Each matrix element is represented by a rectangular region of nx by ny pixels, separated by guard pixels.
    Pixels outside these regions are set to the value of "background".
    
    :param matrix: Input matrix to be displayed.
    :param params: Dictionary containing parameters for display generation.
    :param plot: Boolean indicating whether to plot the display matrix as a heatmap.
    :return: The display matrix as a numpy array.
        """
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