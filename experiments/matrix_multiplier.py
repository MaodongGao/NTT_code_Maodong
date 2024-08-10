# Add the parent directory to the path so that the imports work

import os
import sys
if os.path.dirname(os.path.dirname(os.path.abspath(__file__))) not in sys.path: 
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional

import numpy as np
from scipy.io import savemat

from hardware.camera import NITCam
from hardware.waveshaper import Waveshaper
from hardware.slm import SLM
from hardware.yokogawa_osa import YokogawaOSA

from hardware.device import Device # Create dummy device for logging

class MatrixMultiplier:
    def __init__(self, slm1: Optional[SLM],
                    slm2: Optional[SLM],
                    cam: Optional[NITCam],
                    osa: Optional[YokogawaOSA],
                    ws: Optional[Waveshaper]):
        self.slm1 = slm1
        self.slm2 = slm2
        self.cam = cam
        self.osa = osa
        self.ws = ws

        # Initialize matrices and images as empty arrays or None if more appropriate
        self.matrix1 = self.matrix2 = self.result_img = self.result_data = self.result_actual = None
        self.calibration_mask1 = self.calibration_mask2 = None
        self.dark_img = self.bright_img = self.uniform_img = None
        self.dark_data = []
        self.bright_data = self.uniform_data = None

        # Create dummy device for logging
        self.logger = Device(addr='', name='MatrixMultiplier', isVISA=False)
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical

        
    # For backwards compatibility
    @property
    def SysCont(self):
        class SystemController:
            def __init__(self, slm1: Optional[SLM], 
                            slm2: Optional[SLM], 
                            cam: Optional[NITCam], 
                            osa: Optional[YokogawaOSA], 
                            ws: Optional[Waveshaper]):
                self.slm1 = slm1
                self.slm2 = slm2
                self.osa = osa
                self.camera = cam
                if cam is not None:
                    self.element_ij_from_image = cam.element_ij_from_image
                    self.matrix_from_image = cam.matrix_from_image
                    self.plot_rect_array = cam.grid_config.plot_rect_array
                    # self.plot_rect_array_nonuniform = cam.grid_config.plot_rect_array_nonuniform

                self.ws = ws
                if ws is not None:
                    self.flatten_comb = ws.flatten_comb
        return SystemController(self.slm1, self.slm2, self.cam, self.ws, self.osa)
    
    @property
    def cam_config(self):
        return self.cam.config
    
    @cam_config.setter
    def cam_config(self, new_config):
        self.cam.config = new_config

    @property
    def grid_config_slm1(self):
        return self.slm1.grid_config
    
    @grid_config_slm1.setter
    def grid_config_slm1(self, new_config):
        self.slm1.grid_config = new_config

    @property
    def grid_config_slm2(self):
        return self.slm2.grid_config
    
    @grid_config_slm2.setter
    def grid_config_slm2(self, new_config):
        self.slm2.grid_config = new_config

    @property
    def grid_config_cam(self):
        return self.cam.grid_config
    
    @grid_config_cam.setter
    def grid_config_cam(self, new_config):
        self.cam.grid_config = new_config

    @property
    def rows(self):
        return self.grid_config_cam["matrixsize_0"]
    
    @rows.setter
    def rows(self, new_rows):
        self.grid_config_cam = {**self.grid_config_cam, "matrixsize_0": new_rows}
    
    @property
    def cols(self):
        return self.grid_config_cam["matrixsize_1"]
    
    @cols.setter
    def cols(self, new_cols):
        self.grid_config_cam = {**self.grid_config_cam, "matrixsize_1": new_cols}



    ## Below are copied from previous code matrix_multiplier.py
    def calibrate(self):
        #######################################
        # Calibration of System Nonuniformity #
        #######################################
        print("Calibration started:")

        #if self.dark_img is empty, capture a dark frame first: self.capture_dark()

        row_ratio = np.ones((self.cols,1))
        calibration_mask_1 = np.ones((self.rows,self.cols))
        calibration_mask_2 = np.ones((self.rows,self.cols))

        tensor_2 = np.zeros((self.rows, self.cols, self.cols))

        temp, temp2, temp3 = (np.zeros((self.rows, self.cols)) for _ in range(3))

        # Test attenuation masks for calibration
        # unit column vector in SLM1 & unit row vector in SLM2

        for j in range(self.cols):
            attenuation_mask_1 = np.zeros((self.rows, self.cols))
            attenuation_mask_1[:, j] = 1  # unit column vector in the attenuation_mask_1

            SLM1_pattern_attenuation_mask_1 = self.SysCont.slm1.get_phase_for_attenuation(attenuation_mask_1)
            __ = self.SysCont.slm1.embed_matrix(SLM1_pattern_attenuation_mask_1, self.grid_config_slm1, plot = False)

            for i in range(self.rows):
                attenuation_mask_2 = np.zeros((self.rows, self.cols))
                attenuation_mask_2[i, :] = 1  # unit row vector in the attenuation_mask_2

                SLM2_pattern_attenuation_mask_2 = self.SysCont.slm2.get_phase_for_attenuation(attenuation_mask_2)
                __ = self.SysCont.slm2.embed_matrix(SLM2_pattern_attenuation_mask_2, self.grid_config_slm2, plot = False)
                ##########
    
                img  = self.SysCont.camera.captureFrames(2)[-1]
                image_data = self.SysCont.matrix_from_image(img - self.dark_img , self.grid_config_cam)


                for k in range(self.cols):
                        tensor_2[i][j][k] =(image_data.T)[j,k] 
                # print(image_data)
                # print(tensor_2)



        # unit row vector in SLM1 & all 1 matrix in SLM2

        attenuation_mask_2 = np.ones((self.rows, self.cols))
        SLM2_pattern_attenuation_mask_2 = self.SysCont.slm2.get_phase_for_attenuation(attenuation_mask_2)
        __ = self.SysCont.slm2.embed_matrix(SLM2_pattern_attenuation_mask_2, self.grid_config_slm2, plot = False)

        for j in range(self.rows):
            attenuation_mask_1 = np.zeros((self.rows, self.cols))
            attenuation_mask_1[j, :] = 1  # unit rows vector in the attenuation_mask_1
            SLM1_pattern_attenuation_mask_1 = self.SysCont.slm1.get_phase_for_attenuation(attenuation_mask_1)
            __ = self.SysCont.slm1.embed_matrix(SLM1_pattern_attenuation_mask_1, self.grid_config_slm1, plot = False)

            img  = self.SysCont.camera.captureFrames(2)[-1]
            image_data = self.SysCont.matrix_from_image(img - self.dark_img , self.grid_config_cam)

            image_data_VerticalSum = np.sum(image_data.T,axis = 0)
            temp3[j,:] = image_data_VerticalSum


        # # calibration_depends on rows
        for i in range(self.cols):
            row_ratio[i] = sum(temp3[j][i] for j in range(self.rows))

        row_ratio = np.min(row_ratio)/row_ratio

        # # calibration_mask_2 (SLM2)
        for i in range(self.rows):
            for k in range(self.cols):
                temp2[i][k] = sum(tensor_2[i][j][k] for j in range(self.cols))


        for i in range(self.rows):
            for k in range(self.cols):
                calibration_mask_2[i][k] = np.min(temp2)/temp2[i][k] # *row_ratio[i]

        calibration_mask_2 = calibration_mask_2/np.max(calibration_mask_2)

        self.tensor_2 = tensor_2

        # Test attenuation masks for calibration
        # unit column vector in SLM1 & unit row vector in SLM2

        for j in range(self.cols):
            attenuation_mask_1 = np.zeros((self.rows, self.cols))
            attenuation_mask_1[:, j] = 1  # unit column vector in the attenuation_mask_1
            SLM1_pattern_attenuation_mask_1 = self.SysCont.slm1.get_phase_for_attenuation(attenuation_mask_1)
            __ = self.SysCont.slm1.embed_matrix(SLM1_pattern_attenuation_mask_1, self.grid_config_slm1, plot = False)
            for i in range(self.rows):
                attenuation_mask_2 = np.zeros((self.rows, self.cols))
                attenuation_mask_2[i, :] = 1  # unit row vector in the attenuation_mask_2
                SLM2_pattern_attenuation_mask_2 = self.SysCont.slm2.get_phase_for_attenuation(attenuation_mask_2*calibration_mask_2)
                __ = self.SysCont.slm2.embed_matrix(SLM2_pattern_attenuation_mask_2, self.grid_config_slm2, plot = False)
                ##########
                img  = self.SysCont.camera.captureFrames(2)[-1]
                image_data = self.SysCont.matrix_from_image(img - self.dark_img , self.grid_config_cam)

                for k in range(self.cols):
                        tensor_2[i][j][k] = (image_data.T)[j,k] ## to do


        #######################
        #Calculate Calibration matrices
        #######################
        for i in range(self.rows):
            for j in range(self.cols):
                temp[i][j] = sum(tensor_2[i][j][k] for k in range(self.cols))

        for i in range(self.rows):
            for j in range(self.cols):
                calibration_mask_1[i][j] = np.min(temp)/temp[i][j]

        self.calibration_mask_1 = calibration_mask_1
        self.calibration_mask_2 = calibration_mask_2

                # Save the calibration masks and tensor as numpy arrays
        np.save('calibration_mask_1.npy', self.calibration_mask_1)
        np.save('calibration_mask_2.npy', self.calibration_mask_2)
        np.save('tensor_2.npy', self.tensor_2)

        # Also save them as MATLAB files
        savemat('calibration_data.mat', {
            'calibration_mask_1': self.calibration_mask_1,
            'calibration_mask_2': self.calibration_mask_2,
            'tensor_2': self.tensor_2
        })

        print("Calibration data saved.")
        return self.calibration_mask_1, self.calibration_mask_2


    def capture_dark(self, N_frames):
        self.SysCont.camera.clearFrames()  # Clear the camera buffer

        # Capture dark frame (with SLMs off or blocked)
        zero_r_phase1 = self.SysCont.slm1.get_phase_for_attenuation(0)
        zero_r_phase2 = self.SysCont.slm2.get_phase_for_attenuation(0)
        self.SysCont.slm1.uniform_phase(int(zero_r_phase1))
        self.SysCont.slm2.uniform_phase(int(zero_r_phase2))

        dark_img = np.mean(self.SysCont.camera.captureFrames(N_frames)[0:N_frames],axis = 0)#[-1]
        print(np.shape(dark_img))
        self.dark_img = dark_img
        self.dark_data = self.SysCont.matrix_from_image(self.dark_img, self.grid_config_cam)

        self.logger.info(f"{np.shape(self.dark_data)} Dark data captured with dark_img shape: {np.shape(self.dark_img)}.", ndarray=self.dark_data)
    
        return self.dark_data

    def capture_bright(self):
        # Capture bright frame (with both SLMs set to all-1 matrices)
        total_r_phase1 = self.SysCont.slm1.get_phase_for_attenuation(1)
        total_r_phase2 = self.SysCont.slm2.get_phase_for_attenuation(1)
        matrix1 = total_r_phase1 * np.ones((self.grid_config_slm1["matrixsize_0"], self.grid_config_slm1["matrixsize_1"] ))
        matrix2 = total_r_phase2 * np.ones((self.grid_config_slm2["matrixsize_0"], self.grid_config_slm2["matrixsize_1"] ))

        __ = self.SysCont.slm1.embed_matrix(matrix1, self.grid_config_slm1, plot = False)
        __ = self.SysCont.slm2.embed_matrix(matrix2, self.grid_config_slm2, plot = False)

        self.bright_img  = self.SysCont.camera.captureFrames(2)[-1]
        self.bright_data = self.SysCont.matrix_from_image(self.bright_img, self.grid_config_cam)

        return self.bright_data
        
        

    def multiply(self, matrix1, matrix2):
        # Check if all elements in both matrices are between 0 and 1
        if not (np.all((0 <= matrix1) & (matrix1 <= 1)) and np.all((0 <= matrix2) & (matrix2 <= 1))):
            e = ValueError("All elements in both matrices must be between 0 and 1.")
            self.error(str(e))
            raise e
        
        # Check if the matrices are sized correctly for multiplication
        if matrix1.shape[1] != matrix2.shape[0]:
            # warnings.warn("The size of the matrices is not compatible for multiplication. "
            #               "The number of columns in the first matrix must equal the number of rows in the second matrix.",
            #               UserWarning)
            self.warning("The size of the matrices is not compatible for multiplication. "
                          "The number of columns in the first matrix must equal the number of rows in the second matrix.")
        
        # If the checks pass, proceed with multiplication
        
        matrix1_phase = self.SysCont.slm1.get_phase_for_attenuation(matrix1)
        matrix2_phase = self.SysCont.slm2.get_phase_for_attenuation(matrix2)

        __ = self.SysCont.slm1.embed_matrix(matrix1_phase, self.grid_config_slm1, plot = False)
        __ = self.SysCont.slm2.embed_matrix(matrix2_phase, self.grid_config_slm2, plot = False)

        self.result_img  = self.SysCont.camera.captureFrames(2)[-1]
        self.result_data = self.SysCont.matrix_from_image(self.result_img, self.grid_config_cam)

        return self.result_data
    

    
    
