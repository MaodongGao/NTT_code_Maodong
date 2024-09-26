import numpy as np
import json
import time
import os
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="to-Python converter for unsigned int already registered")
from .NIT import NITLibrary_x64_320_py38 as NITLibrary

from .device import Device
from .utils.grid_config import GridConfig

#############################################################################
#############################################################################
#########   The Observer class that "observes" the frmes captured by the device
#############################################################################
#############################################################################

class myPyObserver(NITLibrary.NITUserObserver):
    def __init__(self):
        NITLibrary.NITUserObserver.__init__(self)
        self.images = []
        self.counter = 0
        print("Observer initialized")

    def onNewFrame(self, frame):
        """
        Callback function called on each new frame.
        :param frame: The new frame received from the camera.
        """
        new_frame = np.copy(frame.data())
        self.images.append(new_frame)
        self.counter += 1
        print(f"Frame {self.counter}", end="\r")

    def clearFrame(self):
        """
        Callback function called on each new frame.
        :param frame: The new frame received from the camera.
        """
        self.images = []
        self.counter = 0


#############################################################################
#############################################################################
#########   The camera class that implements the camera object
#############################################################################
#############################################################################

class NITCam(Device):
    def __init__(self, addr=None, name='NIT Camera', isVISA=False):
        """
        Initialize the NITCam with the given configuration.
        :param config: Configuration dictionary for the camera.
        """
        super().__init__(addr, name, isVISA)
        self.unconnected_warning = False
        _default_config = {
            "bitDepth": 14,
            "ExposureTime": 4000,
            "Analog_gain": "Low",
            "FPS": 30,
            "offset_x": 0,
            "offset_y": 0,
            "width": 1280,
            "height": 1024,
            "NUC_path": os.path.join(os.path.dirname(__file__), 'NUCs', 'NUC_4000us.yml')
            #os.path.join(ConfigManager.PACKAGE_ROOT, 'hardware', 'NUCs', 'NUC_4000us.yml')
            }
        self.__config = _default_config
        self.device = None
        self.nitManager = NITLibrary.NITManager.getInstance()
        # self.bitDepth = _config.get('bitDepth', 14)  # Default to 14 if not provided
        self.frameObserver = myPyObserver()  # Observer for capturing frames

        self._grid_config = GridConfig() # dictionary like object for grid configuration

    def update_grid_config(self, params):
        self._grid_config.update(params)
        self.info(self.devicename + ': Grid configuration updated to:\n' + str(self.grid_config))

    @property
    def grid_config(self):
        return self._grid_config
    
    @grid_config.setter
    def grid_config(self, params):
        self.update_grid_config(params)

    
    @property
    def config(self):
        if (not self.connected) and self.unconnected_warning: self.warning(self.devicename + f": Camera is not connected. Returning default configuration {self.__config}")
        return self.__config
    
    @config.setter
    def config(self, new_config):
        try:
            self.update_config(new_config)
        except Exception as e:
            self.error(self.devicename + f": Camera must be connected to update configuration. Error: {e}")
            raise e
            
    @property
    def bitDepth(self):
        return self.__config.get('bitDepth', 14)
    
    @bitDepth.setter
    def bitDepth(self, bitDepth):
        self.__config['bitDepth'] = bitDepth
        self.info(self.devicename + ": Bit depth configured to " + str(bitDepth)+'. This setting will be updated after connection or calling update_config method.')
    
    @property
    def exposure_time(self):
        if not self.connected:
            if self.unconnected_warning: self.warning(self.devicename + f": Camera is not connected. Returning default exposure time {self.__config['ExposureTime']}")
            return self.__config['ExposureTime']
        else:
            try:
                return self.get_exposuretime()
            except Exception as e:
                self.error(self.devicename + f": Error getting exposure time: {e}")
                raise e
            
    def __exposure_time_nuc_path(self, exposure_time):
        return os.path.normpath(os.path.join(os.path.dirname(__file__), 'NUCs', 'NUC_' + str(int(exposure_time)) + 'us.yml'))
    
    @exposure_time.setter
    def exposure_time(self, exposureTime):
        if not self.connected:
            if self.unconnected_warning: self.warning(self.devicename + f": Camera is not connected. Setting default exposure time {self.__config['ExposureTime']}. Set the exposure time after connecting the camera.")
            self.__config['ExposureTime'] = exposureTime
            self.__config['NUC_path'] = self.__exposure_time_nuc_path(exposureTime)
        else:
            try:
                self.set_exposuretime(exposureTime)
                self.setNUCfile(self.__exposure_time_nuc_path(exposureTime))
            except Exception as e:
                self.error(self.devicename + f": Error setting exposure time: {e}")
                raise e
        # self.set_exposuretime(exposureTime)

    @property
    def analog_gain(self):
        if not self.connected:
            if self.unconnected_warning: self.warning(self.devicename + f": Camera is not connected. Returning default analog gain {self.__config['Analog_gain']}")
            return self.__config['Analog_gain']
        else:
            try:
                return CameraAnalogGain(self.get_analog_gain()).name
            except Exception as e:
                self.error(self.devicename + f": Error getting analog gain: {e}")
                raise e
    
    @analog_gain.setter
    def analog_gain(self, analog_gain):
        if not self.connected:
            if self.unconnected_warning: self.warning(self.devicename + f": Camera is not connected. Setting default analog gain {self.__config['Analog_gain']}. Set the analog gain after connecting the camera.")
            self.__config['Analog_gain'] = analog_gain
        else:
            try:
                self.set_analog_gain(analog_gain)
            except Exception as e:
                self.error(self.devicename + f": Error setting analog gain: {e}")
                raise e
        # self.set_analog_gain(analog_gain)

    @property
    def offset_x(self):
        if not self.connected:
            if self.unconnected_warning: self.warning(self.devicename + f": Camera is not connected. Returning default X offset {self.__config['offset_x']}")
            return self.__config['offset_x']
        else:
            try:
                return self.get_offset_x()
            except Exception as e:
                self.error(self.devicename + f": Error getting X offset: {e}")
    
    @offset_x.setter
    def offset_x(self, offset_x):
        if not self.connected:
            if self.unconnected_warning: self.warning(self.devicename + f": Camera is not connected. Setting default X offset {self.__config['offset_x']}. Set the X offset after connecting the camera.")
            self.__config['offset_x'] = offset_x
        else:
            try:
                self.set_offset_x(offset_x)
            except Exception as e:
                self.error(self.devicename + f": Error setting X offset: {e}")
                raise e
        # self.set_offset_x(offset_x)

    @property
    def offset_y(self):
        if not self.connected:
            if self.unconnected_warning: self.warning(self.devicename + f": Camera is not connected. Returning default Y offset {self.__config['offset_y']}")
            return self.__config['offset_y']
        else:
            try:
                return self.get_offset_y()
            except Exception as e:
                self.error(self.devicename + f": Error getting Y offset: {e}")
    
    @offset_y.setter
    def offset_y(self, offset_y):
        if not self.connected:
            if self.unconnected_warning: self.warning(self.devicename + f": Camera is not connected. Setting default Y offset {self.__config['offset_y']}. Set the Y offset after connecting the camera.")
            self.__config['offset_y'] = offset_y
        else:
            try:
                self.set_offset_y(offset_y)
            except Exception as e:
                self.error(self.devicename + f": Error setting Y offset: {e}")
                raise e
        # self.set_offset_y(offset_y)

    @property
    def frame_width(self):
        if not self.connected:
            if self.unconnected_warning: self.warning(self.devicename + f": Camera is not connected. Returning default frame width {self.__config['width']}")
            return self.__config['width']
        else:
            try:
                return self.get_frame_width()
            except Exception as e:
                self.error(self.devicename + f": Error getting frame width: {e}")
                raise e
    
    @frame_width.setter
    def frame_width(self, width):
        if not self.connected:
            if self.unconnected_warning: self.warning(self.devicename + f": Camera is not connected. Setting default frame width {self.__config['width']}. Set the frame width after connecting the camera.")
            self.__config['width'] = width
        else:
            try:
                self.set_frame_width(width)
            except Exception as e:
                self.error(self.devicename + f": Error setting frame width: {e}")
                raise e
        # self.set_frame_width(width)
    
    @property
    def frame_height(self):
        if not self.connected:
            if self.unconnected_warning: self.warning(self.devicename + f": Camera is not connected. Returning default frame height {self.__config['height']}")
            return self.__config['height']
        else:
            try:
                return self.get_frame_height()
            except Exception as e:
                self.error(self.devicename + f": Error getting frame height: {e}")
    
    @frame_height.setter
    def frame_height(self, height):
        if not self.connected:
            if self.unconnected_warning: self.warning(self.devicename + f": Camera is not connected. Setting default frame height {self.__config['height']}. Set the frame height after connecting the camera.")
            self.__config['height'] = height
        else:
            try:
                self.set_frame_height(height)
            except Exception as e:
                self.error(self.devicename + f": Error setting frame height: {e}")
                raise e
        # self.set_frame_height(height)

    @property
    def NUC_path(self):
        return self.__config['NUC_path']
    
    @NUC_path.setter
    def NUC_path(self, nuc_path):
        if not self.connected:
            if self.unconnected_warning: self.warning(self.devicename + f": Camera is not connected. Setting default NUC path {self.__config['NUC_path']}. Set the NUC path after connecting the camera.")
            self.__config['NUC_path'] = nuc_path
        else:
            try:
                self.setNUCfile(nuc_path)
            except Exception as e:
                self.error(self.devicename + f": Error setting NUC path: {e}")
                raise e

    @property
    def FPS(self):
        return self.__config['FPS']
    
    @FPS.setter
    def FPS(self, fps):
        # e = NotImplementedError("Setting FPS is not implemented for this camera.")
        # self.error(self.devicename + f": {e}")
        # raise e
        self.device.setFps(fps) # UNTESTED
        self.info(self.devicename + ": FPS set to " + str(fps))
        self.__config['FPS'] = fps

########################################################
#########     connect and disconnect
########################################################

    def connect(self): 
        # Ensure compatibility with the Device class
        self.connectCam()

    def connectCam(self):
        """
        Connect and configure the camera based on the provided settings.
        """
        self.device = self.nitManager.openOneDevice()
        if self.device is None:
            self.info(self.devicename+": No device Connected")
            return False  # Or raise an exception

        # Connect to the camera using the given configuration settings
        self.device.setParamValueOf("Exposure Time", self.__config['ExposureTime'])
        self.device.setRoi(self.__config['offset_x'], self.__config['offset_y'],
                           self.__config['width'], self.__config['height'])
        self.device.updateConfig() #Send the new config to the camera

        #set FPS. See if the requested FPS is within the valid range
        min_fps = self.device.minFps()
        max_fps = self.device.maxFps()
        if self.__config['FPS'] <= max_fps and self.__config['FPS'] >= min_fps:
            self.device.setFps(self.__config['FPS'])
        else:
            self.warning(self.devicename+": Requested FPS is out of acceptable range. FPS set to max allowable FPS.")
            self.device.setFps(max_fps)
        
        # self.device.setNucFile(self.__config['NUC_path'])
        # self.device.activateNuc()

        if self.device.connectorType() == NITLibrary.GIGE:
            self.info(self.devicename+": GIGE Camera Detected")
            self.device.setParamValueOf("OutputType", "RAW")
        else:
            self.info(self.devicename+": "+"USB Camera Detected")

        self.device.setParamValueOf("Analog Gain", self.__config['Analog_gain'])
        # TODO: implement TEC here and in the config file
        #self.device.setParamValueOf("Tec Mode", 3)
        #self.device.setParamValueOf("Temperature Tec", 25.0)
        self.device.updateConfig()

        if self.bitDepth == 8:
            self.automaticGainControl = NITLibrary.NITToolBox.NITAutomaticGainControl()
            self.device << self.automaticGainControl << self.frameObserver
        elif self.bitDepth == 14:
            self.device << self.frameObserver

        self.info(self.devicename+": "+"Camera connected successfully.")
        self.connected = True
        return True

    def disconnect(self):
        """
        Disconnect the camera and release resources.
        """
        self.nitManager.reset()
        self.info(self.devicename+": "+"Camera disconnected.")
        self.connected = False

########################################################
#########     set or get values of key camera parameters
########################################################

    def set_exposuretime(self, exposureTime):
        """
        Set the exposure time of the camera. 
        :param exposureTime: The exposure time in milliseconds.
        """
        self.device.setParamValueOf("Exposure Time", exposureTime)
        self.device.updateConfig()
        self.info(self.devicename+": "+"Exposure time set to "
                  + str(exposureTime) + "ms")
        self.__config['ExposureTime'] = exposureTime
        
    def get_exposuretime(self):
        return self.device.paramValueOf( "ExposureTime" ) 

    def setNUCfile(self, nuc_path):
        """
        Set the NUC file path
        """
        # Check if the file exists, raise an error if it doesn't
        if not os.path.exists(nuc_path):
            self.error(self.devicename+": "+"NUC file not found: " + nuc_path)
            return
            # raise FileNotFoundError(f"NUC file not found: {nuc_path}")
        self.device.setNucFile(nuc_path)
        self.device.activateNuc()
        self.device.updateConfig()
        self.info(self.devicename+": "+"New NUC file set: " + nuc_path)
        self.__config['NUC_path'] = nuc_path

    def set_frame_width(self, width):
        """
        Set the frame width of the camera. 
        """
        self.device.setParamValueOf("Width", width)
        self.device.updateConfig()
        self.info(self.devicename+": "+"Width set to "+ str(width))
        self.__config['width'] = width
    
    def get_frame_width(self):
        return self.device.paramValueOf( "Width" )

    def set_frame_height(self, height):
        """
        Set the frame height of the camera. 
        """
        self.device.setParamValueOf("Height", height)
        self.device.updateConfig()
        self.info(self.devicename+": "+"Height set to "+ str(height))
        self.__config['height'] = height
    
    def get_frame_height(self):
        return self.device.paramValueOf( "Height" )
    

    def set_offset_x(self, offset_x):
        """
        Set the X offset of the camera. 
        """
        self.device.setParamValueOf("OffsetX", offset_x)
        self.device.updateConfig()
        self.info(self.devicename+": "+"X offset set to " + str(offset_x)+ " pixels")
        self.__config['offset_x'] = offset_x
    
    def get_offset_x(self):
        return self.device.paramValueOf( "OffsetX" )
    

    def set_offset_y(self, offset_y):
        """
        Set the Y offset of the camera. 
        """
        self.device.setParamValueOf("OffsetY", offset_y)
        self.device.updateConfig()
        self.info(self.devicename+": "+"Y offset set to "+ str(offset_y) + " pixels")
        self.__config['offset_y'] = offset_y

    def set_temperature(self, tec_mode, temperature):
            self.device.setParamValueOf( "Tec Mode", tec_mode )
            self.device.setParamValueOf( "Temperature Tec", temperature )
            self.device.updateConfig()
            self.info(self.devicename+": "+"TEC mode set to " + str(tec_mode) + "and temp is set to " + str(temperature))
    
    def get_offset_y(self):
        return self.device.paramValueOf( "OffsetY" )
    
    

    def set_analog_gain(self, analog_gain):
        """
        Set the analog gain of the camera. 
        """
        analog_gain = CameraAnalogGain(analog_gain)
        self.device.setParamValueOf("Analog Gain", str(analog_gain))
        self.device.updateConfig()
        self.info(self.devicename+": "+"analog gain set to "+ str(analog_gain))
        self.__config['Analog_gain'] = str(analog_gain)
    
    def get_analog_gain(self):
        return self.device.paramValueOf( "Analog Gain" )
     
    def update_config(self, new_config):
        """
        Update the camera's configuration using the provided configuration dictionary.
        
        :param new_config: A dictionary with the new configuration settings.
        """
        # Iterate through the new configuration dictionary
        for key, value in new_config.items():
            # Check and call the appropriate method to update the camera setting
            if key == 'ExposureTime':
                self.set_exposuretime(value)
            elif key == 'Analog_gain':
                self.set_analog_gain(value)
            elif key == 'FPS':
                # Directly set FPS if a specific method is not available
                # Example: self.device.setFps(value) if direct attribute setting is supported
                pass
            elif key == 'offset_x':
                self.set_offset_x(value)
            elif key == 'offset_y':
                self.set_offset_y(value)
            elif key == 'width':
                self.set_frame_width(value)
            elif key == 'height':
                self.set_frame_height(value)
            elif key == 'NUC_path':
                self.setNUCfile(value)
            # Add more elif blocks for other keys as necessary

        # Update the camera's configuration to apply changes
        if self.device is not None:
            self.device.updateConfig()
            # Print confirmation message with the applied configuration
            self.info(self.devicename+": "+"The following config successfully applied:\n"+json.dumps(new_config, indent=4))
        else:
            self.info(self.devicename+": "+"Camera device is not initialized. Please connect the camera first.")
        

    def clearFrames(self):
        self.frameObserver.clearFrame()
        self.info(self.devicename+": "+"Cleared all frames.")

############################################################
##############         Methods for capturing frames
############################################################

    def capture_single_frame(self, save_log = False):
        single_frame = self.captureFrames(2)[-1]
        if save_log:
            self.info(self.devicename + "Logged Captured Single Frame.", ndarray=(single_frame*1e5).astype(int))
            # with np.printoptions(threshold=np.inf, precision=0, suppress=True):
            #     tt = str((single_frame*1e5).astype(int)) # 5 significant digits
            #     self.info(self.devicename + ": Captured frame: \n" + tt)
            #     self.info(self.devicename + f": Single Frame logged, memory size: {len(tt.encode('utf-8'))/1024/1024:.3f} MB")
        return single_frame


    def captureFrames(self, numberOfFrames, save_log = False, clear_frames = False):
        """
        Capture a specified number of frames.
        :param numberOfFrames: The number of frames to capture.
        :return: List of captured images.
        """
        if save_log and not clear_frames:
            # only proceed in this case after manual confirmation
            input(f"Warning: Logging frames without clearing the previous frames. This will take up many storage space. Press Enter to confirm you want to proceed...")
        if clear_frames:
            self.clearFrames()
        # minimum number of frames to capture is 2
        if numberOfFrames <= 1:
            raise ValueError("Number of frames must be at least 2.")
        
        self.device.captureNFrames(numberOfFrames)
        self.device.waitEndCapture()
        if save_log:
            self.info(self.devicename + f": Logging {self.frameObserver.images.__len__()} Captured frames: \n", ndarray=(np.array(self.frameObserver.images)*1e5).astype(int))
        return self.frameObserver.images
    
    def captureForDuration(self, duration):
        """
        Capture for a given duration in milliseconds
        :param duration: 
        :return: List of captured images.
        """
        self.device.captureForDuration(duration)
        self.device.waitEndCapture()
        return self.frameObserver.images

    
    '''
    # TODO: third way: start and stop.
    def captureStart(self):
        """
        Start continuous capture until stopped
        """
        self.device.start()
    
    def captureStop(self):
        """
        stop the capture
        """
        self.device.stop()
        return self.frameObserver.images
    
    '''

    def capture_mean(self, num_frames = 2 , config = None, plot = False ):
        """
        Using captureFrames, Capture a specified number of frames with some 
        exposure time and plot their average.
        :param numberOfFrames: The number of frames to capture.
        :return: List of captured images.
        """
        if num_frames <=1:
            raise ValueError("Number of frames must be larger than 1.")
        
        # Connect and configure the camera
        if not self.connectCam():
            raise Exception("Failed to connect the camera.")
        
        if config != None:
            self.update_config(self, config)
        
        # Capture frames
        captured_frames = self.captureFrames(num_frames)
        print('sleeping for 1 second to transfer the frames...')
        time.sleep(1)
        frames_mean = np.mean(np.stack(captured_frames), axis = 0)
        if plot == True:
            plt.figure(figsize=(10, 10))
            plt.imshow(frames_mean, cmap='inferno')
            plt.colorbar()
            plt.show()

        return frames_mean
    
    @classmethod
    def plot_frame(self, frame, title = None, cmap = 'gray'):
        fig, ax = plt.subplots()
        cax = ax.imshow(frame, cmap=cmap)
        fig.colorbar(cax)
        if title != None:
            ax.set_title(title)
        plt.show()
    
    # this code was adapted from systemcontroller.py(deprecated)
    @classmethod
    def element_ij_from_image(self, img: np.ndarray, index: tuple, elemsize: tuple, topleft: tuple, gap: tuple) -> float:
        #extract and return data(index). Here index = (i,j)
        topleft_ij  = topleft + (elemsize + gap)*np.array([index[1], index[0]])
        data_ij       = img[ topleft_ij[1]: topleft_ij[1] + elemsize[1], topleft_ij[0]: topleft_ij[0] + elemsize[0]]
        return np.mean(data_ij)

    # this code was adapted from systemcontroller.py(deprecated)
    @classmethod
    def matrix_from_image(self, img, grid_config=None):
        """
        sample grid_config dictionary:
         grid_config = {
        "matrixsize_0": 10,
        "matrixsize_1": 10,
        "elem_width": 14,
        "elem_height": 14,
        "topleft_x": 10,
        "topleft_y": 10,
        "gap_x": 5,
        "gap_y": 5 }
        """
        if grid_config is None:
            grid_config = self.grid_config

        N           =   [grid_config["matrixsize_0"], grid_config["matrixsize_1"]]
        topleft     =   np.array([grid_config["topleft_x"], grid_config["topleft_y"]])
        elem_size   =   np.array([grid_config["elem_width"], grid_config["elem_height"]])
        gap         =   np.array([grid_config["gap_x"], grid_config["gap_y"]])

        data = np.empty(N)
        for n0 in range(N[0]):
            for n1 in range(N[1]):
                # data[n0,n1] = self.element_ij_from_image(img, [n0,n1], elem_size, topleft+np.array([offset_x[n1,n0], offset_y[n1,n0]]), gap)
                data[n0,n1] = self.element_ij_from_image(img, [n0,n1], elem_size, topleft, gap)
        return data

    
    #TODO: context management
    #def __enter__(self):
    #    return self.connectCam()

    #def __exit__(self, exc_type, exc_value, traceback):
    #    self.disconnect()
    #    print("Camera disconnected.")

class CameraAnalogGain:
    _analog_gain_map = {
        "LOW": 0,
        "HIGH": 1
    }

    _reversed_map = {v: k for k, v in _analog_gain_map.items()}

    def __init__(self, value):
        # Check if value is an integer, a float integer, or a string
        if isinstance(value, int):
            self._init_with_int(value)
        elif isinstance(value, float) and value.is_integer():
            # Convert float integer to int and initialize
            self._init_with_int(int(value))
        elif isinstance(value, str):
            self._init_with_str(value)
        else:
            raise TypeError(f"Value must be an integer, a float integer, or a string, not {type(value).__name__}.")


    def _init_with_int(self, value):
        if value in self._reversed_map:
            self.value = value
            self.name = self._reversed_map[value]
        else:
            raise ValueError(f"Invalid analog gain value: {value}. Valid values are {', '.join(self._reversed_map.keys())}.")

    def _init_with_str(self, value):
        normalized_value = value.strip().upper()
        if normalized_value.isdigit():  # Check if it's a string representation of an integer
            int_value = int(normalized_value)
            self._init_with_int(int_value)
        elif normalized_value in self._analog_gain_map:
            self.value = self._analog_gain_map[normalized_value]
            self.name = normalized_value
        else:
            raise ValueError(
                f"Invalid analog gain name or value: '{value}'. "
                "Valid names are: " + ", ".join(self._analog_gain_map.keys())
            )

    def __str__(self):
        return self.name

    @classmethod
    def get_name(cls, value):
        if value in cls._reversed_map:
            return cls._reversed_map[value]
        raise ValueError(f"Analog gain value should be {', '.join(cls._reversed_map.keys())}, not {value}.")

    @classmethod
    def get_value(cls, name):
        normalized_name = name.strip().upper()
        if normalized_name in cls._analog_gain_map:
            return cls._analog_gain_map[normalized_name]
        raise ValueError(
            f"Invalid analog gain name: '{name}'. "
            "Valid names are: " + ", ".join(cls._analog_gain_map.keys())
        )
