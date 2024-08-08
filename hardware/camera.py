import numpy as np
import json
import time
import os
import matplotlib.pyplot as plt
from .NIT import NITLibrary_x64_320_py38 as NITLibrary
from .device import Device

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
        self.config = _default_config
        self.device = None
        self.nitManager = NITLibrary.NITManager.getInstance()
        # self.bitDepth = _config.get('bitDepth', 14)  # Default to 14 if not provided
        self.frameObserver = myPyObserver()  # Observer for capturing frames

    @property
    def bitDepth(self):
        return self.config.get('bitDepth', 14)
    
    @property
    def exposure_time(self):
        return self.get_exposuretime()
    
    @exposure_time.setter
    def exposure_time(self, exposureTime):
        self.set_exposuretime(exposureTime)

    @property
    def analog_gain(self):
        return self.get_analog_gain()
    
    @analog_gain.setter
    def analog_gain(self, analog_gain):
        self.set_analog_gain(analog_gain)

    @property
    def offset_x(self):
        return self.get_offset_x()
    
    @offset_x.setter
    def offset_x(self, offset_x):
        self.set_offset_x(offset_x)

    @property
    def offset_y(self):
        return self.get_offset_y()
    
    @offset_y.setter
    def offset_y(self, offset_y):
        self.set_offset_y(offset_y)

    @property
    def frame_width(self):
        return self.get_frame_width()
    
    @frame_width.setter
    def frame_width(self, width):
        self.set_frame_width(width)
    
    @property
    def frame_height(self):
        return self.get_frame_height()
    
    @frame_height.setter
    def frame_height(self, height):
        self.set_frame_height(height)

    @property
    def NUC_path(self):
        return self.config['NUC_path']
    
    @NUC_path.setter
    def NUC_path(self, nuc_path):
        self.setNUCfile(nuc_path)

    @property
    def FPS(self):
        return self.config['FPS']
    
    @FPS.setter
    def FPS(self, fps):
        self.device.setFps(fps) # UNTESTED
        self.config['FPS'] = fps

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
        self.device.setParamValueOf("Exposure Time", self.config['ExposureTime'])
        self.device.setRoi(self.config['offset_x'], self.config['offset_y'],
                           self.config['width'], self.config['height'])
        self.device.updateConfig() #Send the new config to the camera

        #set FPS. See if the requested FPS is within the valid range
        min_fps = self.device.minFps()
        max_fps = self.device.maxFps()
        if self.config['FPS'] <= max_fps and self.config['FPS'] >= min_fps:
            self.device.setFps(self.config['FPS'])
        else:
            self.warning(self.devicename+": Requested FPS is out of acceptable range. FPS set to max allowable FPS.")
            self.device.setFps(max_fps)
        
        # self.device.setNucFile(self.config['NUC_path'])
        # self.device.activateNuc()

        if self.device.connectorType() == NITLibrary.GIGE:
            self.info(self.devicename+": GIGE Camera Detected")
            self.device.setParamValueOf("OutputType", "RAW")
        else:
            self.info(self.devicename+": "+"USB Camera Detected")

        self.device.setParamValueOf("Analog Gain", self.config['Analog_gain'])
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
        return True

    def disconnect(self):
        """
        Disconnect the camera and release resources.
        """
        self.nitManager.reset()
        self.info(self.devicename+": "+"Camera disconnected.")

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
        self.config['ExposureTime'] = exposureTime
        
    def get_exposuretime(self):
        return self.device.paramValueOf( "ExposureTime" ) 

    def setNUCfile(self, nuc_path):
        """
        Set the NUC file path
        """
        self.device.setNucFile(nuc_path)
        self.device.activateNuc()
        self.device.updateConfig()
        self.info(self.devicename+": "+"New NUC file set: " + nuc_path)
        self.config['NUC_path'] = nuc_path

    def set_frame_width(self, width):
        """
        Set the frame width of the camera. 
        """
        self.device.setParamValueOf("Width", width)
        self.device.updateConfig()
        self.info(self.devicename+": "+"Width set to "+ str(width))
        self.config['width'] = width
    
    def get_frame_width(self):
        return self.device.paramValueOf( "Width" )

    def set_frame_height(self, height):
        """
        Set the frame height of the camera. 
        """
        self.device.setParamValueOf("Height", height)
        self.device.updateConfig()
        self.info(self.devicename+": "+"Height set to "+ str(height))
        self.config['height'] = height
    
    def get_frame_height(self):
        return self.device.paramValueOf( "Height" )
    

    def set_offset_x(self, offset_x):
        """
        Set the X offset of the camera. 
        """
        self.device.setParamValueOf("OffsetX", offset_x)
        self.device.updateConfig()
        self.info(self.devicename+": "+"X offset set to " + str(offset_x)+ " pixels")
        self.config['offset_x'] = offset_x
    
    def get_offset_x(self):
        return self.device.paramValueOf( "OffsetX" )
    

    def set_offset_y(self, offset_y):
        """
        Set the Y offset of the camera. 
        """
        self.device.setParamValueOf("OffsetY", offset_y)
        self.device.updateConfig()
        self.info(self.devicename+": "+"Y offset set to "+ str(offset_y) + " pixels")
        self.config['offset_y'] = offset_y

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
        self.device.setParamValueOf("Analog Gain", analog_gain)
        self.device.updateConfig()
        self.info(self.devicename+": "+"analog gain set to "+ str(analog_gain))
        self.config['Analog_gain'] = analog_gain
    
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

    def captureFrames(self, numberOfFrames):
        """
        Capture a specified number of frames.
        :param numberOfFrames: The number of frames to capture.
        :return: List of captured images.
        """
        self.device.captureNFrames(numberOfFrames)
        self.device.waitEndCapture()
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
    
    #TODO: context management
    #def __enter__(self):
    #    return self.connectCam()

    #def __exit__(self, exc_type, exc_value, traceback):
    #    self.disconnect()
    #    print("Camera disconnected.")