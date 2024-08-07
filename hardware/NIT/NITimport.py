#Here we are creating a wrapper for the NIT camera so that we can make calls to it to "capture" a frame and return the data as a numpy array
import matlab.engine
import NITLibrary_x64_320_py39 as NITLibrary #Change xxx to match the version of NITLibrary you are using. Change ZZ to match the python version you are using
import numpy as np
import time
import heapq
import threading 
import cv2
from scipy.io import savemat

#from PIL import Image
import time
#import matplotlib.pyplot as plt
import os


class NITCam():

    def __init__(self):
        self.dev = None
        self.agc = None
        self.player = None
        self.nm = None
        self.myObserver = None
        self.images = []
        self.bitDepth=None

    def connectCam(self,setBitDepth, offset_x,offset_y, width, height):
        self.bitDepth=setBitDepth 
        '''If bitDepth=8: an AGC filter is added to the pipeline to convert incoming images to 8bit. If fixed conversion parameters are desired, switch to a NITManualGainControl filter
        bitDepth=14: images with no processing are provided'''
        self.nm = NITLibrary.NITManager.getInstance()
        self.dev = self.nm.openOneDevice()
        # If self.device is not open
        if (self.dev == None):
            print("No self.device Connected")
        else:
            self.dev.setParamValueOf("Exposure Time", 2000.0 )
            self.dev.updateConfig()

            if (
                    self.dev.connectorType() == NITLibrary.GIGE):  # To use self.agc camera must provide a RAW data; as Gige cameras can output different data types,
                # we force RAW output type
                print("GIGE CAM")
                self.dev.setParamValueOf("OutputType", "RAW").updateconfig()
            else:
                print("USB CAM")
            #def setRoi(offset_x, offset_y, width, height)
            #self.dev.setRoi(604,204,176,640)
            self.dev.setRoi(offset_x, offset_y, width, height)
            self.dev.updateConfig()
                # Fps configuration: set fps to a mean value
            min_fps = self.dev.minFps()
            max_fps = self.dev.maxFps()
            #self.dev.setParamValueOf( "Analog Gain", "Low" )
            #self.dev.updateConfig()
            self.dev.setFps(100)
            self.dev.updateConfig()
            self.dev.setNucFile("C:\\Users\\NTTRi LAB\\Desktop\\MVM\\Camera2d\\NUC\\NUCFactory_2000us.yml")
            self.dev.activateNuc()
            self.dev.updateConfig()

            self.mpo = myPyObserver()
            if self.bitDepth==8:
                self.myAGC = NITLibrary.NITToolBox.NITAutomaticGainControl() #Automatic gain control, allows to convert 14 bit images coming from the camera to 8 bit for displaying
                                                                             #For fixed conversion parameter change this to a NITManualGainControl
                self.dev << self.myAGC << self.mpo
            elif self.bitDepth==14:
                self.dev << self.mpo #Raw processing pipeline, outputs 14 bit images

            print("Device connected")
            #self.dev.start()
            #time.sleep(10)
            #self.dev.stop()

    def captureFrames(self, nbFrames=5):
        # Start Capture!!!
        self.dev.captureNFrames(nbFrames)
        self.dev.waitEndCapture()  # Wait end of capture
        self.images = self.mpo.images
        return (self.images)

    def disconnect(self):
        # Don't forget Observers Diconnection
        self.nm.reset()

class myPyObserver(NITLibrary.NITUserObserver):
    def __init__(self):
        NITLibrary.NITUserObserver.__init__(self)
        self.images = []
        self.counter=0
        print("init")

    def onNewFrame(self, frame):  # You MUST define this function - It will be called on each frames
        # print("On new frame")
        new_frame = np.copy(frame.data())
        self.images.append(new_frame)
        self.counter+=1
        print(self.counter, end="\r")


def CaptureNFramesAvg(cam, offset_x,offset_y, width, height, nbFrames):
        imgData_tmp = np.zeros((nbFrames, height, width))
        #offset_x,offset_y, width, height = 604,204,176,640
        cam.connectCam(14, offset_x,offset_y, width, height)
        nitFrame =  cam.captureFrames(nbFrames)
        #time.sleep(0.4)
        cam.disconnect()
        for frame_idx in range(nbFrames):
            imgData_tmp[frame_idx,:,:] = nitFrame[frame_idx]
        image = np.mean(imgData_tmp, axis = 0)
        return image

if __name__=="__main__":
    cam = NITCam()
    nbFrames = 10
    nbMasks = 5
    offset_x, offset_y, width, height = 604, 204 ,176 ,640
    #eng = matlab.engine.start_matlab()
    #eng.wsmask1(matlab.double(vec_in))
    imgData = np.zeros((nbMasks, height, width))
    fname = 'C:\\Users\\NTTRi LAB\\Desktop\\MVM\\mvm128\\randresults_calib'
    for Mdx in range(nbMasks):
        #r = eng.slmmasks_path(matlab.double(Mdx+1))
        time.sleep(0.1)
        image = CaptureNFramesAvg(cam, offset_x, offset_y, width, height, nbFrames)
        imgData[Mdx,:,:] = image
        #vec_in_real[Mdx,:] = eng.wsmask1(matlab.double(vec_in))
        savemat(fname +'.mat', {'imgData': imgData})