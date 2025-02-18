from .device import Device
from decimal import Decimal
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import savemat
from astropy.io import fits
import time

import toptica.lasersdk as lasersdk
import toptica.lasersdk.dlcpro.v2_4_0 as dlcsdk
from toptica.lasersdk import decop, client


class TopticaLaser(Device, dlcsdk.DLCpro):
    def __init__(self, addr="192.168.1.18", name = "toptica laser", **kwargs):
        super().__init__(addr=addr, name=name, isVISA=False, **kwargs)
        

        connection = dlcsdk.NetworkConnection(addr)
        dlcsdk.DLCpro.__init__(self, connection)

        self.connection = connection
        self.client = client.Client(self.connection)
        # self.dlc = dlcsdk.DLCpro(self.connection)
    
    def connect(self):
        self.open()
        self.connected = True
        self.info("toptica laser connected")

    def disconnect(self):
        if self.connected:
            self.close()
            self.connected = False
            self.info("toptica laser disconnected")
    
    def get_time_avged_value(self, avg_time = 5, interval = 0.1,
                             get_func = None):
        '''
        Get the time-averaged value of the laser over avg_time seconds.
        '''
        if get_func is None:
            get_func = self.laser1.ctl.wavelength_act.get
        t_start = time.time()
        values = []
        while time.time() - t_start < avg_time:
            values.append(get_func())
            if not np.isclose(interval, 0):
                time.sleep(interval)
        return np.mean(values)

        