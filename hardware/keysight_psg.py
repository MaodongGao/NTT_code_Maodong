from .device import Device
from decimal import Decimal
import numpy as np
import os
import matplotlib.pyplot as plt

import time

class KeysightPSG(Device):
    def __init__(
        self,
        addr="TCPIP::192.168.1.14::INSTR",
        name="Keysight Function Generator",
        isVISA=True,
    ):
        super().__init__(addr=addr, name=name, isVISA=isVISA)

    def get_freq_hz(self):
        return float(self.query("FREQ:CW?"))
    
    def set_freq_hz(self, freq):
        self.write(f"FREQ:CW {freq}")
        self.info(f"Set frequency to {freq} Hz")

    def get_power_dbm(self):
        return float(self.query("POW:AMPL?"))
    
    def set_power_dbm(self, power):
        self.write(f"POW:AMPL {power}")
        self.info(f"Set power to {power} dBm")

    def get_output(self):
        return self.query("OUTP:STAT?")
    
    def set_output(self, state):
        self.write(f"OUTP:STAT {state}")
        self.info(f"Set output to {state}")

    def get_modulation_state(self):
        return self.query("OUTP:MOD:STAT?")
    
    def set_modulation_state(self, state):
        self.write(f"OUTP:MOD:STAT {state}")
        self.info(f"Set modulation state to {state}")