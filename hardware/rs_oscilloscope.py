
# Rode Schwarz Oscilloscope RTA4004

from hardware.device import Device
from RsInstrument import *
import time
import matplotlib.pyplot as plt

class RsOSC(Device):
    def __init__(self, addr='TCPIP::192.168.1.25::INSTR',
                       name='Rode Schwarz OSC RTA4004', **kwargs):
        super().__init__(addr=addr, name=name, isVISA=False, **kwargs)
    
    def connect(self):
        self.inst = RsInstrument(self.addr, 
                                 True,  # id_query to True (default is True) checks, whether your instrument can be used with the RsInstrument module
                                 False  # reset to False Do NOT resets the instrument to factory settings 
                                 )
        self.connected = True
        self.info("Rode Schwarz Oscilloscope connected")
    
    def disconnect(self):
        if self.connected:
            self.inst.close()
            self.connected = False
            self.info("Rode Schwarz Oscilloscope disconnected")
        
    def query(self, cmd: str):
        return self.inst.query_str(cmd)
    
    def write(self, cmd: str):
        self.inst.write_str(cmd)
    
    @property
    def IDN(self):
        return self.query("*IDN?")
    
    @property
    def time_center(self):
        return self.query("TIMebase:POSition?")

    @property
    def time_range(self):
        return self.query("TIMebase:RANGe?")
    
    def get_trace(self, channel: int, plot: bool = True):
        start = time.time()
        self.write("FORMat:DATA REAL,32")
        self.inst.bin_float_numbers_format = BinFloatFormat.Single_4bytes_swapped
        self.inst.data_chunk_size = 100000
        data_bin = self.inst.query_bin_or_ascii_float_list(f"CHANnel{channel}:DATA?")
        self.info(f"Binary waveform transfer elapsed time: {time.time()-start} seconds.")
        if plot:
            plt.plot(data_bin)
            plt.xlabel('Sample')
            plt.ylabel('Voltage')
            plt.title(f'Channel {channel} waveform')
            plt.show()
        
        return data_bin

    def get_avg_voltage(self, channel: int):
        data = self.get_trace(channel, plot=False)
        return sum(data)/len(data)