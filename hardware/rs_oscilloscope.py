
# Rode Schwarz Oscilloscope RTA4004

from hardware.device import Device
from RsInstrument import *
import time
import matplotlib.pyplot as plt
import numpy as np

class RsOSC(Device):
    def __init__(self, addr='TCPIP::192.168.1.4::INSTR',
                       name='Rode Schwarz OSC RTA4004', **kwargs):
        super().__init__(addr=addr, name=name, isVISA=False, **kwargs)
        self.last_measured_mean = None
    
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
        result = sum(data)/len(data)
        self.last_measured_mean = result
        return result
    
    def get_time_avged_voltage(self, channel: int, time: float):
        import time as t
        import numpy as np
        start = t.time()
        data = []
        while t.time()-start < time:
            # data.append(self.get_trace(channel, plot=False))
            data.append(self.get_avg_voltage(channel))
        data = np.array(data).flatten()
        return data.mean(), data.std() / np.sqrt(len(data))
    
    def get_vertical_scale_volt(self, channel: int):
        return float(self.query(f"CHANnel{channel}:SCALe?"))
    
    def set_vertical_scale_volt(self, channel: int, scale: float):
        if scale < 0.0005:
            self.warning("Vertical scale is too small. Setting to 0.5 mV / div")
            scale = 0.0005
        self.write(f"CHANnel{channel}:SCALe {scale:.4f}")
        self.info(f"Vertical scale of channel {channel} set to {scale:.4f} V/div")

    def update_vertical_scale_by_last_measurement(self, channel: int = 1, mean_scale_to = 0.4):
        if mean_scale_to < 0 or mean_scale_to > 1:
            info = f"Given mean_scale_to {mean_scale_to} should be between 0 and 1, representing the fraction of displayed position of the last measured mean voltage"
            self.error(info)
            raise ValueError(info)
        
        if self.last_measured_mean is None:
            self.warning("No last measurement found. Reading mean voltage from channel {channel}")
            self.last_measured_mean = self.get_avg_voltage(channel)
        
        pos_div = 4.9
        if self.last_measured_mean >= 0:
            self.set_vertical_position_div(channel, -pos_div)
        else:
            self.set_vertical_position_div(channel, pos_div)

        if np.isclose(np.abs(self.last_measured_mean), (5+pos_div)*self.get_vertical_scale_volt(channel)) or np.abs(self.last_measured_mean) > (5+pos_div)*self.get_vertical_scale_volt(channel):
            self.warning("Mean voltage is already at the edge of the screen. Setting vertical scale again.")
            num_tries = 0
            max_tries = 10
            while np.isclose(np.abs(self.last_measured_mean), (5+pos_div)*self.get_vertical_scale_volt(channel)) and num_tries < max_tries:
                self.set_vertical_scale_volt(channel, self.last_measured_mean / 2)
                self.last_measured_mean = self.get_avg_voltage(channel)
                num_tries += 1
            if num_tries == max_tries:
                self.error("Failed to set vertical scale. Please check manually.")
                return

        scale = np.abs(self.last_measured_mean / mean_scale_to / 10)
        self.set_vertical_scale_volt(channel, scale)

        

    def get_time_scale_sec(self):
        return float(self.query("TIMebase:SCALe?"))

    def set_time_scale_sec(self, scale: float):
        self.write(f"TIMebase:SCALe {scale:.4e}")
        self.info(f"Time scale set to {scale:.4e} s/div")

    def get_vertical_position_div(self, channel: int):
        return float(self.query(f"CHANnel{channel}:POSition?"))

    def set_vertical_position_div(self, channel: int, position: float):
        self.write(f"CHANnel{channel}:POSition {position:.4f}")
        self.info(f"Vertical position of channel {channel} set to {position:.4f} div")