from .device import Device
from decimal import Decimal
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import serial

import socket
import time

# Adapted from  https://github.com/CrinitusFeles/HMP4040/blob/master/hmp4040/HMP4040.py
class RsHMP4040(object):
    def __init__(self, ip='192.168.1.8', port="5025"):
        self.Debug = True
        self.IP = ip
        self.Port = port
        self.client = socket.socket()
        self.connectionStatus = False
        # if self.connect(ip, port):
        #     self.connectionStatus = True
        # else:
        #     self.connectionStatus = False


        # Create dummy device for logging
        self.logger = Device(addr='', name='HMP4040 Power Supply', isVISA=False)
        self.debug = self.logger.debug
        self.info = self.logger.info
        self.warning = self.logger.warning
        self.error = self.logger.error
        self.critical = self.logger.critical
        self.devicename = self.logger.devicename


    def __str__(self):
        return "LBS HMP4040 connection = " + str(self.connectionStatus) + \
               "\nIP: %s" % self.IP + "\nPort: " + str(self.Port)

    def __repr__(self):
        return str(self)

    def get_ip(self):
        return self.IP

    def get_port(self):
        return self.Port

    def connect(self, ip='192.168.1.8', port="5025"):
        """
        Establishes a connection to the power supply.
        :param ip: IP address of the power supply.
        :param port: Port of the power supply.
        :return: Returns True if the connection is successful, otherwise False.
        """
        self.IP = ip
        self.Port = port
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client.settimeout(1)
        self._myprint("trying to connect to " + str(ip) + " : %d" % int(port))
        try:
            self.client.connect((ip, int(port)))
        except socket.error as exc:
            self._myprint('Caught exception socket.error : %s' % exc)
            self.connectionStatus = False
            return False
        else:
            self._myprint('connected to server ' + ip + ':' + str(port))
            self.connectionStatus = True
            return True

    def reconnect(self):
        self.disconnect()
        return self.connect(self.get_ip(), self.get_port())

    def disconnect(self):
        self.client.close()
        self.connectionStatus = False
        self._myprint('disconnected')

    def _myprint(self, text):
        if self.Debug:
            print(text)
            self.info(text)

    def __check_voltage(self, voltage="1.0"):
        """
        Checks if the specified voltage is valid for this power supply.
        :param voltage: Voltage for the channel.
        :return: Returns True if the voltage is valid, otherwise False.
        """
        try:
            float(voltage)
            if 0 <= float(voltage) <= 30:
                return True
            else:
                return False
        except ValueError:
            return False

    def __check_current(self, current="0.1"):
        """
        Checks if the specified current is valid for this power supply.
        :param current: Current for the channel.
        :return: Returns True if the current is valid, otherwise False.
        """
        try:
            float(current)
            if 0 <= float(current) <= 10:
                return True
            else:
                return False
        except ValueError:
            return False

    def __check_channel(self, channels_=('1', '2', '3', '4')):
        """
        Checks if the provided channels are valid (between 1 and 4).
        :param channels_: Tuple of channel numbers.
        :return: Returns False if any channel is invalid; otherwise True.
        """
        for i in channels_:
            try:
                int(i)
                if not (1 <= int(i) <= 4):
                    return False
            except ValueError:
                return False
        return True

    def __check_fuse_delay(self, delay=50):
        """
        Checks if the fuse delay is within acceptable limits.
        :param delay: Delay in milliseconds.
        :return: Returns True if the delay is valid, otherwise False.
        """
        try:
            int(delay)
            if 0 <= delay <= 250:
                return True
        except ValueError:
            return False
        
    def __for_each_channel(self, *__check_functions, ch=('1', '2', '3', '4'), cmd=''):
        """
        A universal function to reduce code duplication. It validates input data, and if invalid, returns False.
        If all data is valid, it checks the command type.
        Regular commands return True, while data query commands return a list of data for all specified channels.
        :param __check_functions: List of validation functions for input data.
        :param ch: List of channels.
        :param cmd: Command sent to the specified channels.
        :return: Returns True for valid input data, otherwise False. For data query commands, returns a list of data.
        """
        # Input data validation block (start)
        for func in __check_functions:
            if func:
                continue
            else:
                return False
        # Input data validation block (end)

        # Command type check. Data query commands return a list. Regular commands return True if inputs are valid.
        if cmd.find('?') != -1:
            received_data = []
            for i in ch:
                self.send_command("INST OUT" + str(i))
                received_data.append(self.send_command(cmd))
            return received_data
        elif cmd == '':
            return False
        else:
            for i in ch:
                self.send_command('INST OUT' + str(i))
                self.send_command(cmd)
            return True

    def send_command(self, cmd):
        """
        Sends a command to the power supply. If a response is expected, the program pauses until the response is received.
        :param cmd: Command to send to the power supply.
        :return: If a response is expected, returns the response string; otherwise, returns None.
        """
        self._myprint('client send: ' + cmd)
        if cmd.find('?') != -1:
            self.client.send((cmd + '\n').encode())
            received_data = self.client.recv(1024).decode()
            self._myprint("received: " + received_data)
            return received_data
        else:
            self.client.send((cmd + '\n').encode())
            time.sleep(0.05)
            return None

    def set_voltage(self, channels_=('1', '2', '3', '4'), voltage="0.0"):
        """
        Sets the specified voltage on the selected channels.
        :param channels_: Tuple of selected channels.
        :param voltage: Voltage level for the selected channels.
        :return: Returns True if no errors occur. Returns False if an invalid channel is specified.
        """
        return self.__for_each_channel((self.__check_channel(channels_), self.__check_voltage(voltage)),
                                       ch=channels_, cmd="VOLT " + str(voltage))

    def get_voltage(self, channels_=('1', '2', '3', '4')):
        """
        Queries the current voltage parameters on the specified channels from the power supply.
        :param channels_: List of channels to measure voltage.
        :return: Returns a tuple of voltage values for each specified channel if arguments are valid; otherwise, False.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd="VOLT?")

    def set_current(self, channels_=('1', '2', '3', '4'), current="0.1"):
        """
        Sets the current on the selected channels.
        :param channels_: List of channels where the current value will be set.
        :param current: Current value.
        :return: Returns True if inputs are valid; otherwise, False.
        """
        return self.__for_each_channel((self.__check_channel(channels_), self.__check_current(current)),
                                       ch=channels_, cmd="CURR " + str(current))

    def get_current(self, channels_=('1', '2', '3', '4')):
        """
        Queries the current on the selected channels.
        :param channels_: List of channels.
        :return: Returns a tuple of current values for each specified channel if inputs are valid; otherwise, False.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd="CURR?")

    def get_status_byte(self):
        return self.send_command('*STB?')

    def get_event_status(self):
        return self.send_command('*ESR?')

    def check_sound(self):
        self.send_command('SYST:BEEP')

    def get_version(self):
        """
        :return: Returns the SCPI version number.
        """
        return self.send_command('SYST:VERS?')

    def get_errors(self):
        """
        :return: Returns the error code.
                 0 - no errors;
                 -100 - command error;
                 -102 - syntax error;
                 -350 - queue overflow.
        """
        return self.send_command('SYST:ERR?')

    def get_identification_info(self):
        """
        :return: Returns the device type, serial number, and firmware version.
        """
        return self.send_command('*IDN?')

    def get_last_channel(self):
        """
        :return: Returns the active channel number as a string "OUTx", where x is the channel number.
        """
        return self.send_command('INST?')

    def set_step_voltage(self, step_="1.0"):  # To be modified for channel list.
        """
        Sets the step size for voltage. Default is 1.000. Valid values range from 0.000 to 32.050.
        :param step_: Voltage step size.
        :return: Returns True if arguments are valid; otherwise, False.
        """
        if self.__check_voltage(step_):
            self.send_command('VOLT:STEP ' + str(step_))
            return True
        else:
            return False

    def get_step_voltage(self):
        """
        :return: Returns the voltage step size.
        """
        return self.send_command('VOLT:STEP?')

    def voltage_up(self, channels_=('1', '2', '3', '4')):
        """
        Increases the voltage on the selected channels by the step size specified in the previous command
        or by the default value.
        :param channels_: Channels on which the voltage will be adjusted.
        :return: Returns True if the arguments are valid, otherwise False.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='VOLT UP')

    def voltage_down(self, channels_=('1', '2', '3', '4')):
        """
        Decreases the voltage on the selected channels by the step size set by set_step_voltage(step).
        :param channels_: Channels where the voltage will be decreased.
        :return: Returns True if the arguments are valid, otherwise False.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='VOLT DOWN')

    def set_step_current(self, step="0.1"):
        """
        Checks the specified step value for current to ensure it is within the acceptable range. 
        If valid, sets the step value. Otherwise, returns False.
        :param step: Step value for current.
        :return: Returns True if arguments are valid, otherwise False.
        """
        if self.__check_current(step):
            self.send_command('CURR:STEP ' + str(step))
            return True
        else:
            return False

    def get_step_current(self):
        """
        :return: Returns the current step value.
        """
        return self.send_command('CURR:STEP?')

    def current_up(self, channels_=('1', '2', '3', '4')):
        """
        Increases the current on the selected channels by the step value.
        :return: Returns True if the arguments are valid, otherwise False.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='CURR UP')

    def current_down(self, channels_=('1', '2', '3', '4')):
        """
        Decreases the current on the selected channels.
        :param channels_: List of channels.
        :return: Returns True if the arguments are valid, otherwise False.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='CURR DOWN')

    def set_channel_params(self, channels_=('1', '2', '3', '4'), voltage="1.0", current="1.0"):
        """
        Sets the specified voltage and current values on the selected list of channels.
        :param channels_: List of channels.
        :param voltage: Voltage level.
        :param current: Current level.
        :return: Returns True if the arguments are valid, otherwise False.
        """
        return self.__for_each_channel(
            (self.__check_channel(channels_), self.__check_voltage(voltage), self.__check_current(current)),
            ch=channels_, cmd='APPL ' + str(voltage) + ',' + str(current))

    def get_channel_params(self, channels_=('1', '2', '3', '4')):
        """
        Retrieves the current voltage and current values for the selected list of channels.
        :param channels_: List of channels.
        :return: Returns a list of voltage and current data for each specified channel if arguments are valid,
                 otherwise False.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='APPL?')

    def is_output_turned_on(self):
        """
        Checks if the output is enabled.
        :return: Returns the status of the output.
        """
        return self.send_command('OUTP:GEN?')

    def turn_on_selected_channels(self):
        """
        Enables the selected channels for physical output from the power supply.
        :return: Returns True.
        """
        return self.send_command('OUTP:GEN 1')

    def turn_off_selected_channels(self):
        """
        Disables all channels from the physical output of the power supply.
        :return: Returns True.
        """
        return self.send_command('OUTP:GEN 0')

    def select_on_channel(self, channels_=('1', '2', '3', '4')):
        """
        Turns on the selected channels.
        :param channels_: List of channels.
        :return: Returns True if the arguments are valid, otherwise False.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='OUTP:SEL 1')

    def select_off_channel(self, channels_=('1', '2', '3', '4')):
        """
        Turns off the selected channels.
        :param channels_: List of channels.
        :return: Returns True if the arguments are valid, otherwise False.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='OUTP:SEL 0')

    def get_active_channel(self, channels_=('1', '2', '3', '4')):
        """
        Retrieves the status of each channel as a list, where 1 means the channel is active and 0 means inactive.
        :param channels_: List of channels.
        :return: Returns True if the arguments are valid, otherwise False.
        """
        return self.__for_each_channel(self.__check_channel(channels_), cmd='OUTP?')

    def set_overvoltage_protection_value(self, channels_=('1', '2', '3', '4'), max_voltage="10.0"):
        """
        Sets the overvoltage protection threshold for each channel in the list.
        :param channels_: List of channels.
        :param max_voltage: Overvoltage protection threshold.
        :return: Returns True if the arguments are valid, otherwise False.
        """
        return self.__for_each_channel((self.__check_channel(channels_), self.__check_voltage(max_voltage)),
                                       ch=channels_, cmd='VOLT:PROT ' + str(max_voltage))

    def get_overvoltage_protection_value(self, channels_=('1', '2', '3', '4')):
        """
        :param channels_: List of channels.
        :return: Returns a list of overvoltage protection threshold values for each channel.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='VOLT:PROT?')

    def clear_overvoltage_protection(self, channels_=('1', '2', '3', '4')):
        """
        A utility function that removes the flashing "ovp" message. To disable the protection, reduce the voltage
        to an acceptable level and re-enable the disconnected power supply output.
        :param channels_: List of channels.
        :return: Returns a list of channel states if the arguments are valid, otherwise False.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='VOLT:PROT:CLE')

    def get_overvoltage_channels_tripped(self, channels_=('1', '2', '3', '4')):
        """
        Checks the specified channels to see if overvoltage protection has been triggered.
        :param channels_: List of channels.
        :return: Returns a list of channels where overvoltage protection was triggered, if the arguments are valid.
        """
        tripped_channels = []
        if self.__check_channel(channels_):
            for i in channels_:
                self.send_command('INST OUT' + str(i))
                if int(self.send_command('VOLT:PROT:TRIP?')):
                    tripped_channels.append(i)
            return tripped_channels
        else:
            return False

    def is_overvoltege_channel_tripped(self, channels_=('1', '2', '3', '4')):
        """
        Checks the specified channels to see if overvoltage protection has been triggered.
        :param channels_: List of channels.
        :return: Returns a list containing the states of overvoltage protection for the channels.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='VOLT:PROT:TRIP?')

    def meas_overvoltage_protection(self, channels_=('1', '2', '3', '4')):
        """
        Enables overvoltage protection in MEAS mode. After setting the trigger level below the maximum value 
        and enabling channel protection, protection will always activate if voltage exceeds the channel limit,
        disconnecting the channel from the output. To avoid triggering, increase the trigger threshold.
        :param channels_: List of channels.
        :return: Returns a list containing the states of overvoltage protection for the channels if the data is valid.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='VOLT:PROT:MODE MEAS')

    def is_overvoltage_protection_active(self, channels_=('1', '2', '3', '4')):
        """
        :param channels_: List of channels.
        :return: Returns the protection mode for the specified channels.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='VOLT:PROT:MODE MEAS?')

    def measure_voltage(self, channels_=('1', '2', '3', '4')):
        """
        Internal voltage measurement.
        :param channels_: List of channels.
        :return: Returns a list of voltage values for each channel.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='MEAS:VOLT?')

    def reset_hmp4040(self):
        """
        Resets the device.
        :return: Returns the reset response.
        """
        return self.send_command("*RST")

    def measure_current(self, channels_=('1', '2', '3', '4')):
        """
        Internal current measurement.
        :param channels_: List of channels.
        :return: Returns a list of current values for each channel.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='MEAS:CURR?')

    def set_fuse_delay(self, channels_=('1', '2', '3', '4'), delay_=10):
        """
        Sets the fuse activation delay time for the specified channels.
        :param channels_: List of channels where the function will apply.
        :param delay_: Delay time in milliseconds.
        :return: Returns a list containing the states of the fuse protection for the channels if the data is valid.
        """
        return self.__for_each_channel((self.__check_channel(channels_), self.__check_fuse_delay(delay_)),
                                       ch=channels_, cmd='FUSE:DEL ' + str(delay_))

    def get_fuse_delay(self, channels_=('1', '2', '3', '4')):
        """
        :param channels_: List of channels.
        :return: Returns a list containing the fuse delay time for each specified channel.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='FUSE:DEL?')

    def set_link_fuse(self, source_channel, channels_=('1', '2', '3', '4')):
        """
        Links the specified source channel with all channels in the list, so that if protection triggers on 
        the source channel, the linked channels are also disconnected.
        :param source_channel: Channel to link others with.
        :param channels_: List of channels to link with the source channel.
        :return: Returns True if the data is valid, otherwise False.
        """
        if self.__check_channel([source_channel]) and self.__check_channel(channels_):
            self.send_command('INST OUT' + str(source_channel))
            for i in channels_:
                if source_channel != i:
                    self.send_command('FUSE:LINK ' + str(i))
            return True
        else:
            return False

    def get_link_fuse(self, source_channel, channels_=('1', '2', '3', '4')):
        """
        :param source_channel: Source channel.
        :param channels_: List of channels.
        :return: Returns a list of channels linked to the source channel. Returns False if the input data is invalid.
        """
        if self.__check_channel([source_channel]) and self.__check_channel(channels_):
            self.send_command('INST OUT' + str(source_channel))
            linked_channels = []
            for i in channels_:
                if source_channel != i:
                    linked_channels.append(self.send_command('FUSE:LINK? ' + str(i)))
            return linked_channels
        else:
            return False

    def unlink_fuse(self, source_channel, channels_=('1', '2', '3', '4')):
        """
        Unlinks the specified channels from the source channel.
        :param source_channel: Source channel.
        :param channels_: List of channels.
        :return: Returns True if the input data is valid, otherwise False.
        """
        if self.__check_channel([source_channel]) and self.__check_channel(channels_):
            self.send_command('INST OUT' + str(source_channel))
            for i in channels_:
                if source_channel != i:
                    self.send_command('FUSE:UNL ' + str(i))
            return True
        else:
            return False

    def get_fuse_channels_tripped(self, channels_=('1', '2', '3', '4')):
        """
        Checks the specified channels to see if fuse protection has been triggered.
        :param channels_: List of channels.
        :return: Returns a list of channels where fuse protection was triggered, if the input data is valid.
        """
        tripped_channels = []
        if self.__check_channel(channels_):
            for i in channels_:
                self.send_command('INST OUT' + str(i))
                if int(self.send_command('FUSE:TRIP?')):
                    tripped_channels.append(i)
            return tripped_channels
        else:
            return False

    def set_on_fuse_channels(self, channels_=('1', '2', '3', '4')):
        """
        Enables fuse protection on the specified channels.
        :param channels_: List of channels.
        :return: Returns True if the input data is valid, otherwise False.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='FUSE 1')

    def set_off_fuse_channels(self, channels_=('1', '2', '3', '4')):
        """
        Disables fuse protection on the specified channels.
        :param channels_: List of channels.
        :return: Returns True if the input data is valid, otherwise False.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='FUSE 0')

    def clear_arbitrary_data(self, channels_=('1', '2', '3', '4')):
        """
        Clears arbitrary data on the specified channels.
        :param channels_: List of channels.
        :return: Returns True if successful.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='ARB:CLEAR 1')

    def set_arbitrary_sequence(self, channels_=('1', '2', '3', '4'), sequence=""):
        """
        Sets an arbitrary sequence on the specified channels.
        :param channels_: List of channels.
        :param sequence: The arbitrary sequence to set.
        :return: Returns True if successful.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='ARB:DATA ' + str(sequence))

    def set_arbitrary_sequence_repeat(self, channels_=('1', '2', '3', '4'), repeat="1"):
        """
        Sets the repeat count for an arbitrary sequence on the specified channels.
        :param channels_: List of channels.
        :param repeat: The repeat count.
        :return: Returns True if successful.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='ARB:REP ' + str(repeat))

    def start_arbitrary_sequence(self, channels_=('1', '2', '3', '4')):
        """
        Starts an arbitrary sequence on the specified channels.
        :param channels_: List of channels.
        :return: Returns True if successful.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='ARB:STAR 1')

    def stop_arbitrary_sequence(self, channels_=('1', '2', '3', '4')):
        """
        Stops an arbitrary sequence on the specified channels.
        :param channels_: List of channels.
        :return: Returns True if successful.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='ARB:STOP 1')

    def transfer_arbitrary(self, channels_=('1', '2', '3', '4')):
        """
        Transfers arbitrary data to the specified channels.
        :param channels_: List of channels.
        :return: Returns True if successful.
        """
        return self.__for_each_channel(self.__check_channel(channels_), ch=channels_, cmd='ARB:TRAN 1')
    

if __name__ == "__main__":
    LBS = HMP4040()
    print(LBS)
    if LBS.connect(ip="10.6.1.4", port='5025'):
        #LBS.check_sound()
        LBS.get_version()
        print(LBS.get_identification_info())
        LBS.get_errors()
        print(LBS.get_status_byte())

        channels = ['1', '2', '3', '4']
        # LBS.set_voltage(channels, 3.1)
       #  LBS.set_current(channels, 1.5)
       #  LBS.turn_off_selected_channels()
       #  LBS.select_off_channel(channels)
        # LBS.select_on_channel(channels)
        # LBS.turn_on_selected_channels()
        # print(LBS.set_overvoltage_protection_value(['1', '2', '3'], 5.1))
        # LBS.meas_overvoltage_protection(['1'])
        # LBS.set_voltage(['1', '3'], 5.3)
       #  LBS.is_overvoltage_protection_active(['1'])
       #  print(LBS.get_overvoltage_channels_tripped(channels))
        # LBS.clear_overvoltage_protection(channels)
        # LBS.voltage_up(channels)
        # LBS.set_on_fuse_channels(channels)
        # LBS.set_link_fuse(1, channels)
        # LBS.set_link_fuse(2, ['3', '4'])
        # print(LBS.get_link_fuse(1))
        # print(LBS.get_link_fuse(2))
        # LBS.set_off_fuse_channels(channels)
        # print(LBS.get_active_channel(channels))
        # LBS.currentDown()
        while True:
            req = input('enter some text: ')
            if req == 'exit':
                break
            LBS.send_command(req)
        LBS.disconnect()


# class RsHMP4040(Device):
#     def __init__(self, addr: str = 'TCPIP::192.168.1.8::INSTR', 
#                  name: str = 'Rohde & Schwarz HMP4040', **kwargs):
#         super().__init__(addr=addr, name=name, isVISA=True, **kwargs)
#         self.channel_list = [1, 2, 3, 4]

#     def connect(self):
#         super().connect()
#         self.hmp4040 = self.inst #pyvisa_instr # this is the pyvisa instrument, rm.open_resource('ASRL6::INSTR')
        

#     def get_inst_state(self):
#         all_scpi_list = []
#         for channel in self.channel_list:
#             self.hmp4040.write('INSTrument:NSELect {0}'.format(channel))
#             voltage          = float(self.hmp4040.query('SOURce:VOLTage?'))
#             current_lim      = float(self.hmp4040.query('SOURce:CURRent?'))
#             status           = int(self.hmp4040.query('OUTPut:STATe?'))
#             current          = float(self.hmp4040.query('MEASure:CURRent?'))
#             volt_prot        = float(self.hmp4040.query('VOLTage:PROTection?'))
#             volt_prot_active = self.hmp4040.query('VOLTage:PROTection:MODE?').rstrip('\r\n')
#             print("Channel %d voltage is %.02f V, overvoltage limit is %.02f overvolt config is %s, current limit is %.02f A, current is %.03f A status is %s" % (channel,voltage,volt_prot,volt_prot_active,current_lim,current,status))

#     def get_channel_scpi_list(self, channel=1):
#         result_list = []
#         self.hmp4040.write('INSTrument:NSELect {0}'.format(channel))
#         result_list.append('INSTrument:NSELect {0}'.format(channel))
#         for command in self.cmd_dict:
#             result = (self.hmp4040.query(command.format("?"))).rstrip('\r\n')
#             result = " " + result
#             result_list.append(command.format(result))
#             time.sleep(0.1)
#         return result_list

#     def get_unique_scpi_list(self):
#         unique_scpi_list = []
#         unique_scpi_channel_list = []
#         for channel in self.channel_list:
#             channel_settings_list = self.get_channel_scpi_list(channel)
#             found = 0
#             unique_scpi_channel_list = []
#             for setting in channel_settings_list:
#                 if (setting not in self.por_scpi_list):
#                     found = 1
#                     unique_scpi_channel_list.append(setting)
#             if (found == 1):
#                 unique_scpi_list.append(channel_settings_list[0]) # this is the channel selection
#                 unique_scpi_list.extend(unique_scpi_channel_list)
#         return unique_scpi_list


#     select_cmd_dict = {
#         "INSTrument:NSELect {0}": "Selects a channel by number" }

#     activate_cmd_dict = {
#         "OUTPut:GENeral {0}": "Enables the outputs for all activated channels",
#         "OUTPut:SELect {0}": "Activites a selected channel" }

#     cmd_dict = {
#         "SOURce:VOLTage{0}"          : "Sets/Queries the voltage value of the selected channel",
#         "SOURce:CURRent{0}"          : "Sets/Queries the current limit value of the selected channel",
#         "VOLTage:PROTection:MODE{0}" : "Sets/Queries the voltage protection mode, measured or protected",
#         "OUTPut:STATe{0}"            : "Activates and enables the output for a selected channel"
#     }

#     query_cmd_dict = {
#         "MEASure:VOLTage?" : "Measures voltage of selected channel",
#         "MEASure:CURRent?" : "Measures voltage of selected channel"
#     }

#     por_scpi_list = [
#         'INSTrument:NSELect 1',
#         'INSTrument:NSELect 2',
#         'INSTrument:NSELect 3',
#         'INSTrument:NSELect 4',
#         'SOURce:CURRent 0.1000',
#         'SOURce:VOLTage 0.000',
#         'VOLTage:PROTection:MODE measured',
#         'OUTPut:STATe 0' ]
    