import os, sys, warnings, inspect

# ------------ Logger start ------------
from .utils.logger import LoggerClient, get_call_kwargs
logger = LoggerClient()
# ------------ Logger end ------------


# Base class for devices
class Device:
    import pyvisa
    rm = pyvisa.ResourceManager()

    def __init__(self, addr, name='', isVISA=True):
        self.addr = addr
        self.devicename = name
        self.isVISA = isVISA
        self.connected = False
        if isVISA:
            try:
                self.inst = self.rm.open_resource(addr)
            except Exception as e:
                self.error(self.devicename + f": Error init device instance:{e}")
                self.connected = False

    def connect(self):
        if not self.connected:
            try:
                if self.isVISA:
                    self.inst.open()
                self.connected = True
                self.info(self.devicename + " connected")
                return 1
            except Exception as e:
                self.error(self.devicename + f": Error connecting device:{e}")
                return -1
        return 0

    def disconnect(self):
        if self.connected:
            if self.isVISA:
                self.inst.close()
            self.connected = False
            self.info(self.devicename + " disconnected")
            return 1
        return 0

    def write(self, cmd):
        self.inst.write(cmd)

    def read(self):
        return self.inst.read()

    def query(self, cmd):
        return self.inst.query(cmd)

    # logging
    logger = logger
    
    def debug(self, x, name='', level=1, **kwargs):
        try:
            logger.debug(x, devicename=name or self.devicename, **get_call_kwargs(level), **kwargs)
        except Exception as e:
            print(x)
            print("Logger error:", e)
        # print(x)

    def info(self, x, name='', level=1, **kwargs):
        try:
            logger.info(x, devicename=name or self.devicename, **get_call_kwargs(level), **kwargs)
        except Exception as e:
            print(x)
            print("Logger error:", e)
        

    def warning(self, x, name='', level=1, **kwargs):
        try:
            logger.warning(x, devicename=name or self.devicename, **get_call_kwargs(level), **kwargs)
            warnings.warn(x)
        except Exception as e:
            print(x)
            print("Logger error:", e)
        # logger.warning(x, devicename=name or self.devicename, **get_call_kwargs(level))
        # warnings.warn(x)

    def error(self, x, name='', level=1, **kwargs):
        # logger.error(x, devicename=name or self.devicename, **get_call_kwargs(level))
        # # raise InstrExceptionByNTT(name + ': ' + x)
        try:
            logger.error(x, devicename=name or self.devicename, **get_call_kwargs(level), **kwargs)
        except Exception as e:
            print(x)
            print("Logger error:", e)

    def critical(self, x, name='', level=1, **kwargs):
        # logger.critical(x, devicename=name or self.devicename, **get_call_kwargs(level))
        # # raise InstrExceptionByNTT(name + ': ' + x)
        try:
            logger.critical(x, devicename=name or self.devicename, **get_call_kwargs(level), **kwargs)
        except Exception as e:
            print(x)
            print("Logger error:", e)


class InstrExceptionByNTT(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
