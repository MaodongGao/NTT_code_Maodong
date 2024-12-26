from .device import Device
from decimal import Decimal
import numpy as np
import os
import matplotlib.pyplot as plt
import time
from thorlabs_elliptec import ELLx, ELLError, ELLStatus, list_devices

class Elliptec(Device):
    def __init__(self, addr: str = 'COM7', name: str = 'Thorlabs Elliptec Motor', **kwargs):
        super().__init__(addr=addr, name=name, isVISA=False, **kwargs)
        # use print(list_devices()) to get the vid and pid of the device
        self.connected = False

    def connect(self):
        self.inst = ELLx(serial_port=self.addr)
        self.connected = True
        self.info("Elliptec connected")

    def disconnect(self):
        if self.connected:
            self.inst.close()
            self.connected = False
            self.info("Elliptec disconnected")

    # The following methods are implemented in the thorlabs_elliptec module
    # API documentation: https://thorlabs-elliptec.readthedocs.io/en/latest/api/thorlabs_elliptec.html
    def get_position(self):
        """
        Get the current position of the motor in real device units.
        """
        if self.connected:
            return self.inst.get_position()
        else:
            self.error("Device not connected")

    def get_position_raw(self):
        """
        Get the current position of the motor in raw encoder counts.
        """
        if self.connected:
            return self.inst.get_position_raw()
        else:
            self.error("Device not connected")

    def home(self, direction=0, blocking=True):
        """
        Move the motor to the home position.
        """
        if self.connected:
            self.inst.home(direction=direction, blocking=blocking)
        else:
            self.error("Device not connected")

    def move_absolute(self, position, blocking=True):
        """
        Move the motor to an absolute position in real device units.
        """
        if self.connected:
            self.inst.move_absolute(position, blocking=blocking)
        else:
            self.error("Device not connected")

    def move_absolute_raw(self, counts, blocking=True):
        """
        Move the motor to an absolute position in raw encoder counts.
        """
        if self.connected:
            self.inst.move_absolute_raw(counts, blocking=blocking)
        else:
            self.error("Device not connected")

    def move_relative(self, amount, blocking=True):
        """
        Move the motor by a relative amount in real device units.
        """
        if self.connected:
            self.inst.move_relative(amount, blocking=blocking)
        else:
            self.error("Device not connected")

    def move_relative_raw(self, counts, blocking=True):
        """
        Move the motor by a relative amount in raw encoder counts.
        """
        if self.connected:
            self.inst.move_relative_raw(counts, blocking=blocking)
        else:
            self.error("Device not connected")

    def is_moving(self, raise_errors=False):
        """
        Check if the motor is currently performing a move operation.
        """
        if self.connected:
            return self.inst.is_moving(raise_errors=raise_errors)
        else:
            self.error("Device not connected")

    def wait(self, raise_errors=False):
        """
        Block until any current movement is completed.
        """
        if self.connected:
            self.move_absolute_raw()
        else:
            self.error("Device not connected")

    # @property
    # def status(self):
    #     """
    #     Get the current state of the device.
    #     """
    #     if self.connected:
    #         return self.inst.status
    #     else:
    #         self.error("Device not connected")


class ElliptecDualPositioner(Elliptec):
    def __init__(self, addr: str = 'COM7', name: str = 'Thorlabs Elliptec Dual Positioner', **kwargs):
        super().__init__(addr=addr, name=name, **kwargs)
    
    def get_position(self):
        """
        Get the current position of the motor in real device units.
        """
        if self.connected:
            return Ell6Position(self.inst.get_position_raw())
        else:
            self.error("Device not connected")
        
    def set_position(self, position, blocking=True):
        """
        Set the position of the motor in real device units.
        """
        p = Ell6Position(position)
        if self.connected:
            self.move_absolute_raw(p.value, blocking=blocking)
            self.info(f"Position set to {p.name}")
        else:
            self.error("Device not connected")

    def toggle_position(self, blocking=True):
        """
        Toggle the position of the dual positioner between OPEN and BLOCK.
        """
        if self.connected:
            current_position = Ell6Position(self.inst.get_position_raw())
            # Determine the next position
            new_position = 31 if current_position.value == 0 else 0
            self.set_position(new_position, blocking=blocking)
        else:
            self.error("Device not connected")

    def cycle_position(self, cycles, delay=1.0, blocking=True):
        """
        Cycle the position of the dual positioner between OPEN and BLOCK for a specified number of cycles.

        Parameters:
            cycles (int): Number of cycles to perform.
            delay (float): Delay in seconds between position changes.
            blocking (bool): Wait for each movement to complete before proceeding.
        """
        if not self.connected:
            self.error("Device not connected")
            return

        # Query the current position only once
        current_position = Ell6Position(self.inst.get_position_raw())
        next_position = Ell6Position(31 if current_position.value == 0 else 0)
        starting_position = current_position

        for _ in range(cycles):
            # Set the new position
            self.set_position(next_position.value, blocking=blocking)
            time.sleep(delay)
            # Toggle positions for the next cycle
            current_position, next_position = next_position, current_position
    
        # Return to the starting position
        self.set_position(starting_position.value, blocking=blocking)


    
class Ell6Position: # ELL6 is the dual position stage
    _position_map = {
        "OPEN": 0,
        "BLOCK": 31,
    }

    _reversed_map = {v: k for k, v in _position_map.items()}

    def __init__(self, value):
        if isinstance(value, int):
            self._init_with_int(value)
        elif isinstance(value, float) and value.is_integer():
            self._init_with_int(int(value))
        elif isinstance(value, str):
            self._init_with_str(value)
        else:
            raise TypeError(
                f"Value must be an integer, a float integer, or a string, not {type(value).__name__}."
            )

    def _init_with_int(self, value):
        if value in self._reversed_map:
            self.value = value
            self.name = self._reversed_map[value]
        else:
            valid_values = ", ".join(str(v) for v in self._reversed_map.keys())
            raise ValueError(
                f"Invalid position value: {value}. Valid values: {valid_values}."
            )

    def _init_with_str(self, value):
        normalized_value = value.strip().upper()
        if normalized_value.isdigit():
            int_value = int(normalized_value)
            self._init_with_int(int_value)
        elif normalized_value in self._position_map:
            self.value = self._position_map[normalized_value]
            self.name = normalized_value
        else:
            valid_names = ", ".join(self._position_map.keys())
            raise ValueError(
                f"Invalid position name: '{value}'. Valid names: {valid_names}."
            )

    @classmethod
    def get_name(cls, value):
        if value in cls._reversed_map:
            return cls._reversed_map[value]
        raise ValueError(f"Invalid position value: {value}.")

    @classmethod
    def get_value(cls, name):
        normalized_name = name.strip().upper()
        if normalized_name in cls._position_map:
            return cls._position_map[normalized_name]
        raise ValueError(f"Invalid position name: '{name}'.")

    def __repr__(self):
        return f"ElliptecPosition(name='{self.name}', value={self.value})"

    def __eq__(self, other):
        if isinstance(other, Ell6Position):
            return self.value == other.value
        if isinstance(other, int):
            return self.value == other
        if isinstance(other, str):
            return self.name == other.upper()
        if isinstance(other, float) and other.is_integer():
            return self.value == int(other)
        return NotImplemented