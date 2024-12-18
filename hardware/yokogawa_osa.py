from .device import Device
from decimal import Decimal
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.io import savemat
from astropy.io import fits
import time

C_CONST = 299792458  # Speed of light in m/s

class YokogawaOSA(Device):
    def __init__(self, addr="192.168.1.27", # 'GPIB0::1::INSTR'
                  name="OSA"):
       

        if addr.startswith('GPIB'):
            self.com_protocol = 'GPIB'

            super().__init__(addr=addr, name=name, isVISA=True)

            
            self.inst.timeout = 25000  # Communication timeout in ms
            self.inst.baud_rate = 19200  # Baud rate for communication
            self.inst.read_termination = ''  # Read termination
            self.inst.write_termination = ''  # Default is '\r\n'

        else:
            self.com_protocol = 'vxi11'

            super().__init__(addr=addr, name=name, isVISA=False)

        # Default device settings
        self.__activation_timeout = 3  # Time to wait for device activation/deactivation in seconds
        self.__wavelength_prec_nm = '0.00'  # Wavelength precision up to 2 decimal places
        self.__available_traces = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        self.osaacquiring = False

    def connect(self):
        if self.com_protocol == 'GPIB':
            super().connect()
        else:
            if not self.connected:
                try:
                    import vxi11
                    self.inst =  vxi11.Instrument(self.addr)
                    self.connected = True
                    self.info(f'{self.devicename}: Connected via {self.com_protocol} protocol')
                    return 1
                except Exception as e:
                    self.error(f'{self.devicename}: Failed to connect via {self.com_protocol} protocol. Error: {e}')
                    return -1
            return 0
    
    def disconnect(self):
        if self.com_protocol == 'GPIB':
            super().disconnect()
        else:
            if self.connected:
                self.inst.close()
                self.connected = False
                self.info(f'{self.devicename}: via {self.com_protocol} protocol device released.')
                return 1
            return 0
    
    def query(self, cmd):
        if self.com_protocol == 'GPIB':
            return super().query(cmd)
        else:
            return self.inst.ask(cmd)
            

    def set_resolution_nm(self, resolution_nm):
        try:
            resolution_nm = Decimal(resolution_nm).quantize(Decimal(self.__wavelength_prec_nm))
            self.write(f':SENSE:BANDWIDTH:RESOLUTION {resolution_nm}NM')
            self.info(f'{self.devicename}: Set resolution to {resolution_nm} nm')
        except Exception as e:
            self.error(f'{self.devicename}: Failed to set resolution to {resolution_nm} nm. Error: {e}')
            raise

    def get_resolution_nm(self):
        try:
            return float(self.query(':SENSE:BANDWIDTH:RESOLUTION?')) / 1e-9
        except Exception as e:
            self.error(f'{self.devicename}: Failed to get resolution. Error: {e}')
            raise

    def set_wavelength_center_nm(self, wavelength_center_nm):
        try:
            wavelength_center_nm = Decimal(wavelength_center_nm).quantize(Decimal(self.__wavelength_prec_nm))
            self.write(f':SENSE:WAVELENGTH:CENTER {wavelength_center_nm}NM')
            self.info(f'{self.devicename}: Set wavelength center to {wavelength_center_nm} nm')
        except Exception as e:
            self.error(f'{self.devicename}: Failed to set wavelength center to {wavelength_center_nm} nm. Error: {e}')
            raise

    def get_wavelength_center_nm(self):
        try:
            return float(self.query(':SENSE:WAVELENGTH:CENTER?')) / 1e-9
        except Exception as e:
            self.error(f'{self.devicename}: Failed to get wavelength center. Error: {e}')
            raise

    def set_wavelength_span_nm(self, wavelength_span_nm):
        try:
            wavelength_span_nm = Decimal(wavelength_span_nm).quantize(Decimal(self.__wavelength_prec_nm))
            self.write(f':SENSE:WAVELENGTH:SPAN {wavelength_span_nm}NM')
            self.info(f'{self.devicename}: Set wavelength span to {wavelength_span_nm} nm')
        except Exception as e:
            self.error(f'{self.devicename}: Failed to set wavelength span to {wavelength_span_nm} nm. Error: {e}')
            raise

    def get_wavelength_span_nm(self):
        try:
            return float(self.query(':SENSE:WAVELENGTH:SPAN?')) / 1e-9
        except Exception as e:
            self.error(f'{self.devicename}: Failed to get wavelength span. Error: {e}')
            raise

    def set_wavelength_start_nm(self, wavelength_start_nm):
        try:
            wavelength_start_nm = Decimal(wavelength_start_nm).quantize(Decimal(self.__wavelength_prec_nm))
            self.write(f':SENSE:WAVELENGTH:START {wavelength_start_nm}NM')
            self.info(f'{self.devicename}: Set wavelength start to {wavelength_start_nm} nm')
        except Exception as e:
            self.error(f'{self.devicename}: Failed to set wavelength start to {wavelength_start_nm} nm. Error: {e}')
            raise

    def get_wavelength_start_nm(self):
        try:
            return float(self.query(':SENSE:WAVELENGTH:START?')) / 1e-9
        except Exception as e:
            self.error(f'{self.devicename}: Failed to get wavelength start. Error: {e}')
            raise

    def set_wavelength_stop_nm(self, wavelength_stop_nm):
        try:
            wavelength_stop_nm = Decimal(wavelength_stop_nm).quantize(Decimal(self.__wavelength_prec_nm))
            self.write(f':SENSE:WAVELENGTH:STOP {wavelength_stop_nm}NM')
            self.info(f'{self.devicename}: Set wavelength stop to {wavelength_stop_nm} nm')
        except Exception as e:
            self.error(f'{self.devicename}: Failed to set wavelength stop to {wavelength_stop_nm} nm. Error: {e}')
            raise

    def get_wavelength_stop_nm(self):
        try:
            return float(self.query(':SENSE:WAVELENGTH:STOP?')) / 1e-9
        except Exception as e:
            self.error(f'{self.devicename}: Failed to get wavelength stop. Error: {e}')
            raise

    def set_sensitivity(self, sensitivity):
        try:
            sensitivity = OSASensitivity(sensitivity)
            self.write(f':SENSE:SENSE {sensitivity.name}')
            self.info(f'{self.devicename}: Set sensitivity to {sensitivity.name}')
        except Exception as e:
            self.error(f'{self.devicename}: Failed to set sensitivity to {sensitivity}. Error: {e}')
            raise

    def get_sensitivity(self):
        try:
            return OSASensitivity(self.query(':SENSE:SENSE?')).name
        except Exception as e:
            self.error(f'{self.devicename}: Failed to get sensitivity. Error: {e}')
            raise

    def set_sampling_step_nm(self, sampling_step_nm):
        try:
            sampling_step_nm = Decimal(sampling_step_nm).quantize(Decimal(self.__wavelength_prec_nm))
            self.write(f':SENSE:SWEEP:STEP {sampling_step_nm}NM')
            self.info(f'{self.devicename}: Set sampling step to {sampling_step_nm} nm')
        except Exception as e:
            self.error(f'{self.devicename}: Failed to set sampling step to {sampling_step_nm} nm. Error: {e}')
            raise

    def get_sampling_step_nm(self):
        try:
            return float(self.query(':SENSE:SWEEP:STEP?')) / 1e-9
        except Exception as e:
            self.error(f'{self.devicename}: Failed to get sampling step. Error: {e}')
            raise

    def set_sample_points(self, sample_points):
        try:
            sample_points = int(sample_points)
            self.write(f':SENSE:SWEEP:POINTS {sample_points}')
            self.info(f'{self.devicename}: Set sample points to {sample_points}')
        except Exception as e:
            self.error(f'{self.devicename}: Failed to set sample points to {sample_points}. Error: {e}')
            raise

    def get_sample_points(self):
        try:
            return int(self.query(':SENSE:SWEEP:POINTS?'))
        except Exception as e:
            self.error(f'{self.devicename}: Failed to get sample points. Error: {e}')
            raise

    def set_horizontal_unit(self, unit):
        try:
            unit = OSAHorizontalUnit(unit)
            self.write(f":UNIT:X {unit.value}")
            self.info(f'{self.devicename}: Set horizontal unit to {unit.name}')
        except Exception as e:
            self.error(f'{self.devicename}: Failed to set horizontal unit to {unit}. Error: {e}')
            raise

    def get_horizontal_unit(self):
        try:
            unit_value = int(self.query(":UNIT:X?"))
            return OSAHorizontalUnit.get_name(unit_value)
        except Exception as e:
            self.error(f'{self.devicename}: Failed to get horizontal unit. Error: {e}')
            raise

    def run(self, trace_to_write=None):
        try:
            if trace_to_write is not None:
                self.fix_trace('all')
                self.write_trace(trace_to_write)
            self.write(":INITiate:SMODe REPeat")
            self.write(":INITiate")
            self.info(f'{self.devicename}: Initiated repeat mode')
        except Exception as e:
            self.error(f'{self.devicename}: Failed to initiate repeat mode. Error: {e}')
            raise

    def single(self):
        try:
            self.write(":INITiate:SMODe SINGle")
            self.write(":INITiate")
            self.info(f'{self.devicename}: Initiated single mode')
        except Exception as e:
            self.error(f'{self.devicename}: Failed to initiate single mode. Error: {e}')
            raise

    def single_and_wait(self):
        '''Initiate a single sweep and wait for it to complete.'''
        self.single()
        self.wait_sweep()

    def run_and_wait(self, trace_to_write=None):
        '''Initiate a repeat sweep and wait for the first sweep to complete.'''
        self.run(trace_to_write)
        self.wait_sweep()

    def stop(self):
        try:
            self.write(":ABORt")
            self.info(f'{self.devicename}: Stopped operation')
        except Exception as e:
            self.error(f'{self.devicename}: Failed to stop operation. Error: {e}')
            raise

    def get_operation_status_register(self):
        try:
            flag = int(self.query(":STATUS:OPERATION:CONDITION?"))
        except Exception as e:
            self.error(f'{self.devicename}: Failed to get operation status register. Error: {e}')
        a = flag // 16 # Auto Sweep: Completion of auto sweep running action
        b = (flag - a*16) // 8 # Cal: Completion of a Cal wavelength calibration or resolution calibration
        c = (flag - a*16 - b*8) // 4 # File: Completion of file operation
        d = (flag - a*16 - b*8 - c*4) // 2 # Program: Completion of execution of the program functions
        e = flag - a*16 - b*8 - c*4 - d*2 # Sweep: Completion of a sweep
        return {'Auto Sweep': a, 'Cal': b, 'File': c, 'Program': d, 'Sweep': e}

    def wait_sweep(self):
        while True:
            status = self.get_operation_status_register()
            if status['Sweep']:
                break
            time.sleep(1)

    def get_trace_status(self, trace):
        """Get the status of the specified trace."""
        try:
            trace = self._validate_trace(trace)
            dp = self.query(f":TRAC:STAT:TR{trace}?") # Display='1', Blank='0'
            wr = self.query(f":TRAC:ATTR:TR{trace}?") # Write='0', Fix='1'
            dp = True if dp == '1' else False
            wr = True if wr == '0' else False
            return {'Display': dp, 'Write': wr, 'Blank': not dp, 'Fix': not wr}
        except Exception as e:
            self.error(f'{self.devicename}: Failed to get trace status for trace {trace}. Error: {e}')
            raise

    def display_trace(self, trace):
        """Display the specified trace."""
        if "".join([x for x in trace if x.isalpha()]).lower().strip().startswith('allbut'):
            try:
                allbut = "".join([x for x in trace if x.isalpha()]).lower().strip().replace('allbut', '').strip()
                allbut = [self._validate_trace(t) for t in allbut]
                for t in self.__available_traces:
                    if t not in allbut:
                        self.display_trace(t)
                    else:
                        self.blank_trace(t)
                self.info(f'{self.devicename}: All traces are displayed except {allbut}')
                return
            except Exception as e:
                self.error(f'{self.devicename}: Failed to display all traces except {allbut}. Error: {e}')
                raise
            
        if trace.lower().strip() == 'all':
            try:
                for t in self.__available_traces:
                    self.display_trace(t)
                self.info(f'{self.devicename}: All traces are displayed.')
                return
            except Exception as e:
                self.error(f'{self.devicename}: Failed to display all traces. Error: {e}')

        try:
            trace = self._validate_trace(trace)
            self.write(f":TRAC:STAT:TR{trace} ON")
            self.info(f'{self.devicename}: Trace {trace} is displayed.')
        except Exception as e:
            self.error(f'{self.devicename}: Failed to display trace {trace}. Error: {e}')
            raise

    def blank_trace(self, trace):
        """Blank the specified trace."""
        if "".join([x for x in trace if x.isalpha()]).lower().strip().startswith('allbut'):
            try:
                allbut = "".join([x for x in trace if x.isalpha()]).lower().strip().replace('allbut', '').strip()
                allbut = [self._validate_trace(t) for t in allbut]
                for t in self.__available_traces:
                    if t not in allbut:
                        self.blank_trace(t)
                    else:
                        self.display_trace(t)
                self.info(f'{self.devicename}: All traces are blanked except {allbut}')
                return
            except Exception as e:
                self.error(f'{self.devicename}: Failed to blank all traces except {allbut}. Error: {e}')
                raise
        
        if trace.lower().strip() == 'all':
            try:
                for t in self.__available_traces:
                    self.blank_trace(t)
                self.info(f'{self.devicename}: All traces are blanked.')
                return
            except Exception as e:
                self.error(f'{self.devicename}: Failed to blank all traces. Error: {e}')

        try:
            trace = self._validate_trace(trace)
            self.write(f":TRAC:STAT:TR{trace} OFF")
            self.info(f'{self.devicename}: Trace {trace} is blanked.')
        except Exception as e:
            self.error(f'{self.devicename}: Failed to blank trace {trace}. Error: {e}')
            raise

    def write_trace(self, trace):
        """Write the specified trace."""
        try:
            trace = self._validate_trace(trace)
            self.write(f"TRAC:ATTR:TR{trace} WRIT")
            self.info(f'{self.devicename}: Trace {trace} is being written.')
        except Exception as e:
            self.error(f'{self.devicename}: Failed to write trace {trace}. Error: {e}')
            raise

    def fix_trace(self, trace):
        """Fix the specified trace."""
        if "".join([x for x in trace if x.isalpha()]).lower().strip().startswith('allbut'):
            try:
                allbut = "".join([x for x in trace if x.isalpha()]).lower().strip().replace('allbut', '').strip()
                allbut = [self._validate_trace(t) for t in allbut]
                for t in self.__available_traces:
                    if t not in allbut:
                        self.fix_trace(t)
                    else:
                        self.write_trace(t)
                self.info(f'{self.devicename}: All traces are written except {allbut}')
                return
            except Exception as e:
                self.error(f'{self.devicename}: Failed to write all traces except {allbut}. Error: {e}')
                raise

        try:
            trace = self._validate_trace(trace)
            self.write(f"TRAC:ATTR:TR{trace} FIX")
            self.info(f'{self.devicename}: Trace {trace} is being fixed.')
        except Exception as e:
            self.error(f'{self.devicename}: Failed to fix trace {trace}. Error: {e}')
            raise

    def _validate_trace(self, trace):
        """Validate and normalize the trace name."""
        trace = str(trace).strip().upper()
        if trace not in self.__available_traces:
            raise ValueError(
                self.devicename + ": Trace name " + trace + " unrecognized. "
                + "Trace must select from " + str(self.__available_traces)
            )
        return trace

    def get_XData(self, trace, logdata=True):
        """Retrieve X data (wavelengths) for the specified trace."""
        try:
            trace = self._validate_trace(trace)
            result = np.asarray(self.query(f':TRACE:X? TR{trace}').strip().split(','), dtype=float)/1e-9
            if logdata:
                result_str = np.array2string(result, separator=',', formatter={'float_kind': lambda x: f'{x:.2f}'}, threshold=np.inf, max_line_width=np.inf)
                self.info(f'{self.devicename}: Retrieved X data for trace {trace} is:\n {result_str}')
            return result
        except Exception as e:
            self.error(f'{self.devicename}: Failed to get X data for trace {trace}. Error: {e}')
            raise

    def get_YData(self, trace, logdata=True):
        """Retrieve Y data (intensities) for the specified trace."""
        try:
            trace = self._validate_trace(trace)
            result = np.asarray(self.query(f':TRACE:Y? TR{trace}').strip().split(','), dtype=float)
            if logdata:
                result_str = np.array2string(result, separator=',', formatter={'float_kind': lambda x: f'{x:.2f}'}, threshold=np.inf, max_line_width=np.inf)
                self.info(f'{self.devicename}: Retrieved Y data for trace {trace} is:\n {result_str}')
            return result
        except Exception as e:
            self.error(f'{self.devicename}: Failed to get Y data for trace {trace}. Error: {e}')
            raise

    def peak_search(self, trace, adv_marker=1):
        """Search for peaks in the specified trace."""
        try:
            trace = self._validate_trace(trace)
            # self.display_trace(trace)
            flag = False
            if not self.get_trace_status(trace)['Display']:
                self.display_trace(trace)
                flag = True
            self.wait_sweep()
            self.write(f":CALC:AMAR{adv_marker}:TRAC TR{trace}")
            self.write(f":CALC:AMAR{adv_marker} 1")
            self.write(f":CALC:AMAR{adv_marker}:MAX")
            wl = float(self.query(f":CALC:AMAR{adv_marker}:X?"))
            pr = float(self.query(f":CALC:AMAR{adv_marker}:Y?"))
            self.info(f'{self.devicename}: Peak search is performed on trace {trace}, X: {wl}, Y: {pr}.')
            if flag:
                self.blank_trace(trace)
            return wl, pr
        except Exception as e:
            self.error(f'{self.devicename}: Failed to perform peak search on trace {trace}. Error: {e}')
            raise

    def get_trace(self, trace, plot=True, filename=None, logdata=False):
        """Retrieve both X and Y data for the specified trace and optionally plot it."""
        trace = self._validate_trace(trace)
        wl = self.get_XData(trace, logdata=logdata)
        intensity = self.get_YData(trace, logdata=logdata)

        if plot:
            self.plot_trace(wl, intensity, trace, filename)
        return wl, intensity

    def plot_trace(self, wl, intensity, trace, filename=None):
        """Plot the trace data."""
        plt.figure()
        plt.plot(wl, intensity)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity (dBm)')
        plt.ylim(bottom=-75)

        if filename is not None:
            # Check if filename already has an extension
            base_filename, ext = os.path.splitext(filename)
            if ext:
                filename = base_filename  # Strip existing extension
            plt.savefig(filename + '.png')  # Always save as .png

        plt.show()
        self.info(self.devicename + ": Trace " + trace + " data is collected and shown in the plot.")

    def save_trace(self, trace, filename, extensions=['.mat', '.txt', '.npy'], plot=True, logdata=False):
        trace = self._validate_trace(trace)

        # Check if filename already has an extension
        base_filename, ext = os.path.splitext(filename)
        if ext:
            # Use the given extension only, and ignore default extensions
            extensions = [ext.lower()]
            filename = base_filename  # Strip extension from filename
        else:
            # Handle default extensions
            extensions = self._validate_extensions(extensions)

        # Prepare the file
        filename = self._prepare_file(filename, extensions)

        # Read data
        wl, intensity = self.get_trace(trace, plot=plot, logdata=logdata)

        # Save data
        self._save_data(wl, intensity, filename, extensions)

    def _validate_extensions(self, extensions):
        """Validate and normalize the list of file extensions."""
        available_extensions = ['.mat', '.txt', '.fits', '.csv', '.npy']
        if isinstance(extensions, str) and extensions.casefold() == 'all':
            return available_extensions
        if not isinstance(extensions, list):
            extensions = [extensions]
        extensions = [str(ext).lower() for ext in extensions]
        for ext in extensions:
            if ext not in available_extensions:
                raise ValueError(self.devicename + ": Extension " + ext + " is not available. Available: " +
                                 str(available_extensions))
        return extensions

    def _prepare_file(self, filename, extensions):
        """Prepare the file by creating directories and handling existing files."""
        filedir, single_filename = os.path.split(filename)
        if not os.path.isdir(filedir) and filedir != '':
            self.warning(self.devicename + ": Directory " + filedir + " does not exist. Creating new directory.")
            os.makedirs(filedir)

        for ext in extensions:
            if os.path.isfile(filename + ext):
                self.warning(self.devicename + ": Filename " + filename + ext + " already exists. Previous file renamed.")
                now = time.strftime("%Y%m%d_%H%M%S")  # Current date and time
                new_name = f"{filename}_{now}_bak{ext}"
                os.rename(filename + ext, new_name)

        return filename

    def _save_data(self, wl, intensity, filename, extensions):
        """Save trace data to specified file formats."""
        # Get the current vertical and horizontal units
        vertical_unit = self.get_unit()
        horizontal_unit = self.get_horizontal_unit()

        for ext in extensions:
            try:
                if ext == '.mat':
                    self._save_as_mat(wl, intensity, filename, vertical_unit, horizontal_unit)
                elif ext == '.txt':
                    self._save_as_txt(wl, intensity, filename, vertical_unit, horizontal_unit)
                elif ext == '.fits':
                    self._save_as_fits(wl, intensity, filename, vertical_unit, horizontal_unit)
                elif ext == '.csv':
                    self._save_as_csv(wl, intensity, filename, vertical_unit, horizontal_unit)
                elif ext == '.npy':
                    self._save_as_npy(wl, intensity, filename, vertical_unit, horizontal_unit)
            except Exception as e:
                self.warning(self.devicename + f": Save trace to {filename}{ext} failed.")
                self.error(f"Error: {e}")

    def _save_as_mat(self, wl, intensity, filename, vertical_unit, horizontal_unit):
        """Save data as .mat file."""
        savemat(filename + '.mat', {
            'OSAWavelength': wl,
            'OSAPower': intensity,
            'VerticalUnit': vertical_unit,
            'HorizontalUnit': horizontal_unit,
            'resolution': self.get_resolution_nm(),
            'timestamp': time.ctime()
        }, oned_as='column')
        self.info(self.devicename + ": Trace data is saved to " + filename + '.mat')

    def _save_as_txt(self, wl, intensity, filename, vertical_unit, horizontal_unit):
        """Save data as .txt file."""
        data = np.column_stack((wl, intensity))
        np.savetxt(filename + '.txt', data, fmt='%.6f', 
                   header=f'Wavelength ({horizontal_unit}), Intensity ({vertical_unit})', comments='')
        self.info(self.devicename + ": Trace data is saved to " + filename + '.txt')

    def _save_as_fits(self, wl, intensity, filename, vertical_unit, horizontal_unit):
        """Save data as .fits file."""
        hdu = fits.PrimaryHDU(np.array([wl, intensity]).T)
        hdu.header['VUNIT'] = vertical_unit
        hdu.header['HUNIT'] = horizontal_unit
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(filename + '.fits', overwrite=True)
        hdulist.close()
        self.info(self.devicename + ": Trace data is saved to " + filename + '.fits')

    def _save_as_csv(self, wl, intensity, filename, vertical_unit, horizontal_unit):
        """Save data as .csv file."""
        np.savetxt(filename + '.csv', np.array([wl, intensity]).T, delimiter=",",
                   header=f'Wavelength ({horizontal_unit}), Intensity ({vertical_unit})', comments='')
        self.info(self.devicename + ": Trace data is saved to " + filename + '.csv')

    def _save_as_npy(self, wl, intensity, filename, vertical_unit, horizontal_unit):
        """Save data as .npy file."""
        np.save(filename + '.npy', {'wavelength': wl, 'intensity': intensity, 'vertical_unit': vertical_unit, 'horizontal_unit': horizontal_unit})
        self.info(self.devicename + ": Trace data is saved to " + filename + '.npy')

    def set_ref_level_dbm(self, ref_level_dbm):
        """Set the reference level in dBm."""
        try:
            self.write(f":DISP:TRAC:Y1:RLEV {ref_level_dbm}DBM")
            self.info(f'{self.devicename}: Set ref level to {ref_level_dbm} dBm')
        except Exception as e:
            self.error(f'{self.devicename}: Failed to set ref level to {ref_level_dbm} dBm. Error: {e}')
            raise

    def get_ref_level_dbm(self):
        """Get the reference level in dBm."""
        try:
            return float(self.query(":DISP:TRAC:Y1:RLEV?"))
        except Exception as e:
            self.error(f'{self.devicename}: Failed to get ref level in dBm. Error: {e}')
            raise

    def set_ref_scale_db_per_div(self, ref_scale_db_per_div):
        """Set the reference scale in dB/Div."""
        try:
            self.write(f":DISP:TRAC:Y1:PDIV {ref_scale_db_per_div}DB")
            self.info(f'{self.devicename}: Set ref scale to {ref_scale_db_per_div} dB/Div')
        except Exception as e:
            self.error(f'{self.devicename}: Failed to set ref scale to {ref_scale_db_per_div} dB/Div. Error: {e}')
            raise

    def get_ref_scale_db_per_div(self):
        """Get the reference scale in dB/Div."""
        try:
            return float(self.query(":DISP:TRAC:Y1:PDIV?"))
        except Exception as e:
            self.error(f'{self.devicename}: Failed to get ref scale in dB/Div. Error: {e}')
            raise

    def set_unit(self, unit):
        """Set the vertical unit."""
        try:
            unit = OSAVerticalUnit(unit)
            self.write(f":DISP:TRAC:Y1:UNIT {unit.name}")
            self.info(f'{self.devicename}: Set unit to {unit.name}')
        except Exception as e:
            self.error(f'{self.devicename}: Failed to set unit to {unit}. Error: {e}')
            raise

    def get_unit(self):
        """Get the vertical unit."""
        try:
            unit_value = int(self.query(":DISP:TRAC:Y1:UNIT?"))
            return OSAVerticalUnit.get_name(unit_value)
        except Exception as e:
            self.error(f'{self.devicename}: Failed to get unit. Error: {e}')
            raise

    def set_ref_level_w(self, ref_level_w):
        """Set the reference level in Watts."""
        try:
            self.write(f":DISP:TRAC:Y1:RLEV {ref_level_w}W")
            self.info(f'{self.devicename}: Set ref level to {ref_level_w} W')
        except Exception as e:
            self.error(f'{self.devicename}: Failed to set ref level to {ref_level_w} W. Error: {e}')
            raise

    def get_ref_level_w(self):
        """Get the reference level in Watts."""
        try:
            return float(self.query(":DISP:TRAC:Y1:RLEV?"))
        except Exception as e:
            self.error(f'{self.devicename}: Failed to get ref level in Watts. Error: {e}')
            raise

    @property
    def resolution_nm(self):
        return self.get_resolution_nm()

    @resolution_nm.setter
    def resolution_nm(self, value):
        self.set_resolution_nm(value)

    @property
    def wavelength_center_nm(self):
        return self.get_wavelength_center_nm()

    @wavelength_center_nm.setter
    def wavelength_center_nm(self, value):
        self.set_wavelength_center_nm(value)

    @property
    def wavelength_span_nm(self):
        return self.get_wavelength_span_nm()

    @wavelength_span_nm.setter
    def wavelength_span_nm(self, value):
        self.set_wavelength_span_nm(value)

    @property
    def wavelength_start_nm(self):
        return self.get_wavelength_start_nm()

    @wavelength_start_nm.setter
    def wavelength_start_nm(self, value):
        self.set_wavelength_start_nm(value)

    @property
    def wavelength_stop_nm(self):
        return self.get_wavelength_stop_nm()

    @wavelength_stop_nm.setter
    def wavelength_stop_nm(self, value):
        self.set_wavelength_stop_nm(value)

    @property
    def sensitivity(self):
        return self.get_sensitivity()

    @sensitivity.setter
    def sensitivity(self, value):
        self.set_sensitivity(value)

    @property
    def sampling_step_nm(self):
        return self.get_sampling_step_nm()

    @sampling_step_nm.setter
    def sampling_step_nm(self, value):
        self.set_sampling_step_nm(value)

    @property
    def sample_points(self):
        return self.get_sample_points()

    @sample_points.setter
    def sample_points(self, value):
        self.set_sample_points(value)

    @property
    def horizontal_unit(self):
        return self.get_horizontal_unit()

    @horizontal_unit.setter
    def horizontal_unit(self, value):
        self.set_horizontal_unit(value)

    @property
    def ref_level_dbm(self):
        return self.get_ref_level_dbm()

    @ref_level_dbm.setter
    def ref_level_dbm(self, value):
        self.set_ref_level_dbm(value)

    @property
    def ref_scale_db_per_div(self):
        return self.get_ref_scale_db_per_div()

    @ref_scale_db_per_div.setter
    def ref_scale_db_per_div(self, value):
        self.set_ref_scale_db_per_div(value)

    @property
    def unit(self):
        return self.get_unit()

    @unit.setter
    def unit(self, value):
        self.set_unit(value)

    @property
    def ref_level_w(self):
        return self.get_ref_level_w()

    @ref_level_w.setter
    def ref_level_w(self, value):
        self.set_ref_level_w(value)

    @classmethod
    def get_comb_peak_spec(self, 
                           spec_nm, spec_dbm, fsr_Ghz, 
                           spec_dbm_noise = -40, plot=False):
        """Get the combined peak spectrum."""

        osa_wl_center = np.mean(spec_nm)
        fsr_nm = osa_wl_center - 1/(1/osa_wl_center + fsr_Ghz/C_CONST)
        wl_increment = np.mean(np.diff(spec_nm))
        min_distance = fsr_nm / wl_increment
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(spec_dbm, height=spec_dbm_noise, distance=0.9*min_distance)

        if plot:
            plt.figure()
            plt.plot(spec_nm, spec_dbm)
            plt.plot(spec_nm[peaks], spec_dbm[peaks], "x")
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Intensity (dBm)')
            plt.ylim(bottom=spec_dbm_noise)
            plt.show()
        return spec_nm[peaks], spec_dbm[peaks]




class OSASensitivity:
    _sensitivity_map = {
        0: "NHLD",
        1: "NAUT",
        2: "MID",
        3: "HIGH1",
        4: "HIGH2",
        5: "HIGH3",
        6: "NORMAL",
    }

    _reversed_map = {v: k for k, v in _sensitivity_map.items()}

    # def __init__(self, value):
    #     if isinstance(value, int):
    #         self._init_with_int(value)
    #     elif isinstance(value, str):
    #         self._init_with_str(value)
    #     else:
    #         raise TypeError(f"Value must be an integer or a string, not {type(value).__name__}.")
        
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
        if value in self._sensitivity_map:
            self.value = value
            self.name = self._sensitivity_map[value]
        else:
            raise ValueError(f"Invalid sensitivity value: {value}. Valid values are integers from 0 to 6.")

    def _init_with_str(self, value):
        normalized_value = value.strip().upper()
        if normalized_value.isdigit():  # Check if it's a string representation of an integer
            int_value = int(normalized_value)
            self._init_with_int(int_value)
        elif normalized_value in self._reversed_map:
            self.value = self._reversed_map[normalized_value]
            self.name = normalized_value
        else:
            raise ValueError(
                f"Invalid sensitivity name or value: '{value}'. "
                "Valid names are: " + ", ".join(self._sensitivity_map.values())
            )

    @classmethod
    def get_name(cls, value):
        if value in cls._sensitivity_map:
            return cls._sensitivity_map[value]
        raise ValueError(f"OSA sensitivity should choose from 0 to 6, not {value}.")

    @classmethod
    def get_value(cls, name):
        normalized_name = name.upper()
        if normalized_name in cls._reversed_map:
            return cls._reversed_map[normalized_name]
        raise ValueError(
            f"Invalid sensitivity name: '{name}'. "
            "Valid names are: " + ", ".join(cls._sensitivity_map.values())
        )

class OSAVerticalUnit:
    _unit_map = {
        "DBM": 0,
        "W": 1,
        "DBM/NM": 2,
        "W/NM": 3,
    }

    _reversed_map = {v: k for k, v in _unit_map.items()}

    # def __init__(self, value):
    #     if isinstance(value, int):
    #         self._init_with_int(value)
    #     elif isinstance(value, str):
    #         self._init_with_str(value)
    #     else:
    #         raise TypeError(
    #             f"Value must be an integer or a string, not {type(value).__name__}."
    #         )
        
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
            raise ValueError(
                f"Invalid vertical unit value: {value}. Valid values are 0 to 3."
            )

    def _init_with_str(self, value):
        normalized_value = value.strip().upper()
        if normalized_value.isdigit():
            int_value = int(normalized_value)
            self._init_with_int(int_value)
        elif normalized_value in self._unit_map:
            self.value = self._unit_map[normalized_value]
            self.name = normalized_value
        else:
            valid_names = ", ".join(self._unit_map.keys())
            raise ValueError(
                f"Invalid vertical unit name: '{value}'. Valid names: {valid_names}."
            )

    @classmethod
    def get_name(cls, value):
        if value in cls._reversed_map:
            return cls._reversed_map[value]
        raise ValueError(f"Invalid vertical unit value: {value}.")

    @classmethod
    def get_value(cls, name):
        normalized_name = name.upper()
        if normalized_name in cls._unit_map:
            return cls._unit_map[normalized_name]
        raise ValueError(f"Invalid vertical unit name: '{name}'.")

class OSAHorizontalUnit:
    _unit_map = {
        "NM": 0,
        "THZ": 1,
    }

    _reversed_map = {v: k for k, v in _unit_map.items()}

    # def __init__(self, value):
    #     if isinstance(value, int):
    #         self._init_with_int(value)
    #     elif isinstance(value, str):
    #         self._init_with_str(value)
    #     else:
    #         raise TypeError(
    #             f"Value must be an integer or a string, not {type(value).__name__}."
    #         )
        
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
            raise ValueError(
                f"Invalid horizontal unit value: {value}. Valid values are 0 or 1."
            )

    def _init_with_str(self, value):
        normalized_value = value.strip().upper()
        if normalized_value.isdigit():
            int_value = int(normalized_value)
            self._init_with_int(int_value)
        elif normalized_value in self._unit_map:
            self.value = self._unit_map[normalized_value]
            self.name = normalized_value
        else:
            valid_names = ", ".join(self._unit_map.keys())
            raise ValueError(
                f"Invalid horizontal unit name: '{value}'. Valid names: {valid_names}."
            )

    @classmethod
    def get_name(cls, value):
        if value in cls._reversed_map:
            return cls._reversed_map[value]
        raise ValueError(f"Invalid horizontal unit value: {value}.")

    @classmethod
    def get_value(cls, name):
        normalized_name = name.upper()
        if normalized_name in cls._unit_map:
            return cls._unit_map[normalized_name]
        raise ValueError(f"Invalid horizontal unit name: '{name}'.")
