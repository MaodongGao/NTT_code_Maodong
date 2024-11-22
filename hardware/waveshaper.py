import time
import requests
import numpy as np
import matplotlib.pyplot as plt
import json
from .device import Device


class waveshaper(Device):
    c_const = 299792458  # Speed of light in m/s

    def __init__(self, addr="192.168.1.22", name='wsp', isVISA=False):
        super().__init__(addr, name, isVISA)
        config = {
            "ip": self.addr,
            "max_retries" : 4,
            "retry_delay" : 0.5,
            "timeout" : 3,
            "frequency_step_THz" : 0.001,
            "max_attenuation" : 60, #50,
            "startfreq_default": 187.275,
            "stopfreq_default": 196.274
            }
        self.ip = config['ip']
        self.max_retries = config['max_retries']
        self.retry_delay = config['retry_delay']
        self.timeout     = config['timeout']
        self.frequency_step_THz = config['frequency_step_THz']
        self.max_attenuation = config['max_attenuation']
        self.startfreq_default = config['startfreq_default']
        self.stopfreq_default = config['stopfreq_default']
        
        # Query device frequency grid using the waveshaper RESTful Interface
        attempt = 0
        while attempt <= self.max_retries:
            try:
                url = f'http://{self.ip}/waveshaper/devinfo'
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
                self.deviceinfo = response.json()
                self.freq_start  = self.deviceinfo['startfreq']
                self.freq_end    = self.deviceinfo['stopfreq']
                self.info(self.devicename+f": Successfuly received device info from waveshaper model {self.deviceinfo['model']}")
                break  # Success, exit the loop
            except requests.exceptions.Timeout:
                self.info(self.devicename+": "+f"Attempt {attempt+1}: The request timed out. Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
            except requests.exceptions.HTTPError as err:
                self.error(self.devicename+": "+f"During connection, HTTP error occurred: {err}")
                break
            except requests.exceptions.RequestException as e:
                self.error(self.devicename+": "+f"An error occurred: {e}")
                break
            attempt += 1
        else:
            self.warning(self.devicename+": Max retries reached. Using the default frequency vector.")
            # freq_start  = self.deviceinfo['startfreq_default']
            # freq_end    = self.deviceinfo['stopfreq_default']
            self.freq_start = self.startfreq_default
            self.freq_end = self.stopfreq_default

        # self.wsFreq = np.linspace(freq_start, freq_end, int((freq_end - freq_start) / self.frequency_step_THz + 1))

    @property
    def MAX_ATTEN(self): # for code compatibility
        return self.max_attenuation
    
    def connect(self):
        print("WaveShaper through ip , no need to connect")
        pass
    
    def disconnect(self):
        print("WaveShaper through ip , no need to disconnect")
        pass

    @property
    def wsFreq(self):
        return np.linspace(self.freq_start, self.freq_end, int((self.freq_end - self.freq_start) / self.frequency_step_THz + 1))

    @property
    def config(self):
        return {'ip': self.ip,
                'max_retries': self.max_retries,
                'retry_delay': self.retry_delay,
                'timeout': self.timeout,
                'frequency_step_THz': self.frequency_step_THz,
                'max_attenuation': self.max_attenuation,
                'startfreq_default': self.startfreq_default,
                'stopfreq_default': self.stopfreq_default}

    @property
    def current_profile(self):
        """Get the current profile from the waveshaper.
        
            Returns:
                tuple: A tuple containing the frequency, 
                                            attenuation, 
                                            phase, 
                                            and port values
        """
        url = f'http://{self.ip}/waveshaper/getprofile'
        attempt = 0
        while attempt <= self.max_retries:
            try:
                response = requests.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    wspString = response.text
                    wsFreq, wsAttn, wsPhase, wsPort = self.decode_wsp_string(wspString)
                    return wsFreq, wsAttn, wsPhase, wsPort
                else:
                    self.info(self.devicename+": "+f"Get profile failed with status code: {response.status_code}")
                    return None
            except requests.exceptions.Timeout:
                self.info(self.devicename+": "+f"Attempt {attempt+1}: Connection to the Waveshaper timed out. Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
            except Exception as e:
                self.info(self.devicename+": "+f"An error occurred: {e}")
                return None
            attempt += 1
        else:
            self.error(self.devicename+": "+"Max retries reached. Get profile failed.")
            return None

    def create_wsp_string(self, wsAttn, wsPhase, wsPort):
        #creates the string that the waveshaper RESTful Interface can understand

        wsFreq      = self.wsFreq

        wsAttn[np.isnan(wsAttn)] = self.MAX_ATTEN
        wsAttn[wsAttn > self.MAX_ATTEN] = self.MAX_ATTEN
        wsAttn[wsAttn <= 0]        = 0
        wsPhase[np.isnan(wsPhase)] = 0
        wspString = ''

        for i in range(len(wsFreq)):
            
            wspString += f"{wsFreq[i]:.4f}\t{wsAttn[i]:.4f}\t{wsPhase[i]:.4f}\t{wsPort[i]}\n"
        return wspString
    
    def create_wsp_string_totest(self, wsAttn, wsPhase, wsPort):
        """Create the string that the waveshaper RESTful Interface can understand."""
        wsFreq = self.wsFreq

        # Ensure attenuation and phase values are within acceptable ranges
        wsAttn = np.clip(np.nan_to_num(wsAttn, nan=60), 0, self.config["max_attenuation"])
        wsPhase = np.nan_to_num(wsPhase, nan=0)

        wspString = "\n".join(
            f"{wsFreq[i]:.4f}\t{wsAttn[i]:.4f}\t{wsPhase[i]:.4f}\t{wsPort[i]}"
            for i in range(len(wsFreq))
        )
        return wspString

    def decode_wsp_string(self, wspString):
        """Decode the string from Waveshaper.
        Args:
            wspString (str): The string from the Waveshaper.
        Returns:
            tuple: A tuple containing the frequency, attenuation, phase, and port values
        """
        wspList = wspString.split("\n")
        wsFreq = []
        wsAttn = []
        wsPhase = []
        wsPort = []
        for line in wspList:
            if line:
                freq, attn, phase, port = line.split("\t")
                wsFreq.append(float(freq))
                wsAttn.append(float(attn))
                wsPhase.append(float(phase))
                wsPort.append(int(port))
        return wsFreq, wsAttn, wsPhase, wsPort


    def upload_profile(self, wsAttn, wsPhase, wsPort, plot=False):
        wspString = self.create_wsp_string(wsAttn, wsPhase, wsPort)
        data = {
            "type": "wsp",
            "wsp": wspString
        }
        jsonString = json.dumps(data)
        url = f'http://{self.ip}/waveshaper/loadprofile'

        # Plotting
        
        if plot:
            wsFreq      = self.wsFreq
            # Plotting frequency vs attenuation
            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.plot(wsFreq, wsAttn, label='Attenuation', color='blue')
            plt.xlabel('Frequency (THz)')
            plt.ylabel('Attenuation (dB)')
            plt.title('Frequency vs Attenuation')
            plt.grid(True)
            plt.legend()

            # Plotting frequency vs phase
            plt.subplot(1, 2, 2)
            plt.plot(wsFreq, wsPhase, label='Phase', color='red')
            plt.xlabel('Frequency (THz)')
            plt.ylabel('Phase (Degrees)')
            plt.title('Frequency vs Phase')
            plt.grid(True)
            plt.legend()

            plt.tight_layout()
            plt.show()

            # # To test the plot_profile function
            # self.plot_profile(wsFreq, wsAttn, wsPhase, wsPort)

        attempt = 0
        while attempt <= self.max_retries:
            try:
                response = requests.post(url, data=jsonString, timeout=self.timeout)
                if response.status_code == 200:
                    result = response.json()
                    # print(f"Upload succeeded: {result}")

                    from winsound import Beep
                    Beep(800, 500)

                    return result
                else:
                    print(f"Upload failed with status code: {response.status_code}")
                    return None
            except requests.exceptions.Timeout:
                print(f"Attempt {attempt+1}: Connection to the Waveshaper timed out. Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
            except Exception as e:
                print(f"An error occurred: {e}")
                return None
            attempt += 1
        else:
            print("Max retries reached. Upload failed.")
            return None
        


    def plot_current_profile(self, unit='THz'):
        """Plot the current profile of the waveshaper.
        Args:
            unit (str): The unit of the frequency values, either 'THz' or 'nm'.
        """
        wsFreq, wsAttn, wsPhase, wsPort = self.current_profile
        self.plot_profile(wsFreq, wsAttn, wsPhase, wsPort, unit)

    def reset(self, port):
        
        wsFreq  = self.wsFreq
        wsPhase = np.zeros(len(wsFreq))
        wsPort  = port * np.ones(len(wsFreq), dtype=int)
        wsAttn  = np.zeros(len(wsFreq))

        self.upload_profile(wsAttn, wsPhase, wsPort, plot = False)

    ### Maodong added 
    def plot_profile(self, wsFreq, wsAttn, wsPhase, wsPort, unit='THz'):
        """Plot the profile of the waveshaper for all ports in one figure.
        Args:
            wsFreq (array): The frequency values.
            wsAttn (array): The attenuation values.
            wsPhase (array): The phase values.
            wsPort (array): The port values.
            unit (str): The unit of the frequency values, either 'THz' or 'nm'.
        """
        freq = wsFreq
        # Convert frequency to desired unit
        # Validate the unit
        if unit.lower() not in ['thz', 'nm']:
            self.error(self.devicename+": "+f"Given plot unit: '{unit}' is invalid. Please use 'THz' or 'nm'. "+" Not plotting.")

        def get_plot_x(freq, unit):
            if unit.lower() == 'nm':
                plot_x = np.array([self.thz2nm(f) for f in freq])
                plot_x_label = "Wavelength (nm)"
            else:
                plot_x = freq
                plot_x_label = "Frequency (THz)"
            return plot_x, plot_x_label

        # Separate data by port
        port_data = self.separate_by_port(wsFreq, wsAttn, wsPhase, wsPort)

        # Create a new figure with two subplots
        plt.figure(figsize=(14, 6))

        # Plot frequency vs attenuation
        plt.subplot(1, 2, 1)
        for port, (freq, attn, _) in port_data.items():
            plot_x, plot_x_label = get_plot_x(freq, unit)
            plt.plot(plot_x, -attn, label=f'Port {port}', linestyle='-')
        plt.xlabel(plot_x_label)
        plt.ylabel('-Attenuation (dB)')
        plt.grid(True)
        plt.legend()

        # Plot frequency vs phase
        plt.subplot(1, 2, 2)
        for port, (freq, _, phase) in port_data.items():
            plot_x, plot_x_label = get_plot_x(freq, unit)
            plt.plot(plot_x, phase, label=f'Port {port}', linestyle='-')
        plt.xlabel(plot_x_label)
        plt.ylabel('Phase (Degrees)')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()


    def separate_by_port(self, wsFreq, wsAttn, wsPhase, wsPort):
        """Separate data by port.
        Args:
            wsFreq (array): The frequency values.
            wsAttn (array): The attenuation values.
            wsPhase (array): The phase values.
            wsPort (array): The port values.
        Returns:
            dict: A dictionary where keys are ports and values are tuples of (frequency, attenuation, phase).
        """
        port_data = {}
        unique_ports = np.unique(wsPort)

        for port in unique_ports:
            indices = [i for i, p in enumerate(wsPort) if p == port]
            port_data[port] = (
                np.array([wsFreq[i] for i in indices]).flatten(), 
                np.array([wsAttn[i] for i in indices]).flatten(),
                np.array([wsPhase[i] for i in indices]).flatten()
            )

        return port_data

  ################################################
  ##########  Methods for spectral shaping
  ################################################

    def flatten_comb(self, spectrum_X, spectrum_Y, target_intensity, waveshaper_port, plot = False, spectrum_unit='m'):

        # c = 299792458  # Speed of light in m/s
        # # Convert wavelength (m) to frequency (THz)
        # osaFreq = (c / (np.array(spectrum_X) )) / 1e12

        if spectrum_unit.lower() == 'nm':
            spectrum_X = self.nm2thz(np.array(spectrum_X))
        elif spectrum_unit.lower() == 'm':
            spectrum_X = np.array(spectrum_X) / 1e-9 # Convert m to nm
            return self.flatten_comb(spectrum_X, spectrum_Y, target_intensity, waveshaper_port, plot, spectrum_unit='nm')
        elif spectrum_unit.lower() == 'thz':
            spectrum_X = np.array(spectrum_X)
        else:
            self.error(self.devicename+": "+f"Given spectrum unit: '{spectrum_unit}' is invalid in flatten_comb. Please use 'THz' or 'nm'. ")
        
        # After this, make sure spectrum_X is in THz
        osaFreq = spectrum_X

        from scipy.interpolate import interp1d
        # Interpolate the OSA spectrum onto the waveshaper frequency grid. Assign NaN to points outside OSA spectrum
        interp_func     = interp1d(osaFreq, spectrum_Y, bounds_error=False, fill_value=np.nan)
        osaFreq_interp   = interp_func(self.wsFreq)
        
        # Identify and remove NaN values
        valid_indices = ~np.isnan(osaFreq_interp)
        maskFreq      = self.wsFreq[valid_indices]

        # Calculate attenuation needed to flatten the spectrum
        maskAttn = osaFreq_interp[valid_indices] - target_intensity
        if np.any(maskAttn < 0):
            self.warning(self.devicename+": "+"Some points required negative attenuation and were set to 0. Check target intensity levels.")
            # warnings.warn("Some points required negative attenuation and were set to 0. Check target intensity levels.")
        maskAttn[maskAttn < 0] = 0  # Set negative attenuation values to 0


        # Calculate the WS masks
        wsAttn = np.full(self.wsFreq.shape, self.max_attenuation)  # Default attenuation value
        wsPhase = np.zeros(self.wsFreq.shape)    # Phase is zero for all frequencies
        wsPort = np.full(self.wsFreq.shape, waveshaper_port)  # Port is constant for all frequencies

        # Ensure maskFreq and maskAttn are arrays without NaN values (from previous steps)
        # Calculate the index for minimum frequency difference
        print(type(maskFreq))
        print((maskFreq))
        idx_f_min = np.argmin(np.abs(np.array(self.wsFreq) - np.min(maskFreq)))

        # Calculate bandwidth in terms of array size
        BW = len(maskAttn)
        # Update the attenuation values in the waveshaper range
        wsAttn[idx_f_min:idx_f_min+BW] = maskAttn


        if plot:
            # Waveshaper frequency grid to wavelength
            # wsLambda = c / (self.waveshaper.wsFreq * 1e9) * 1e9  # Convert GHz to Hz, then to m, and finally to nm
            wsLambda = self.thz2nm(self.wsFreq)  # Convert THz to nm
            maskLambda = wsLambda[valid_indices]  # Corresponding wavelengths

            # Plotting
            plt.figure(figsize=(12, 6))
            # Plot original spectrum (wavelength vs amplitude)
            plt.plot(spectrum_X, spectrum_Y, label='Original Spectrum')
            # Plot the attenuation mask (converted to wavelength vs attenuation)
            # Invert maskAttn for visualization purposes (attenuation decreases the amplitude)
            plt.plot(maskLambda, -maskAttn, 'r--', label='Attenuation Mask')

            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Amplitude / Attenuation')
            plt.title('Spectrum and Attenuation Mask')
            plt.legend()
            plt.grid(True)
            plt.show()

        return self.wsFreq, wsAttn, wsPhase, wsPort

    def nm2thz(self, nm):
        return self.__nm2thz(nm)

    def thz2nm(self, thz):
        return self.__thz2nm(thz)

    def __nm2thz(self, nm):
        return waveshaper.c_const / nm / 1000

    def __thz2nm(self, thz):
        return waveshaper.c_const / thz / 1000
    
    #### Maodong added
    def get_2nd_disper_lambda(self, d2, center=1560, centerunit='nm'):
        # d2 in ps/nm
        if centerunit.lower() == 'nm':
            center = self.__nm2thz(center) # Always convert to THz
        beta2 = (waveshaper.c_const / center)**2 / (2*np.pi*waveshaper.c_const) * (d2 * 1e-3)
        return lambda thz: beta2 * ((thz - center) * 2*np.pi)**2 / 2

    def get_3rd_disper_lambda(self, d2, d3, center=1560, centerunit='nm'):
        # d2 in ps/nm, d3 in ps/nm^2
        if centerunit.lower() == 'nm':
            center = self.__nm2thz(center)
        beta2 = (waveshaper.c_const / center)**2 / (2*np.pi*waveshaper.c_const) * (d2 * 1e-3)
        beta3 = (waveshaper.c_const)**2 / (4*np.pi*np.pi*center**4)*(d3*1e-6)
        return lambda thz: beta2 * ((thz - center) * 2*np.pi)**2 / 2 + beta3 * ((thz - center) * 2*np.pi)**3 / 6
    
    def get_bandpass_lambda(self, center=192.175, span=0.1, unit='thz', passband_atten = 0):
        passband_atten = np.max([0, passband_atten])
        if unit.lower() == 'nm':
            startthz = self.__nm2thz(center + span/2)
            stopthz = self.__nm2thz(center - span/2)
            center = (startthz + stopthz) / 2 
            span = stopthz - startthz
        startf = center - span/2
        stopf = center + span/2
        return lambda thz: passband_atten if (thz>startf and thz<stopf) else self.MAX_ATTEN

    def set_bandpass(self, center=192.175, span=0.1, unit='thz', port=1, passband_atten=0):
        bandpass = self.get_bandpass_lambda(center=center, span=span, unit=unit, passband_atten=passband_atten)
        wsFreq = self.wsFreq
        wsAttn = np.array([bandpass(f) for f in wsFreq])
        wsPhase = np.zeros(len(wsFreq))
        wsPort = np.array([int(port) for _ in wsFreq])
        try:
            self.upload_profile(wsAttn, wsPhase, wsPort, plot=False)
        finally:
            return wsFreq, wsAttn, wsPhase, wsPort

            
    #### Maodong added finish
    
class Waveshaper(waveshaper): # For backwards compatibility with the old naming convention
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class WaveShaper(waveshaper): # For backwards compatibility with the old naming convention
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)