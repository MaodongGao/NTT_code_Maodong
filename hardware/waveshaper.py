import time
import requests
import numpy as np
import matplotlib.pyplot as plt
import json

class waveshaper:
    def __init__(self, config):
        self.ip = config['ip']
        self.max_retries = config['max_retries']
        self.retry_delay = config['retry_delay']
        self.timeout     = config['timeout']
        self.frequency_step_THz = config['frequency_step_THz']
        
        # Query device frequency grid using the waveshaper RESTful Interface
        attempt = 0
        while attempt <= self.max_retries:
            try:
                url = f'http://{self.ip}/waveshaper/devinfo'
                response = requests.get(url, str(self.timeout))
                response.raise_for_status()
                self.deviceinfo = response.json()
                freq_start  = self.deviceinfo['startfreq']
                freq_end    = self.deviceinfo['stopfreq']
                print(f"Successfuly received device info from waveshaper model {self.deviceinfo['model']}")
                break  # Success, exit the loop
            except requests.exceptions.Timeout:
                print(f"Attempt {attempt+1}: The request timed out. Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
            except requests.exceptions.HTTPError as err:
                print(f"HTTP error occurred: {err}")
                break
            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {e}")
                break
            attempt += 1
        else:
            print("Max retries reached. Using the default frequency vector.")
            freq_start  = self.deviceinfo['startfreq_default']
            freq_end    = self.deviceinfo['stopfreq_default']

        self.wsFreq = np.linspace(freq_start, freq_end, int((freq_end - freq_start) / self.frequency_step_THz + 1))

    def create_wsp_string(self, wsAttn, wsPhase, wsPort):
        #creates the string that the waveshaper RESTful Interface can understand

        wsFreq      = self.wsFreq

        wsAttn[np.isnan(wsAttn)]   = 60
        wsAttn[wsAttn > 60]        = 60
        wsAttn[wsAttn <= 0]        = 0
        wsPhase[np.isnan(wsPhase)] = 0
        wspString = ''

        for i in range(len(wsFreq)):
            
            wspString += f"{wsFreq[i]:.4f}\t{wsAttn[i]:.4f}\t{wsPhase[i]:.4f}\t{wsPort[i]}\n"
        return wspString

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

        attempt = 0
        while attempt <= self.max_retries:
            try:
                response = requests.post(url, data=jsonString, timeout=self.timeout)
                if response.status_code == 200:
                    result = response.json()
                    print(f"Upload succeeded: {result}")
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


    def reset(self, port):
        
        wsFreq  = self.wsFreq
        wsPhase = np.zeros(len(wsFreq))
        wsPort  = port * np.ones(len(wsFreq), dtype=int)
        wsAttn  = np.zeros(len(wsFreq))

        self.upload_profile(wsAttn, wsPhase, wsPort, plot = False)