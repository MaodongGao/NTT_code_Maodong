# coding:utf-8
import importlib.util
import ctypes
import time
import numpy as np
import os
import _slm_win as slm
import matplotlib.pyplot as plt


SLM_OK = 0
SLM_NG = 1
SLM_BS = 2
SLM_ER = 3
Rate120 = True

   
def Upload_csvfile_Memory_mode(CSVfilename, SLMNumber):
    #SLMNumber = 1
    MemoryNumber = 1
    setwl = 0
    
    slm.SLM_Ctrl_Close(SLMNumber)
    
    ret = slm.SLM_Ctrl_Open(SLMNumber)
    if(ret == SLM_OK):
        #print(ret)
       
        for i in range(60):
            ret = slm.SLM_Ctrl_ReadSU(SLMNumber)
           # print(ret)
            if(ret == SLM_OK):
                break
            else:
                time.sleep(1)

        # set mode
        ret = slm.SLM_Ctrl_WriteVI(SLMNumber,0)     # MemoryMode mode = 0
        
        if setwl == 1:
            # set wavelength
            ret = slm.SLM_Ctrl_WriteWL(SLMNumber,1500,200)
            
            #save wavelength
            ret = slm.SLM_Ctrl_WriteAW(SLMNumber)
    
        slm.SLM_Ctrl_WriteGS(SLMNumber,480)
            # (1) send data

        str1 = ctypes.c_char_p(CSVfilename.encode('utf-8'))
        ret = slm.SLM_Ctrl_WriteMI_CSV_A(SLMNumber, MemoryNumber,0,str1)
        if(ret != SLM_OK):
                print('OK')

        # start 
        slm.SLM_Ctrl_WriteDS(SLMNumber,MemoryNumber)
        
   
        # close
        slm.SLM_Ctrl_Close(SLMNumber)
    

def generate_SLM_csv_file_from_phasematrix(filename, M):
    with open(filename, 'w', encoding='cp1252') as file:
        file.write("Y/X," + ','.join(map(str, range(1920))) + '\n')
        for idx, row in enumerate(M):
            line = str(idx) + ',' + ','.join(map(str, row.astype(int))) + '\n'
            file.write(line)

    
def slmmask_phasematrix(W, SLMNumber):
    if np.max(W) > 1023  or np.min(W) < 450:
        raise ValueError('Illegal elements (outside 450-1023 range) detected.')
    
    ZeroLevelPhase = 450

    if SLMNumber == 3:
     ## 5x5, 
        Yshift   = -30
        nx_guard = 11
        nx_data  = 18
        ny_guard = 2
        ny_data  = 16

    elif SLMNumber == 2:
     ## 5x5
        Yshift   = -6
        nx_guard = 4
        nx_data  = 20
        ny_guard = 7
        ny_data  = 18


    phase_matrix = W
    a0 = ZeroLevelPhase * np.ones(((phase_matrix.shape[0]) * ny_data + (phase_matrix.shape[0] - 1) * ny_guard,
                   (phase_matrix.shape[1]) * nx_data + (phase_matrix.shape[1] - 1) * nx_guard))

    for idx in range(phase_matrix.shape[0]):
        for jdx in range(phase_matrix.shape[1]):
            a0[ny_data * idx + ny_guard * idx: ny_data * (idx + 1) + ny_guard * idx,
               nx_data * jdx + nx_guard * jdx: nx_data * (jdx + 1) + nx_guard * jdx] = phase_matrix[idx, jdx]

    ny, nx = a0.shape

    a1Lu = ZeroLevelPhase * np.ones((int(np.ceil((1200 - ny) / 2)) - Yshift, int(np.ceil((1920 - nx) / 2))))
    a1Ru = ZeroLevelPhase * np.ones((int(np.ceil((1200 - ny) / 2)) - Yshift, int(np.floor((1920 - nx) / 2))))
    a1Lb = ZeroLevelPhase * np.ones((int(np.floor((1200 - ny) / 2)) + Yshift, int(np.ceil((1920 - nx) / 2))))
    a1Rb = ZeroLevelPhase * np.ones((int(np.floor((1200 - ny) / 2)) + Yshift, int(np.floor((1920 - nx) / 2))))
    a2u = ZeroLevelPhase * np.ones((int(np.ceil((1200 - ny) / 2)) - Yshift, nx))
    a2b = ZeroLevelPhase * np.ones((int(np.floor((1200 - ny) / 2)) + Yshift, nx))
    a3L = ZeroLevelPhase * np.ones((ny, int(np.ceil((1920 - nx) / 2))))
    a3R = ZeroLevelPhase * np.ones((ny, int(np.floor((1920 - nx) / 2))))

    A = np.concatenate((np.concatenate((a1Lu, a2u, a1Ru), axis=1),
                    np.concatenate((a3L, a0, a3R), axis=1),
                    np.concatenate((a1Lb, a2b, a1Rb), axis=1)), axis=0)
    time.sleep(0.5)


    CSVfile = 'D:\\SLM_DLL_ver.2.4\\sample\\pattern.csv'
    generate_SLM_csv_file_from_phasematrix(CSVfile, A)
    Upload_csvfile_Memory_mode(CSVfile, SLMNumber)
    return A

def slmmask_grayscreen(phase, SLMNumber):
    A = phase * np.ones((1200,1920))
    CSVfile = 'D:\\SLM_DLL_ver.2.4\\sample\\pattern.csv'
    generate_SLM_csv_file_from_phasematrix(CSVfile, A)
    Upload_csvfile_Memory_mode(CSVfile, SLMNumber)
    return A

def ChangeMode(SLMNumber, mode):
    
    if(mode == 0):
        print('Change Memory Mode ',mode)
    elif(mode == 1):
        print('Change DVI Mode',mode)
    else:
        print('No Mode',mode)
        return
    ret = slm.SLM_Ctrl_Open(SLMNumber)
    time.sleep(0.5)
    ret = slm.SLM_Ctrl_WriteVI(SLMNumber,mode)     # 0:Memory 1:DVI
    print('Done',mode)
    return ret

###############################################################################
# Main
###############################################################################
def main():   
    SLMNumber = 2 
    ret = ChangeMode(SLMNumber,0)     # MemoryMode mode = 0
    # set wavelength
    ret = slm.SLM_Ctrl_WriteWL(SLMNumber,1550,100)
    print(ret)        
    #save wavelength
    ret = slm.SLM_Ctrl_WriteAW(SLMNumber)
    print(ret)
    slmmask_grayscreen(430, SLMNumber)
if __name__ == "__main__":
    main()