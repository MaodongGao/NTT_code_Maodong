﻿# coding:utf-8

import ctypes
import time
import numpy as np
import os
import _slm_win as slm


FLAGS_COLOR_R    = 0x00000001
FLAGS_COLOR_G    = 0x00000002
FLAGS_COLOR_B    = 0x00000004
FLAGS_COLOR_GRAY = 0x00000008
FLAGS_RATE120     = 0x20000000

SLM_OK = 0
SLM_NG = 1
SLM_BS = 2
SLM_ER = 3
Rate120 = True

######################################
# ２Dグラデーション作成
######################################
def get_gradation_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T
    
    
def ChangeMode(SLMNumber, mode):
    
    print('ChangeMode',mode)
    slm.SLM_Ctrl_Open(SLMNumber)
    slm.SLM_Ctrl_WriteVI(SLMNumber,mode)     # 0:Memory 1:DVI
    slm.SLM_Ctrl_Close(SLMNumber)
    print('Done',mode)

    
######################################
# Test DVI mode
######################################
def Test_DVI_mode():
    DisplayNumber = 1

    
    width = ctypes.c_ushort(0)
    height = ctypes.c_ushort(0)
    DisplayName =  ctypes.create_string_buffer(64)
    
    
    # Search LCOS-SLM
    for DisplayNumber in range(1,8):        
        ret = slm.SLM_Disp_Info2(DisplayNumber, width, height, DisplayName)
        if(ret == SLM_OK):
            Names = DisplayName.value.decode('mbcs').split(',')
            if(Names[0] in 'LCOS-SLM'): # 'LCOS-SLM,SOC,8001,2018021001'
                print(Names, width, height)
                break

    if(DisplayNumber >= 8):
        print('No SLM')
        return
    

    if(Rate120):
        Flags = FLAGS_RATE120
    else:
        Flags = 0

    slm.SLM_Disp_Open(DisplayNumber)
    count = 0
    for i in range(100):
        gray = i * 10
        ret = slm.SLM_Disp_GrayScale(DisplayNumber, Flags, gray)
        if(ret != SLM_OK):
            print(ret,count)
            count += 1
        time.sleep(0.05)

    
    
    n = get_gradation_2d(0,1023,1920,1200,1)
    n1 = n.astype(np.ushort)
    
    n_h, n_w = n1.shape  # nのサイズを取得
    
    time.sleep(0.1)
    
    # =============================================================================
    # Horizontal scroll
    for i in range(50):
        n1 = np.roll(n1,10)
        c = n1.ctypes.data_as(ctypes.POINTER((ctypes.c_ushort * n_h) * n_w)).contents  # ctypesの3x4
        ret = slm.SLM_Disp_Data(DisplayNumber, n_w, n_h, Flags, c)
        if(ret != SLM_OK): print(ret)
        time.sleep(0.1)
    #    print(i)
    
    time.sleep(1)
    
    print('CSV File')
    ret = slm.SLM_Disp_ReadCSV(DisplayNumber,Flags,'C:\santec\SLM-200\Files\santec_logo.csv')
    print(ret)

    
    time.sleep(2)
    
    
    slm.SLM_Disp_Close(DisplayNumber)

######################################
# Test Memory mode
######################################
def Test_Memory_mode():
    SLMNumber = 1
    
    fol = os.getcwd() 
    fol += '\\img\\'
    setwl = 0
    
    slm.SLM_Ctrl_Close(SLMNumber)
    
    ret = slm.SLM_Ctrl_Open(SLMNumber)
    if(ret != SLM_OK):
        print(ret)
    else:
        dat16 = ctypes.c_uint16(0)
        dat32_1 = ctypes.c_uint32(0)
        dat32_2 = ctypes.c_uint32(0)
            
        
        for i in range(60):
            ret = slm.SLM_Ctrl_ReadSU(SLMNumber)
            print(ret)
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
    
        ret = slm.SLM_Ctrl_ReadWL(SLMNumber, dat32_1, dat32_2)
        print('WL {0} {1:.2f}'.format(dat32_1.value, dat32_2.value/100))
        slm.SLM_Ctrl_WriteGS(SLMNumber,100)
        slm.SLM_Ctrl_ReadGS(SLMNumber,dat16)
        
        print("GS", dat16.value)
        
        # (1) send data
        for no in range(1,31):
            fn = '{0}{1:03d}.png'.format(fol,no)
            print(fn)
            flags = 0
            ret = slm.SLM_Ctrl_WriteMI_BMP(SLMNumber,no,flags,fn)
            if(ret != SLM_OK):
                print(ret)
                break
        
        # change order
        ret = slm.SLM_Ctrl_WriteMT(SLMNumber,11,15)
        ret = slm.SLM_Ctrl_WriteMT(SLMNumber,12,20)
        ret = slm.SLM_Ctrl_WriteMT(SLMNumber,13,25)
        ret = slm.SLM_Ctrl_WriteMT(SLMNumber,14,30)

        # change range
        ret = slm.SLM_Ctrl_WriteMR(SLMNumber,10,14)


        # change interval 0.5s
        if(Rate120):
            slm.SLM_Ctrl_WriteMW(SLMNumber,60)
        else:
            slm.SLM_Ctrl_WriteMW(SLMNumber,30)

        # start position
        slm.SLM_Ctrl_WriteMP(SLMNumber,13)

        # start 
        slm.SLM_Ctrl_WriteDR(SLMNumber,0)
        
        dT = ctypes.c_uint32(0)
        oT = ctypes.c_uint32(0)
        dE = ctypes.c_uint32(0)
        oE = ctypes.c_uint32(0)
        dID = ctypes.create_string_buffer(10)
        oID = ctypes.create_string_buffer(10)
        for i in range(50):
            slm.SLM_Ctrl_ReadT(SLMNumber,dT,oT)
            #slm.SLM_Ctrl_ReadEDO(SLMNumber,dE,oE)
            #slm.SLM_Ctrl_ReadSDO(SLMNumber,dID,oID)
            print('{0}:{1:.1f} {2:.1f} {3} {4} {5} {6}'.format(i,dT.value/10, oT.value/10 , dE.value, oE.value, dID.value, oID.value))
            time.sleep(0.1)
            

        # stop
        slm.SLM_Ctrl_WriteDB(SLMNumber)
    

    
        # close
        slm.SLM_Ctrl_Close(SLMNumber)
    
    
###############################################################################
# Main
###############################################################################
def main():
    ChangeMode(1,1)

    Test_DVI_mode()
    Test_Memory_mode()
    
        
if __name__ == '__main__': 
    
    print('start')
    main()
    print('end')
  
    