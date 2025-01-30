import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
# create an animation of the normalized pattern images
import matplotlib.animation as animation
from matplotlib import rc
from IPython.display import HTML
import json
import warnings
# create union datatype of str and None
from typing import Union

def thz2nm(thz):
    return 299792458 / thz / 1000

# create a type represent Tuple of two integers
from typing import Tuple


class DataprocessSpeckleImaging:
    def __init__(self,
                 raw_dim: Union[Tuple[int, int], None] = None, # dimension of raw image (collected from camera)
                 target_topleft: Union[Tuple[int, int], None] = None, # top left corner (x,y) of targeted subimage (the region of interest)
                 target_dim: Union[Tuple[int, int], None] = None, # dimension of targeted subimage (the region of interest)
                 effective_dim: Union[Tuple[int, int], None] = None, # dimension of effective image (after smoothing and downsampling to lower resolution)
                #  brightness_threshold: float = 0.3 # threshold to determine if a pixel is bright
                 ):

        self.raw_dim = raw_dim
        self.__given_target_topleft = target_topleft # default (0, 0)
        self.__given_target_dim = target_dim # default raw_dim
        self.__given_effective_dim = effective_dim
        # self.brightness_threshold = brightness_threshold
        
        # img dataframe dimension: (num_pixels, num_images)
        self.raw_pat_img = None # will be updated by self.load_pattern_image()
        self.raw_obj_img = None # will be updated by self.load_object_image()

        # pd dataframe dimension: (1, num_images) 
        # (the first dimension is subject to change accroding to how you defined the callback function when collecting the data)
        self.raw_pat_callback = None # will be updated by self.load_pattern_callback()
        self.raw_obj_callback = None # will be updated by self.load_object_callback()

        self.scaled_speckle_pattern = None # will be updated by self.get_scaled_speckle_pattern_df()
        self.scaled_object_image = None # will be updated by self.get_scaled_object_image_df()

    @property
    def target_topleft(self):
        '''Assume don't chop the image if target_topleft is not given'''
        if self.__given_target_topleft is not None:
            return self.__given_target_topleft
        if self.raw_dim is None:
            raise ValueError("raw_dim not specified")
        return (self.raw_dim[0]-self.target_dim[0])//2, (self.raw_dim[1]-self.target_dim[1])//2
    
    @target_topleft.setter
    def target_topleft(self, value: Tuple[int, int]):
        self.__given_target_topleft = value

    @property
    def target_dim(self):
        '''Assume don't chop the image if target_dim is not given'''
        if self.__given_target_dim is not None:
            return self.__given_target_dim
        return self.raw_dim
    
    @target_dim.setter
    def target_dim(self, value: Tuple[int, int]):
        self.__given_target_dim = value

    @property
    def effective_dim(self):
        '''Assume don't downsample the image if effective_dim is not given'''
        if self.__given_effective_dim is not None:
            return self.__given_effective_dim
        return self.target_dim
    
    @effective_dim.setter
    def effective_dim(self, value: Tuple[int, int]):
        self.__given_effective_dim = value

    ######  Change how to get the raw pd data from the callback at here if needed ######
    @property
    def raw_pat_pd(self) -> pd.DataFrame:
        if self.raw_pat_callback is None:
            raise ValueError("Pattern callback not loaded yet")
        return self.raw_pat_callback
    
    @property
    def raw_obj_pd(self) -> pd.DataFrame:
        if self.raw_obj_callback is None:
            raise ValueError("Object callback not loaded yet")
        return self.raw_obj_callback
    ######  Change how to get the raw pd data from the callback at here if needed ######

    @property
    def num_raw_pixels(self):
        return self.raw_dim[0] * self.raw_dim[1]
    
    @property
    def num_target_pixels(self):
        return self.target_dim[0] * self.target_dim[1]
    
    @property
    def num_effective_pixels(self):
        return self.effective_dim[0] * self.effective_dim[1]
    
    # @property
    # def object_image_mask(self):
    #     if self.object_image is None:
    #         raise ValueError("Object image not get by get_object_image_df() yet")
    #     return self.get_bright_value_mask_from_image_df(self.object_image, self.brightness_threshold)
    
    @property
    def wavelength_nm(self):
        assert self.raw_pat_img is not None, "Pattern image not loaded yet"
        assert self.raw_obj_img is not None, "Object image not loaded yet"
        # assert self.raw_pat_img.columns == self.raw_obj_img.columns, "Pattern and object image wavelength does not match"
        assert (self.raw_pat_img.columns == self.raw_obj_img.columns).all(), "Pattern and object image wavelength does not match"
        return self.raw_pat_img.columns
    

    @property
    def speckle_pattern(self)->pd.DataFrame:
        '''
        dim: (num_images, num_effective_pixels)
        Return the speckle pattern used for image reconstruction
        
        each pattern is sum 1 normalized
        '''
        if self.scaled_speckle_pattern is None:
            raise ValueError("Speckle pattern not get by get_scaled_speckle_pattern_df() yet")
        subimg_df = self.get_subimage_df(self.scaled_speckle_pattern, self.target_topleft[0], self.target_topleft[1], self.raw_dim, self.target_dim)
        finalpat_df = self.get_lowerres_image_df(subimg_df, self.target_dim, self.effective_dim)

        speckle_pattern = finalpat_df / np.sum(finalpat_df, axis=0)
        return speckle_pattern.T
  
    @property
    def object_bright_mask(self)->np.ndarray:
        '''
        dim: (num_effective_pixels,)
        Return the mask of bright pixels in the object image
        
        each pixel is True if it is bright, False if it is dark
        '''
        if self.scaled_object_image is None:
            raise ValueError("Object image not get by get_scaled_object_image_df() yet")
        return self.get_bright_value_mask_from_image_df(self.scaled_object_image, 0.3).values
    @property
    def object_bright_mask_eff_dim(self)->np.ndarray:
        '''
        dim: (num_effective_pixels,)
        Return the mask of bright pixels in the object image
        
        each pixel is True if it is bright, False if it is dark
        '''
        return self.get_bright_value_mask_from_image_df(
                self.get_subimage_df(
                self.get_lowerres_image_df(self.scaled_object_image, self.target_dim, self.effective_dim),
                self.target_topleft[0], self.target_topleft[1], self.raw_dim, self.target_dim)
                , 1.2).values

    # Get the used pattern for data processing
    def get_scaled_speckle_pattern_df(self, preprocess_fun = None,
                                exp_scale: Union[float, None] = None,
                                fraction_to_ignore_min: Union[float, None] = None,
                                fraction_to_ignore_max: Union[float, None] = None,) -> pd.DataFrame:
        if self.raw_pat_img is None:
            raise ValueError("Pattern image not loaded yet")
        self.scaled_speckle_pattern = self.get_preprocessed_image_df(self.raw_pat_img, preprocess_fun, exp_scale, fraction_to_ignore_min, fraction_to_ignore_max)
        return self.scaled_speckle_pattern
    
    def get_scaled_object_image_df(self, preprocess_fun = None,
                            exp_scale: Union[float, None] = None,
                            fraction_to_ignore_min: Union[float, None] = None,
                            fraction_to_ignore_max: Union[float, None] = None,
                            ) -> pd.DataFrame:
        if self.raw_obj_img is None:
            raise ValueError("Object image not loaded yet")
        self.scaled_object_image = self.get_preprocessed_image_df(self.raw_obj_img, preprocess_fun, exp_scale, fraction_to_ignore_min, fraction_to_ignore_max)
        return self.scaled_object_image
    



    # get single pixel measurement for reconstruction
    def get_digital_sum_measurement(self,
                                    bright_threshold: float = 0.3,
                                    ) -> np.ndarray:
        if self.scaled_speckle_pattern is None:
            raise ValueError("Speckle pattern not get by get_scaled_speckle_pattern_df() yet")
        if self.scaled_object_image is None:
            raise ValueError("Object image not get by get_scaled_object_image_df() yet")
        bright_mask = self.get_bright_value_mask_from_image_df(self.scaled_object_image, bright_threshold)
        pattern_sum = np.sum(self.scaled_speckle_pattern.values * bright_mask.values[:,None], axis=0)
        # normalize by sum of pattern
        pattern_sum = pattern_sum / np.sum(self.scaled_speckle_pattern.values, axis=0)
        return pattern_sum
    
    def get_actual_pd_measurement(self,
                                  dark_voltage: float = 0.002,
                                  preprocess_pd = lambda x: x
                                  ) -> np.ndarray:
        if self.raw_pat_pd is None:
            raise ValueError("Pattern callback not loaded yet")
        if self.raw_obj_pd is None:
            raise ValueError("Object callback not loaded yet")
        # return preprocess_pd(((self.raw_obj_pd.values - dark_voltage) / (self.raw_pat_pd.values - dark_voltage)).flatten())

        # need to return the mean of columns in raw_obj_pd and raw_pat_pd
        return preprocess_pd(((self.raw_obj_pd.mean(axis=0).values - dark_voltage) / (self.raw_pat_pd.mean(axis=0).values - dark_voltage)).flatten())
        
    def plt_compare_digitalsum_and_actualpd(self,
                                            bright_threshold: float = 0.3,
                                            dark_voltage: float = 0.002,
                                            preprocess_pd = lambda x: x,
                                            ):
        digital_sum = self.get_digital_sum_measurement(bright_threshold)
        actual_pd = self.get_actual_pd_measurement(dark_voltage, preprocess_pd)
        plt.figure(figsize=(10,3))
        # plot to mean 1 for comparison
        plt.plot(self.wavelength_nm, digital_sum / np.mean(digital_sum), label="Digital Sum")
        plt.plot(self.wavelength_nm, actual_pd / np.mean(actual_pd), label="Actual PD")
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("(normalized to mean 1)")
        plt.legend()
        plt.show()





    # Image reconstruction
    def pinv_reconstruction(self, speckle_pattern, measurement):
        '''
        speckle_pattern: (num_images, num_effective_pixels)
        measurement: (num_images,)
        '''
        assert speckle_pattern.shape[0] == len(measurement), f"Number of images in speckle_pattern {speckle_pattern.shape[0]} does not match the length of measurement {len(measurement)}"
        return np.linalg.pinv(speckle_pattern) @ measurement
    
    def digital_reconstruction_pinv(self, 
                                    bright_threshold: float = 0.3,
                                    selected_index = None):
        if selected_index is None:
            # use all data
            selected_index = slice(None)
        digital_sum = self.get_digital_sum_measurement(bright_threshold)
        return self.pinv_reconstruction(self.speckle_pattern[selected_index], digital_sum[selected_index]).reshape(self.effective_dim)

    def actual_pd_reconstruction_pinv(self, dark_voltage: float = 0.0, selected_index = None, preprocess_pd = lambda x: x):
        if selected_index is None:
            # use all data
            selected_index = slice(None)
        actual_pd = self.get_actual_pd_measurement(dark_voltage, preprocess_pd)
        return self.pinv_reconstruction(self.speckle_pattern[selected_index], actual_pd[selected_index]).reshape(self.effective_dim)
    
    def post_process_reconstruction(self, img: np.ndarray,
                                    fraction_to_ignore_min: float = 0.02,
                                    fraction_to_ignore_max: float = 0.01,
                                    exp_scale: float = 4.0,
                                    )->np.ndarray:
        img = self.clip_image_extreme_values(img, fraction_to_ignore_min, fraction_to_ignore_max)

        img = self.linearize_dbscale_image(img, exp_scale)

        return img
    
    def rotate_image_clockwise(self, img: np.ndarray) -> np.ndarray:
        return np.array([x[::-1] for x in img.T])


    # Image data visualization helper functions
    def get_2darray_to_plot(self, img: Union[pd.Series, pd.DataFrame, np.ndarray],
                                    dim: Union[Tuple[int, int], None] = None) -> np.ndarray:
        if isinstance(img, pd.DataFrame):
            # have to be one column
            assert img.shape[1] == 1, "Input image dataframe has more than one column"
            img = img.iloc[:, 0].values
        elif isinstance(img, pd.Series):
            img = img.values
        else:
            # check less than 2 dimensions
            assert len(img.shape) < 2, "Input image has more than 1 dimension"
            img = img.flatten()

        if dim is None:
            # try to assume square root of the length of img, raise error if not a perfect square
            sqrt_len = np.sqrt(len(img))
            assert sqrt_len.is_integer(), "Dimension not specified, and length of input image is not a perfect square"
            dim = (int(sqrt_len), int(sqrt_len))

        assert len(img) == dim[0] * dim[1], "Input image length does not match the specified dimension"
        return img.reshape(dim)
    
    def get_i_th_image_from_df(self, img_df: pd.DataFrame, i: int, dim: Union[Tuple[int, int], None] = None) -> np.ndarray:
        return self.get_2darray_to_plot(img_df.iloc[:, i], dim)


    def plt_i_th_speckle_pattern(self, i: int, dim: Union[Tuple[int, int], None] = None):
        if self.scaled_speckle_pattern is None:
            raise ValueError("Speckle pattern not get by get_scaled_speckle_pattern_df() yet")
        plt.imshow(self.get_i_th_image_from_df(self.scaled_speckle_pattern, i, dim), cmap='gray')
        plt.title(f"The {i}th speckle pattern at {self.wavelength_nm[i]} nm")
        plt.show()
    
    def plt_i_th_object_image(self, i: int, dim: Union[Tuple[int, int], None] = None):
        if self.scaled_object_image is None:
            raise ValueError("Object image not get by get_scaled_object_image_df() yet")
        plt.imshow(self.get_i_th_image_from_df(self.scaled_object_image, i, dim), cmap='gray')
        plt.title(f"The {i}th object image at {self.wavelength_nm[i]} nm")
        # plt.show()

    def plt_i_th_speckle_pattern_raw(self, i: int, dim: Union[Tuple[int, int], None] = None):
        if self.raw_pat_img is None:
            raise ValueError("Pattern image not loaded yet")
        plt.imshow(self.get_i_th_image_from_df(self.raw_pat_img, i, dim), cmap='gray')
        plt.title(f"The {i}th raw speckle pattern at {self.wavelength_nm[i]} nm")
        # plt.show()

    def plt_i_th_object_image_raw(self, i: int, dim: Union[Tuple[int, int], None] = None):
        if self.raw_obj_img is None:
            raise ValueError("Object image not loaded yet")
        plt.imshow(self.get_i_th_image_from_df(self.raw_obj_img, i, dim), cmap='gray')
        plt.title(f"The {i}th raw object image at {self.wavelength_nm[i]} nm")
        plt.show()




    # pattern and object image consistency check functions
    def plt_i_th_pat_with_obj_overlay(self, i: Union[int, None] = None,
                                      pat_df: Union[pd.DataFrame, None] = None,
                                      objimg_df: Union[pd.DataFrame, None] = None,
                                      dim: Union[Tuple[int, int], None] = None,
                                      bright_threshold: float = 0.3):
        if pat_df is None:
            pat_df = self.scaled_speckle_pattern
        if objimg_df is None:
            objimg_df = self.scaled_object_image
        if i is None:
            _, i = self.get_min_masked_region_correlation(pat_df, objimg_df, dim, bright_threshold)
        if np.abs(i) >= pat_df.shape[1]:
            raise ValueError(f"i {i} exceeds the number of images {pat_df.shape[1]} in the pattern image dataframe")
        
        obj_mask = self.get_bright_value_mask_from_image_df(objimg_df, bright_threshold)

        print(f"The {i}th image with wavelength {pat_df.columns[i]} nm")
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(self.get_i_th_image_from_df(objimg_df, i, dim), cmap='jet', alpha = 1)
        ax[0].set_title("Object Image")

        ax[1].imshow(self.get_2darray_to_plot(obj_mask, dim), cmap='gray', alpha = 1)
        ax[1].imshow(self.get_i_th_image_from_df(pat_df, i, dim), cmap='jet', alpha = 0.5)

        # calculate correlation
        obj_mask = self.get_bright_value_mask_from_image_df(objimg_df, bright_threshold)
        corr = objimg_df[obj_mask].iloc[:, i].corr(pat_df[obj_mask].iloc[:, i])

        ax[1].set_title(f"Overlayed (Correlation: {corr:.2f})")
        
        plt.show()

    def get_masked_region_correlation(self, 
                                      pat_df: Union[pd.DataFrame, None] = None,
                                      objimg_df: Union[pd.DataFrame, None] = None,
                                      dim: Union[Tuple[int, int], None] = None,
                                      bright_threshold: float = 0.3):
        if pat_df is None:
            pat_df = self.scaled_speckle_pattern
        if objimg_df is None:
            objimg_df = self.scaled_object_image
        obj_mask = self.get_bright_value_mask_from_image_df(objimg_df, bright_threshold)
        obj_img_masked = objimg_df[obj_mask]
        pat_img_masked = pat_df[obj_mask]
        corr = obj_img_masked.corrwith(pat_img_masked, axis=0)
        return corr
    
    def plt_masked_region_correlation(self,
                                      pat_df: Union[pd.DataFrame, None] = None,
                                      objimg_df: Union[pd.DataFrame, None] = None,
                                      dim: Union[Tuple[int, int], None] = None,
                                      bright_threshold: float = 0.3):
        corr = self.get_masked_region_correlation(pat_df, objimg_df, dim, bright_threshold)
        plt.figure(figsize=(10,3))
        plt.plot(corr)
        plt.title('Correlation between object and pattern images in the masked region')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Correlation')
        plt.show()

    def get_min_masked_region_correlation(self, 
                                          pat_df: Union[pd.DataFrame, None] = None,
                                          objimg_df: Union[pd.DataFrame, None] = None,
                                          dim: Union[Tuple[int, int], None] = None,
                                          bright_threshold: float = 0.3):
        corr = self.get_masked_region_correlation(pat_df, objimg_df, dim, bright_threshold)
        min_corr = corr.idxmin()
        i = objimg_df.columns.get_loc(min_corr)
        return min_corr, i


    

    # pattern correlation check functions
    def get_pattern_correlation(self, pat_df: Union[pd.DataFrame, None] = None):
        if pat_df is None:
            pat_df = self.raw_pat_img
        corr = pat_df.corr()
        return corr
    
    def plt_pattern_correlation(self, pat_df: Union[pd.DataFrame, None] = None):
        corr = self.get_pattern_correlation(pat_df)
        plt.figure(figsize=(9,7))
        sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
        plt.title('Correlation between pattern images')
        plt.show()

    def plt_i_th_pat_correlation(self, i: int, pat_df: Union[pd.DataFrame, None] = None):
        if pat_df is None:
            pat_df = self.raw_pat_img
        corr = pat_df.corr()
        plt.figure(figsize=(10,3))
        plt.plot(corr.iloc[:, i])
        plt.title(f'Correlation of the {i}th pattern image with other pattern images')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Correlation')
        plt.show()





    # Image data dimension manipulation functions
    def get_subimage_df(self, img_df: pd.DataFrame, # (num_pixels, num_images)
                        top_left_x: int, top_left_y: int,
                        original_dim: Tuple[int, int],
                        target_dim: Tuple[int, int]) -> pd.DataFrame:
        if original_dim == target_dim:
            return img_df
        
        size_x, size_y = target_dim
        original_dim_x, original_dim_y = original_dim
        assert original_dim_x * original_dim_y == img_df.shape[0], f"Original dimension {original_dim} with {original_dim_x * original_dim_y} pixels does not match the image dataframe num of pixels {img_df.shape[0]}"
        assert top_left_x + size_x <= original_dim_x, f"top_left_x {top_left_x} + size_x {size_x} exceeds original_dim_x {original_dim_x}"
        assert top_left_y + size_y <= original_dim_y, f"top_left_y {top_left_y} + size_y {size_y} exceeds original_dim_y {original_dim_y}"

        result_df = pd.DataFrame(np.zeros((size_x * size_y, img_df.shape[1])))
        for i in range(img_df.shape[1]):
            image_array = img_df.iloc[:, i].values.reshape(original_dim)
            result_df.iloc[:, i] = image_array[top_left_y:top_left_y + size_y, top_left_x:top_left_x + size_x].reshape(-1)
        return result_df
    
    def get_lowerres_image_df(self, img_df: pd.DataFrame, # (num_pixels, num_images)
                                original_dim: Tuple[int, int],
                                target_dim: Tuple[int, int]) -> pd.DataFrame:
        if original_dim == target_dim:
            return img_df
        original_x_dim, original_y_dim = original_dim
        original_len = original_x_dim * original_y_dim
        assert len(img_df) == original_len
        target_x_dim, target_y_dim = target_dim
        target_len = target_x_dim * target_y_dim

        # get centered subimage if original_x_dim and original_y_dim are not multiples of target_x_dim and target_y_dim
        if original_x_dim % target_x_dim != 0 or original_y_dim % target_y_dim != 0:
            target_region_x_dim = target_x_dim*(original_x_dim//target_x_dim)
            target_region_y_dim = target_y_dim*(original_y_dim//target_y_dim)
            top_left_x = (original_x_dim - target_region_x_dim) // 2
            top_left_y = (original_y_dim - target_region_y_dim) // 2
            img_df = self.get_subimage_df(img_df, top_left_x, top_left_y, original_dim, (target_region_x_dim, target_region_y_dim))

        x_ratio = original_x_dim // target_x_dim
        y_ratio = original_y_dim // target_y_dim
        # First sum over x axis, which need sum every x_ratio rows
        df_sum_x = img_df.groupby(img_df.index // x_ratio).sum().reset_index(drop=True)
        # Then sum over y axis, which need sum rows that are y_ratio apart
        result = np.empty((target_len, img_df.shape[1]))
        final_df = pd.DataFrame(result)
        for i in range(target_y_dim):
            selected_rows = df_sum_x.iloc[i::target_x_dim].reset_index(drop=True)
            final_df.iloc[i::target_x_dim] = selected_rows.groupby(selected_rows.index // y_ratio).sum().reset_index(drop=True)

        return final_df




    # Image data preprocessing functions
    def get_preprocessed_image_df(self, img_df: pd.DataFrame,
                                  preprocess_fun = None,
                                  exp_scale: Union[float, None] = None,
                                  fraction_to_ignore_min: Union[float, None] = None,
                                  fraction_to_ignore_max: Union[float, None] = None,
                                  ) -> pd.DataFrame:
        fraction_to_ignore = None if (fraction_to_ignore_min is None and fraction_to_ignore_max is None) else (0 if fraction_to_ignore_min is None else fraction_to_ignore_min, 0 if fraction_to_ignore_max is None else fraction_to_ignore_max)
        if preprocess_fun is None:
            if exp_scale is None and fraction_to_ignore is None:
                preprocess_fun = self.preprocess_image
            elif fraction_to_ignore is not None:
                preprocess_fun = lambda img: self.preprocess_image(img, fraction_to_ignore_min=fraction_to_ignore[0], fraction_to_ignore_max=fraction_to_ignore[1])
            elif exp_scale is not None:
                preprocess_fun = lambda img: self.preprocess_image(img, exp_scale=exp_scale)
        else:
            assert exp_scale is None and fraction_to_ignore is None, "exp_scale and fraction_to_ignore should not be specified if preprocess_fun is given"
        return img_df.apply(preprocess_fun, axis=0).copy()

    def preprocess_image(self, img: np.ndarray, 
                         exp_scale: float = 2.0, 
                         fraction_to_ignore_min: float = 100/(256*256),
                         fraction_to_ignore_max: float = 100/(256*256) 
                         ) -> np.ndarray:
        img = self.clip_image_extreme_values(img, fraction_to_ignore_min, fraction_to_ignore_max)
        img = self.linearize_dbscale_image(img, exp_scale)
        return img
    
    def linearize_dbscale_image(self, img: np.ndarray, exp_scale: float = 2.0) -> np.ndarray:
        img = np.exp(img * exp_scale)
        img = img - np.min(img)
        img = img / np.max(img)
        return img

    def clip_image_extreme_values(self, img: np.ndarray, 
                                  fraction_to_ignore_min: float = 100/(256*256), # typically 100 pixels to ignore for 256x256 image
                                  fraction_to_ignore_max: float = 100/(256*256), 
                                  ) -> np.ndarray:
        len_img = len(img.flatten()) if isinstance(img, np.ndarray) else len(img)
        if not np.isclose(fraction_to_ignore_min, 0):
            img = img - np.partition(img.flatten() if isinstance(img, np.ndarray) else img
                                     , int(fraction_to_ignore_min*len_img))[int(fraction_to_ignore_min*len_img)]
        else:
            warnings.warn("fraction_to_ignore_min is 0, no minimum value clipping is done")
        if not np.isclose(fraction_to_ignore_max, 0):
            # return (np.partition(img, -int(fraction_to_ignore_max*len(img)))[-int(fraction_to_ignore_max*len(img))])
            img = img / np.partition(img.flatten() if isinstance(img, np.ndarray) else img
                                     , -int(fraction_to_ignore_max*len_img))[-int(fraction_to_ignore_max*len_img)]
        else:
            img = img / np.max(img.flatten() if isinstance(img, np.ndarray) else img)
            warnings.warn("fraction_to_ignore_max is 0, no maximum value clipping is done")
        img = np.clip(img, 0, 1)
        return img

    def get_bright_value_mask_from_image_df(self, img_df: pd.DataFrame, # (num_pixels, num_images)
                                             bright_threshold: float = 0.3) -> pd.Series:
        
        # return a mask, if the value of a pixel of any image is greater than bright_threshold, the pixel is considered bright (True)
        return (img_df > bright_threshold).any(axis=1)
    



    # Data loading functions
    def load_data_all(self, pattern_dir: str = 'dataset2_imaging_withoutobj_2MMFand1500DF', 
                      objimage_dir: str = 'dataset2_imaging_withoutobj_2MMFand1500DF',
                      pattern_filename_prefix: str = 'pattern_1527_1560_25GHz', 
                      objimage_filename_prefix: str = 'pattern_1527_1560_25GHz',
                      num_files: int = 0):
        start = time.time()
        print("Loading pattern image...")
        self.load_pattern_image(pattern_dir=pattern_dir, filename_prefix=pattern_filename_prefix, num_files=num_files)
        print("Pattern image loaded in", time.time()-start, "seconds")

        start_2 = time.time()
        print("Loading object image...")
        self.load_object_image(image_dir=objimage_dir, filename_prefix=objimage_filename_prefix, num_files=num_files)
        print("Object image loaded in", time.time()-start_2, "seconds")

        start_3 = time.time()
        print("Loading pattern callback...")
        self.load_pattern_callback(pattern_dir=pattern_dir, filename_prefix=pattern_filename_prefix, num_files=num_files)
        print("Pattern callback loaded in", time.time()-start_3, "seconds")

        start_4 = time.time()
        print("Loading object callback...")
        self.load_object_callback(image_dir=objimage_dir, filename_prefix=objimage_filename_prefix, num_files=num_files)
        print("Object callback loaded in", time.time()-start_4, "seconds")

        print("All data loaded in", time.time()-start, "seconds")

    def load_pattern_image(self, pattern_dir: str, filename_prefix: str, num_files: int):
        self.raw_pat_img = self.load_structured_csvfiles_from_dir(pattern_dir, filename_prefix=filename_prefix, num_files=num_files)
        if self.raw_dim is None:
            # try assign square root of the number of columns to raw_dim, raise error if not a perfect square
            print("raw_dim not specified, trying to assign square root of the number of columns to raw_dim")
            sqrt_num_cols = np.sqrt(self.raw_pat_img.shape[0])
            assert sqrt_num_cols.is_integer(), "Dim not specified, and Number of columns is not a perfect square"
            self.raw_dim = (int(sqrt_num_cols), int(sqrt_num_cols))
        assert self.raw_pat_img.shape[0] == self.num_raw_pixels, "Pattern image shape does not match the raw dimension"

    def load_object_image(self, image_dir: str, filename_prefix: str, num_files: int):
        self.raw_obj_img = self.load_structured_csvfiles_from_dir(image_dir, filename_prefix=filename_prefix, num_files=num_files)
        if self.raw_dim is None:
            # try assign square root of the number of columns to raw_dim, raise error if not a perfect square
            print("raw_dim not specified, trying to assign square root of the number of columns to raw_dim")
            sqrt_num_cols = np.sqrt(self.raw_obj_img.shape[0])
            assert sqrt_num_cols.is_integer(), "Dim not specified, and Number of columns is not a perfect square"
            self.raw_dim = (int(sqrt_num_cols), int(sqrt_num_cols))
        assert self.raw_obj_img.shape[0] == self.num_raw_pixels, "Object image shape does not match the raw dimension"

    def load_pattern_callback(self, pattern_dir: str, filename_prefix: str, num_files: int):
        self.raw_pat_callback = self.load_structured_csvfiles_from_dir(pattern_dir, filename_prefix=filename_prefix, filename_suffix="callback", num_files=num_files)
    
    def load_object_callback(self, image_dir: str, filename_prefix: str, num_files: int):
        self.raw_obj_callback = self.load_structured_csvfiles_from_dir(image_dir, filename_prefix=filename_prefix, filename_suffix="callback", num_files=num_files)

    def load_structured_csvfiles_from_dir(self, dir_path: str, 
                                       filename_prefix: str = "", 
                                       filename_suffix: str = "",
                                       num_files: int = 0) -> pd.DataFrame:
        """
        Load structured csv files from a directory, 
        with filename in the format of {filename_prefix}_{i+1}_{filename_suffix}.csv
        
        csv file column headers are assumed to be the THz frequencies

        returned df.columns are the wavelengths in nm (float)
        """
        if filename_suffix != "":
            filename_suffix = f"_{filename_suffix}"
        df = pd.concat([pd.read_csv( 
                                      os.path.join(dir_path, 
                                                    f"{filename_prefix}_{i+1}{filename_suffix}.csv")
                                    ) 
                                    for i in range(num_files)], 
                         axis=1)
        
        df.columns = [round(thz2nm(float(i)), 2) for i in df.columns]
        return df
    


    
    # def load_pattern_image(self):
    #     self.raw_pat_img = pd.concat([pd.read_csv(os.path.join(self.pattern_dir, f)) for f in self.filename_camera], axis=1)
    #     if self.raw_dim is None:
    #         # try assign square root of the number of columns to raw_dim, raise error if not a perfect square
    #         sqrt_num_cols = np.sqrt(self.raw_pat_img.shape[0])
    #         assert sqrt_num_cols.is_integer(), "Dim not specified, and Number of columns is not a perfect square"
    #         self.raw_dim = (int(sqrt_num_cols), int(sqrt_num_cols))
    #     assert self.raw_pat_img.shape[0] == self.num_raw_pixels, "Pattern image shape does not match the raw dimension"
                           
    # def load_object_image(self):
    #     self.raw_obj_img = pd.concat([pd.read_csv(os.path.join(self.image_dir, f)) for f in self.filename_camera], axis=1)
    #     if self.raw_dim is None:
    #         # try assign square root of the number of columns to raw_dim, raise error if not a perfect square
    #         sqrt_num_cols = np.sqrt(self.raw_obj_img.shape[0])
    #         assert sqrt_num_cols.is_integer(), "Dim not specified, and Number of columns is not a perfect square"
    #         self.raw_dim = (int(sqrt_num_cols), int(sqrt_num_cols))
    #     assert self.raw_obj_img.shape[0] == self.num_raw_pixels, "Object image shape does not match the raw dimension"

    # def load_pattern_callback(self):
    #     self.raw_pat_pd = pd.concat([pd.read_csv(os.path.join(self.pattern_dir, f)) for f in self.filename_callback], axis=1)
    
    # def load_object_callback(self):
    #     self.raw_obj_pd = pd.concat([pd.read_csv(os.path.join(self.image_dir, f)) for f in self.filename_callback], axis=1)
