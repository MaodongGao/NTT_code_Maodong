
'''
Handles Filenames and Directory name to save new data, 
Mainly renaming existing files or directories to avoid overwriting.
Also prepares the directory to save the file if it does not exist.
'''

import os
import shutil
import time
import warnings
from datetime import datetime

class DirnameHandler:
    def __init__(self, target):
        
        # Example usage:
        # target = r'C:\Users\user\Documents\ test'
        # DirnameHandler(target).prepare() # create new directory if not exist, rename old dir if already exist

        # This class dedicated to preparing target as a directory to save data.
        # Parameters
        # ----------
        # target : str
        #     The target directory name.
        
        self.target = target
    
    def prepare(self):
        '''
        Create dir if not exist, rename old dir if already exist.
        '''
        if not os.path.isdir(self.target):
            os.makedirs(self.target)
            return 'Directory %s does not exist. Created new directory.' % self.target
        else:
            now = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            new_name = f"{self.target}_{now}_bak"
            os.rename(self.target, new_name)
            return 'Directory %s already exists. Renamed previous directory to %s.' % (self.target, new_name)

class FilenameHandler:
    def __init__(self, target):
        
        # Example usage:
        # target = r'C:\Users\user\Documents\ test\ test.csv'
        # FilenameHandler(target).prepare() # return None if no action was taken, create new dir if not exist, rename old file if already exist, return message describing the action taken
        
        # Example usage 2:
        # target = r'C:\Users\user\Documents\test\test' # no extension
        # FilenameHandler(target).prepare('.csv') 

        # Example usage 3:
        # target = r'C:\Users\user\Documents\ test\ test.csv' # with extension
        # FilenameHandler(target).prepare('.csv') # argument '.csv' is unnecessary and will be ignored, the handled filename will be 'test.csv'

        # Example usage 4:
        # target = r'C:\Users\user\Documents\ test\ test.csv'
        # FilenameHandler(target).prepare('.json') # The handled filename will be 'test.csv.json'
    
        # This class dedicated to preparing target as a filename to save data.
        # Parameters
        # ----------
        # target : str
        #     The target file name.
        
        self.target = target

    @property
    def _target_without_extension(self):
        '''
        Returns
        -------
        str
            The target without the extension.
        '''
        return os.path.normpath(os.path.splitext(self.target)[0])

    @property
    def _target_extension(self):
        '''
        Returns
        -------
        str
            The target extension.
        '''
        return os.path.splitext(self.target)[1]

    @property
    def _target_contains_extension(self):
        '''
        Returns
        -------
        bool
            True if the target contains an extension, False otherwise.
        '''
        return bool(self._target_extension)

    def prepare(self, extension=None):
        '''
        Prepares the target file by creating directories and handling existing files.

        Parameters
        ----------
        extension : str, optional
            The extension of the file to prepare.
        '''
        msg = None
        flag_dir = self.prepare_directory(extension)
        flag_file = self.rename_existing_file(extension)

        if flag_dir is not None:
            msg = 'Directory %s does not exist. Created new directory.' % flag_dir
        if flag_file is not None:
            msg = 'Filename already exists. Renamed previous file to %s.' % flag_file

        return msg # return None if no action was taken

    def prepare_directory(self, extension=None):
        '''
        Prepare the directory to self.target if it does not exist.
        '''
        filename_to_handle = self._combine_name_and_extension(extension)
        filedir, single_filename = os.path.split(filename_to_handle)        
        if not os.path.isdir(filedir) and filedir != '':
            os.makedirs(filedir)
            return filedir
        return None
        
    def rename_existing_file(self, extension=None):
        '''
        Renames the target file if it already exists.

        Parameters
        ----------
        extension : str, optional
            The extension of the file to rename.
        '''
        filename_to_handle = self._combine_name_and_extension(extension)
        if os.path.isfile(filename_to_handle):
            new_filename = self._combine_name_and_timestamp(extension)
            os.rename(filename_to_handle, new_filename)
            return new_filename
        else:
            return None
    
    def _combine_name_and_timestamp(self, extension=None):
        '''
        Combines the target name with the current timestamp. Only used when need to rename existing file.

        Parameters
        ----------
        extension : str, optional
            The extension to combine with the target name.

        Returns
        -------
        str
            The combined target name and timestamp. This will be the full normalized filename to use.
        '''
        now = datetime.now().strftime('%Y%m%d_%H%M%S_%f') # example: 20210831_123456_123456
        filename_to_handle = self._combine_name_and_extension(extension)
        new_filename = os.path.splitext(filename_to_handle)[0] + '_' + now + os.path.splitext(filename_to_handle)[1]
        return new_filename

    def _combine_name_and_extension(self, extension):
        '''
        Combines the target name with the provided extension.

        Parameters
        ----------
        extension : str
            The extension to combine with the target name.

        Returns
        -------
        str
            The combined target name and extension. This will be the full normalized filename to use.
        '''
        if extension is not None and not extension.startswith('.'):
            extension = '.' + extension

        if extension is None and not self._target_contains_extension:
            # warnings.warn('No extension provided in target file name, will proceed without an extension.')
            return self.target ### IF AND ONLY IF HERE, THE RETURN VALUE DOES NOT CONTAIN AN EXTENSION
        elif extension is None and self._target_contains_extension:
            return self.target
        elif extension is not None and not self._target_contains_extension:
            return self.target + extension
        else: # handle the case where both are provided
            # check if the provided extension is the same as the existing extension
            if extension.casefold() == self._target_extension.casefold():
                return self.target
            else:
                warnings.warn('Provided extension %s does not match the filename extension %s.' % (extension, self._target_extension) +
                 'The filename extension %s will be ignored as part of the target file name.\n' % self._target_extension + 
                 'The proceeding filename will be %s.' % self.target + extension)
                return self.target + extension