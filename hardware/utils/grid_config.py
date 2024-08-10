import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class GridConfig(dict):
    # Define keys for grid configuration, inherit dict to allow dictionary-like access, such as config['key']
    DEFAULTS = {
        "matrixsize_0": None,
        "matrixsize_1": None,
        "elem_width": None,
        "elem_height": None,
        "topleft_x": None,
        "topleft_y": None,
        "gap_x": None,
        "gap_y": None
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.DEFAULTS.items():
            self.setdefault(key, value)

    def keys(self):
        # Add the DEFAULTS keys to the keys of the dictionary
        return self.DEFAULTS.keys() | super().keys()
    
    def __getitem__(self, key):
        # Return the value if it exists, otherwise the default value
        try:
            return super().__getitem__(key)
        except KeyError:
            try:
                return self.DEFAULTS[key]
            except KeyError:
                raise KeyError(f"Key '{key}' is not a valid key. Valid keys are: {list(self.keys())}")
    
    def __call__(self, new_config):
        # Allow direct assignment with a dictionary or another GridConfig object
        self.update_config(new_config)
        return self

    def missing_keys(self):
        # Return the necessary keys that are still None
        return [key for key in self.DEFAULTS if self[key] is None]
    
    def has_missing_keys(self):
        # Check if all necessary keys have been set
        return not all(self[key] is not None for key in self.DEFAULTS)
    
    @property
    def clean_dict(self):
        # Return the entire configuration, merging defaults with user-defined values
        return {**self.DEFAULTS, **self}   

    def __repr__(self):
        # Return the string representation of the configuration
        return '\n'.join([f"{key}: {value}" for key, value in self.clean_dict.items()])
    
    @property
    def matrixsize_0(self):
        return self["matrixsize_0"]
    @matrixsize_0.setter
    def matrixsize_0(self, value):
        self["matrixsize_0"] = value
    @property
    def matrixsize_1(self):
        return self["matrixsize_1"]
    @matrixsize_1.setter
    def matrixsize_1(self, value):
        self["matrixsize_1"] = value
    @property
    def elem_width(self):
        return self["elem_width"]
    @elem_width.setter
    def elem_width(self, value):
        self["elem_width"] = value
    @property
    def elem_height(self):
        return self["elem_height"]
    @elem_height.setter
    def elem_height(self, value):
        self["elem_height"] = value
    @property
    def topleft_x(self):
        return self["topleft_x"]
    @topleft_x.setter
    def topleft_x(self, value):
        self["topleft_x"] = value
    @property
    def topleft_y(self):
        return self["topleft_y"]
    @topleft_y.setter
    def topleft_y(self, value):
        self["topleft_y"] = value
    @property
    def gap_x(self):
        return self["gap_x"]
    @gap_x.setter
    def gap_x(self, value):
        self["gap_x"] = value
    @property
    def gap_y(self):
        return self["gap_y"]
    @gap_y.setter
    def gap_y(self, value):
        self["gap_y"] = value

    # methods to plot images given the configuration
    # this code was adapted from systemcontroller.py(deprecated)
    def plot_rect_array(self, img, grid_config=None):
        if grid_config is None:
            grid_config = {}
            for key, value in self.items():
                grid_config[key] = value

        N           =   [grid_config["matrixsize_0"], grid_config["matrixsize_1"]]
        topleft     =   np.array([grid_config["topleft_x"], grid_config["topleft_y"]])
        elem_size   =   np.array([grid_config["elem_width"], grid_config["elem_height"]])
        gap         =   np.array([grid_config["gap_x"], grid_config["gap_y"]])

        fig, ax = plt.subplots()
        plt.imshow(img, cmap='gray')
        for n0 in range(N[0]):
            for n1 in range(N[1]):
                topleft_ij  = topleft + (elem_size + gap)*np.array([n1, n0])
                rect        = patches.Rectangle(topleft_ij, elem_size[0], elem_size[1], linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
        plt.show()

    # def plot_rect_array_nonuniform(self, img, grid_config=None):
    #     if grid_config is None:
    #         grid_config = {}
    #         for key, value in self.items():
    #             grid_config[key] = value

    #     N           =   [grid_config["matrixsize_0"], grid_config["matrixsize_1"]]
    #     topleft     =   np.array([grid_config["topleft_x"], grid_config["topleft_y"]])
    #     elem_size   =   np.array([grid_config["elem_width"], grid_config["elem_height"]])
    #     gap         =   np.array([grid_config["gap_x"], grid_config["gap_y"]])
    #     offset_x = np.array(grid_config["offset_x"]).T
    #     offset_y = np.array(grid_config["offset_y"]).T 

    #     fig, ax = plt.subplots()
    #     plt.imshow(img, cmap='gray')
    #     for n0 in range(N[0]):
    #         for n1 in range(N[1]):
    #             topleft_ij  = topleft + (elem_size + gap)*np.array([n1, n0])
    #             rect        = patches.Rectangle(topleft_ij + np.array([offset_x[n1,n0], offset_y[n1,n0]]), elem_size[0], elem_size[1], linewidth=1, edgecolor='r', facecolor='none')
    #             # print(np.array([offset_x[n1,n0], offset_y[n1,n0]]))
    #             ax.add_patch(rect)
    #     plt.show()

