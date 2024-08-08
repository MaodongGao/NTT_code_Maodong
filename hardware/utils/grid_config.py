
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

    

