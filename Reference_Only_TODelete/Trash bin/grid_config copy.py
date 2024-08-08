class GridConfig(dict):
    # Define keys for grid configuration
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

    def __init__(self, config=None):
        # Initialize with provided config or empty dictionary
        if config is None:
            config = {}
        self.config = config
        super().__init__(self.grid_config)

    def __getitem__(self, key):
        # Return the value if it exists, otherwise the default value
        try:
            return self.grid_config[key]
        except KeyError:
            raise KeyError(f"Key '{key}' is not a valid configuration key. Valid keys are: {list(self.grid_config.keys())}")

        # if key in self.config:
        #     return self.config[key]
        # elif key in self.DEFAULTS:
        #     return self.DEFAULTS[key]
        # else:
        #     raise KeyError(f"Key '{key}' is not a valid configuration key. Valid keys are: {list(self.grid_config.keys())}")

    def __setitem__(self, key, value):
        # Set the value for the specified key
        self.config[key] = value

    def __repr__(self):
        # Return the string representation of the configuration
        return '\n'.join([f"{key}: {value}" for key, value in self.grid_config.items()])

    @property
    def grid_config(self):
        # Return the entire configuration, merging defaults with user-defined values
        return {**self.DEFAULTS, **self.config}
    
    def update_config(self, new_config):
        # Update the configuration with a new dictionary
        if isinstance(new_config, dict):
            self.config.update(new_config)  # Merge the new dictionary with existing config
        elif isinstance(new_config, GridConfig):
            self.config.update(new_config.config)  # Merge from another GridConfig object
        else:
            raise TypeError("New configuration must be a dictionary or a GridConfig object.")
        print(f'self.config: {self.config}')

    def __call__(self, new_config):
        # Allow direct assignment with a dictionary or another GridConfig object
        self.update_config(new_config)
        return self

    def missing_keys(self):
        # Return the necessary keys that are still None
        return [key for key in self.DEFAULTS if self[key] is None]
    
    def has_needed_keys(self):
        # Check if all necessary keys have been set
        return ~all(self[key] is not None for key in self.DEFAULTS)
