from models.FNO import FNO2d
from models.GFNO import GFNO2d
from models.PTFNO import PTFNO2d
from models.RFNO import RFNO2d

class Model:
    def __init__(self, training_properties):
        if training_properties.which_model == "FNO":
            self.model = FNO2d(training_properties.modes1,
                                training_properties.modes2,
                                training_properties.width,
                                training_properties.layers,
                                training_properties.in_channel_dim,
                                training_properties.out_channel_dim,
                                training_properties.grid
                                )
        elif training_properties.which_model == "GFNO":
            self.model = GFNO2d(training_properties.modes1,
                                training_properties.modes2,
                                training_properties.width,
                                training_properties.layers,
                                training_properties.in_channel_dim,
                                training_properties.out_channel_dim,
                                training_properties.reflection,
                                training_properties.grid
                                )
        elif training_properties.which_model == "PTFNO":
            self.model = PTFNO2d(training_properties.modes1,
                                training_properties.modes2,
                                training_properties.width,
                                training_properties.layers,
                                training_properties.in_channel_dim,
                                training_properties.out_channel_dim,
                                training_properties.grid,
                                training_properties.in_size,
                                training_properties.polar_size
                                )
        elif training_properties.which_model == "RFNO":
            self.model = RFNO2d(training_properties.modes1,
                                training_properties.modes2,
                                training_properties.width,
                                training_properties.layers,
                                training_properties.in_channel_dim,
                                training_properties.out_channel_dim,
                                training_properties.reflection,
                                training_properties.grid
                                )
