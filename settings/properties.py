class Training_Properties:
    def __init__(self, which_example, which_model):
        self.which_example = which_example
        self.which_model = which_model
        ######## Data Related ########
        if which_example == "darcy":
            self.ntrain = 1000
            self.ntest = 100
            self.h = 241
            self.r = 5
            self.s = int(((241 - 1)/self.r) + 1)   
            self.in_channel_dim = 1
            self.out_channel_dim = 1
        ######## Model and Training Related ########
        if which_model == "FNO":
            self.batch_size = 100
            self.learning_rate = 0.001
            self.epochs = 1000
            self.iterations = self.epochs*(self.ntrain//self.batch_size)
            self.modes1 = 12
            self.modes2 = 12
            self.width = 32
            self.layers = 4
            self.grid = True
        elif which_model == "GFNO":
            self.batch_size = 100
            self.learning_rate = 0.001
            self.epochs = 1000
            self.iterations = self.epochs*(self.ntrain//self.batch_size)
            self.modes1 = 12
            self.modes2 = 12
            self.width = 16          # offset |G| x parameters, |G|^2 computations, width/2, parameters/2^2 
            self.layers = 4
            self.reflection = 0      # 0: SO2 group  1: O2 group
            self.grid = "symmetric"  # grid type: "cartesian", "symmetric", or "None"
        elif which_model == "PTFNO":
            self.batch_size = 100
            self.learning_rate = 0.001
            self.epochs = 1000
            self.iterations = self.epochs*(self.ntrain//self.batch_size)
            self.modes1 = 7
            self.modes2 = 20
            self.width = 32
            self.layers = 4
            self.grid = False
            self.in_size = [49,49]
            self.polar_size = [20,120]
        elif which_model == "RFNO":
            self.batch_size = 100
            self.learning_rate = 0.001
            self.epochs = 1000
            self.iterations = self.epochs*(self.ntrain//self.batch_size)
            self.modes1 = 12
            self.modes2 = 12
            self.width = 64
            self.layers = 4
            self.reflection = 0     # 0: SO2 group  1: O2 group
            self.grid = "symmetric" # grid type: "cartesian", "symmetric", or "None"

        
    def save_as_txt(self, file_path):
        with open(file_path, 'w') as fi:
            for attr, value in self.__dict__.items():
                fi.write(f"{attr} = {value}\n")