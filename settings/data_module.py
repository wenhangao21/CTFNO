from utils.utilities3 import *

class Darcy:
    def __init__(self, training_properties):
        ntrain = training_properties.ntrain
        ntest = training_properties.ntest
        batch_size = training_properties.batch_size
        r = training_properties.r
        s = training_properties.s
        
        TRAIN_PATH = './data/darcy_full_1000.mat'
        TEST_PATH = './data/darcy_test.mat'
        ################################################################
        # load data and data normalization
        ################################################################
        reader = MatReader(TRAIN_PATH)
        x_train = reader.read_field('coef')[:ntrain,::r,::r][:,:s,:s]
        y_train = reader.read_field('sol')[:ntrain,::r,::r][:,:s,:s]

        reader.load_file(TEST_PATH)
        x_test = reader.read_field('coef')[:ntest,::r,::r][:,:s,:s]
        y_test = reader.read_field('sol')[:ntest,::r,::r][:,:s,:s]


        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)

        self.y_normalizer = UnitGaussianNormalizer(y_train)
        y_train = self.y_normalizer.encode(y_train)

        x_train = x_train.reshape(ntrain,1,s,s)
        x_test = x_test.reshape(ntest,1,s,s)
        y_train = y_train.reshape(ntrain,1,s,s)
        y_test = y_test.reshape(ntest,1,s,s)

        self.train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)

