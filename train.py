import torch.nn.functional as F
from timeit import default_timer
import sys
import random
import os

import torch
import numpy as np
from tqdm import tqdm
from utils.utilities3 import *
from settings.properties import Training_Properties
from settings.data_module import *
from settings.model_module import *

######### Get cmd arguments ##########

import argparse
import sys

# Set up the argument parser
parser = argparse.ArgumentParser(description='Train models for various examples.')
parser.add_argument('--which_example', type=str, required=True, choices=["darcy"],
                    help="Specify which example to run (e.g., darcy)")
parser.add_argument('--which_model', type=str, required=True, choices=["FNO", "GFNO", "PTFNO", "RFNO"],
                    help="Specify which model to use (e.g., FNO, GFNO, PTFNO, RFNO)")
parser.add_argument('--random_seed', type=int, required=True, help="Specify the random seed (e.g., 0)")
# Parse the arguments
args = parser.parse_args()
# Save the models based on the input arguments

which_example, which_model, random_seed = args.which_example, args.which_model, args.random_seed
folder = f"TrainedModels/{args.which_model}_{args.which_example}_seed_{args.random_seed}"
######### Set up seeds, device, and result folders ##########
torch.manual_seed(random_seed)
np.random.seed(random_seed)
# random.seed(int(random_seed))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if str(device) == 'cpu':
    print("------------------------------------------")
    print("YOU ARE RUNNING THE CODE ON A CPU.")
    print("WE SUGGEST YOU TO RUN THE CODE ON A GPU!")
    print("------------------------------------------")
    print(" ")

if not os.path.isdir("TrainedModels"):
    print("Generated new folder TrainedModels")
    os.mkdir("TrainedModels")
if not os.path.isdir(folder):
    print("Generated new folder")
    os.mkdir(folder)
######### Get training properties, data, and model from settings ##########
t_properties = Training_Properties(which_example, which_model)
t_properties.save_as_txt(folder + "/training_properties.txt")
data = Darcy(t_properties)
model = Model(t_properties).model.cuda()
n_params = model.print_size()
######### Get dataloaders ##########
y_normalizer = data.y_normalizer
train_loader = data.train_loader
test_loader = data.test_loader
y_normalizer.cuda()
######### Training settings ##########
learning_rate = t_properties.learning_rate
iterations = t_properties.iterations
epochs = t_properties.epochs

######### For calculating errors ##########
batch_size = t_properties.batch_size
s = t_properties.s
ntrain = t_properties.ntrain
ntest = t_properties.ntest


################################################################
# training and evaluation
################################################################
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
p = 2
if p == 1:
    myloss = LpLoss(p=1, size_average=False)
elif p == 2:
    myloss = LpLoss(p=2, size_average=False)

f = open(folder + "/record_l2_losses.txt", "w")  # "w" mode empties the file
f = open(folder + "/record_l2_losses.txt", "a")

for epoch in range(epochs):
    with tqdm(unit="batch") as tepoch:
        model.train()
        tepoch.set_description(f"Epoch {epoch}")
        train_l2 = 0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            
            out = model(x).reshape(batch_size, s, s)
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

            loss = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_l2 += loss.item()
            
            tepoch.set_postfix({'Batch': step + 1, 'Train loss (in progress)': loss.item() / x.shape[0]})
            tepoch.update(1)

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()

                out = model(x).reshape(batch_size, s, s)
                out = y_normalizer.decode(out)

                test_l2 += myloss(out.view(batch_size, -1), y.view(batch_size, -1)).item()

        train_l2 /= ntrain
        test_l2 /= ntest
        
        tepoch.set_postfix({
            "Relative Train": train_l2 *100,
            "Relative Test": test_l2 * 100
        })

    if epoch % 50 == 0 or epoch == epochs-1:
        f.write(f"Epoch {epoch}, Train L2: {train_l2}, Test L2: {test_l2}\n")
f.close()

torch.save(model.state_dict(), folder + "/model.pth")

        
        
