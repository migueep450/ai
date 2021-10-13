#SUMMARY
#IMPUT:
#OUTPUT:

#MODULES NEEDED
import numpy as np
from numpy import random
import pandas as pd

from sklearn.model_selection import KFold
import torch.utils.data
import matplotlib
import os
import sys
matplotlib.use('Agg')
import multiprocessing as mp

import aitac
import plot_utils
import time

#FUNCTIONS (2): -
def benchmark_models(x, y, peak_names, output_file_path, split):
    """
    Helper function to benchmark models --> it executes the training and the testing of the model with a part of our data
    X : array
    y : array
    split : tuple
    Training and test indices (split[train], split[test])
    """
    print('dentro')
    
    torch.set_num_threads(1) # We put the number of CPUs which are going to be needed in this process(inside each step of cross validation)
    
    #split the data
    train = 0
    test = 1
    
    train_data, eval_data = x[split[train], :, :], x[split[test], :, :]
    train_labels, eval_labels = y[split[train], :], y[split[test], :]
    train_names, eval_name = peak_names[split[train]], peak_names[split[test]]
    
    #Data loader
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels))
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    eval_dataset = torch.utils.data.TensorDataset(torch.from_numpy(eval_data), torch.from_numpy(eval_labels))
    eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    
    
    # create model 
    model = aitac.ConvNet(num_classes, num_filters).to(device)
    
    # Loss and optimizer
    criterion = aitac.pearson_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # train model
    model, best_loss = aitac.train_model(train_loader, eval_loader, model, device, criterion,  optimizer, num_epochs, output_file_path)
    
    # Predict on test set
    predictions, max_activations, max_act_index = aitac.test_model(eval_loader, model, device)
    
    # plot the correlations histogram
    correlations = plot_utils.plot_cors(eval_labels, predictions, output_file_path)
    
    results = [predictions, correlations, eval_name]
    
    return results

def log_result(results):
    '''
    Take the results from the different cross validation process and put them together for the next step
    '''
    predictions_all.append(results[0])
    correlations_all.append(results[1])
    peak_order_all.append(results[2])

###############################################################################################################################3

# Hyper parameters
num_epochs = 10
num_classes = 141
batch_size = 10
learning_rate = 0.001
num_filters = 300
run_num = 'second'

#create output directory
output_file_path = "../outputs/valid10x10/" + run_num +'/'
directory = os.path.dirname(output_file_path)
if not os.path.exists(directory):
    print("Creating directory %s" % output_file_path)
    os.makedirs(directory)
else:
     print("Directory %s exists" % output_file_path)

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load all data
x = np.load('../BRCA_data/mini_sample_one_hot_seqs.npy')
x = x.astype(np.float32)
y = np.load('../BRCA_data/mini_sample_cell_type_array.npy')
y = y.astype(np.float32)
peak_names = np.load('../BRCA_data/mini_sample_peak_names.npy')

#Create the folds_index for the cross validation
splitter = KFold(n_splits=10, shuffle=True, random_state = 123) #creamos los cachos que van a ir separados
folds = list(splitter.split(x, y))

#Variables in which the output results is going to be store
predictions_all = []
correlations_all = []
peak_order_all = []

#Preparing the parallelization for the different steps of cross validation
# Python can count the available cores for you in most cases: mp.cpu_count()
# We introduce the number of steps that are going to be done at the same moment
pool = mp.Pool(processes=1)

start_time = time.time()

for fold in range(1): # We do the k-fold cross validation process
    pool.apply_async(benchmark_models, args=(x, y, peak_names, output_file_path,folds[fold]), callback = log_result)

#Dont allow to put any line in side the Pool
pool.close()
# Wait to the paralellization part to be finish for continouing with the rest of the code | without it, the parallelization never finish
pool.join()

end_time = time.time()
print("--- %s seconds ---" % ((end_time - start_time)))

#Create the outputs results

predictions_all = np.vstack(predictions_all)
correlations_all = np.hstack(correlations_all)
peak_order_all =  np.hstack(peak_order_all)

np.save(output_file_path + "predictions_all" + run_num + ".npy", predictions_all)
np.save(output_file_path + "correlations_all" + run_num + ".npy", correlations_all)
np.save(output_file_path + "peak_order" + run_num + ".npy", peak_order_all)

















