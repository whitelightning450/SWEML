# Model Training Sensitivity Tests

import pandas as pd
import os
import DataProcess
import MLP_Model
#Set working directories
cwd = os.getcwd()
os.chdir("..")
os.chdir("..")
datapath = os.getcwd()  

#Define hold out year
HOY = 2019
#Run data processing script to partition key regional dataframes
#note, need to load RegionTrain_SCA.h5,
RegionTrain, RegionTest, RegionObs_Train, RegionObs_Test, RegionTest_notScaled = DataProcess.DataProcess(HOY, datapath, cwd)

# list of epochs to test
epochs_test = [30] #[1, 5, 10, 30, 50, 100]
# list of batch_sizes to test
batch_size_test = [100] #[10, 50, 100, 150, 200, 300, 500]
# list of node_lists to test (determines number of layers and number of nodes per layer)
node_list_test =  [[128, 128, 64, 64, 32, 16]]#,
                 # [128, 128, 64, 32, 16],
                  #[128, 64, 32, 16],
                  #[128, 128],
                  #[256]]  # originally we used [128, 128, 64, 64, 32, 16] e.g. [2**7, 2**7, 2**6, 2**6, 2**5, 2**4]

# shuffle True or False
shuffle = True

# activation method for layers
activation_test = ["relu", "sigmoid", "softmax", "softplus", "softsign", "tanh", "selu", "elu", "exponential"]

# create new file on first iteration
make_file = True

for activation in activation_test:
    print(activation)
    for epochs in epochs_test:
        for batch_size in batch_size_test:
            for node_list in node_list_test:
                #model training with current set of parameters
                MLP_Model.Model_train(cwd, epochs, batch_size, RegionTrain, RegionTest, RegionObs_Train, RegionObs_Test, node_list, shuffle, activation)

                #Need to create Predictions folder if running for the first time
                Predictions = MLP_Model.Model_predict(cwd,  RegionTest, RegionObs_Test, RegionTest_notScaled)

                Performance = MLP_Model.Prelim_Eval(cwd, Predictions, True)

                # add test parameters to dataframe
                Performance['nLayers'] = [len(node_list)] * len(Performance)
                Performance['node_list'] = [node_list] * len(Performance)
                Performance['epochs'] = [epochs] * len(Performance)
                Performance['batch_size'] = [batch_size] * len(Performance)
                Performance['shuffle'] = [shuffle] * len(Performance)
                Performance['activation'] = [activation] * len(Performance)


                if make_file == True:
                    print("making output file")
                    print("write to output file")
                    Performance.to_csv('~/all_tests_performance_activation.csv')
                    make_file = False # set to false so that we append to our existing file on next iterations
                else:
                    print("write to output file")
                    Performance.to_csv('~/all_tests_performance_activation.csv', mode='a', header=False)
            
            