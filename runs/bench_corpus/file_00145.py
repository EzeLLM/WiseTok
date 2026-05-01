import torch
from Perceptron_2T import Perceptron
    
#########################TRAIN PERCEPTRONS##########################################    
#this will not be changed in this program
#store the inputs and outputs
all_inputs = torch.tensor([
    [0.0,0.0],
    [0.0,1.0],
    [1.0,0.0],
    [1.0,1.0]
    ])


input_dimension = 2          #number of inputs at a time
num_hidden = 2


#code for AND
                
and_perceptron = Perceptron()

desired_output = torch.tensor([ [0.0],
                                [0.0],
                                [0.0],
                                [1.0]
                                     ])

and_perceptron.train(all_inputs, desired_output)



#code for OR

or_perceptron = Perceptron()

desired_output = torch.tensor([ [0.0],
                                [1.0],
                                [1.0],
                                [1.0]
                                     ])

or_perceptron.train(all_inputs, desired_output)
              
              
              
#code for XOR

xor_perceptron = Perceptron()

desired_output = torch.tensor([ [0.0],
                                [1.0],
                                [1.0],
                                [0.0]
                                     ])

xor_perceptron.train(all_inputs, desired_output)


    
###############################TEST PERCEPTRONS#########################################

#Same order of data used to train the perceptrons

results = and_perceptron.test(all_inputs)
print("End results for AND")
print(results)


results = or_perceptron.test(all_inputs)
print("End results for OR")
print(results)


results = xor_perceptron.test(all_inputs)
print("End results for XOR")
print(results)


"""

for result in range(4):
    if results[result] >= 0.5:
        result_b[result] = 1
    else:
        result_b[result] = 0
print(result_b)
print()
print()
print()

"""