"""
RNNs are designed for modeling sequential data (sequences) and are capable of
remembering past and processing new events accordingly.

Manually compute the forward pass for an RNN. Create a recurrent layer from RNN
and perform a forward pass on an input sequence of length 3 to compute the output.
We all also manually compute the forward pass and compare the results with those of RNN.

input_size: number of expected features in the input x
hidden_size: number of features in the hidden state h
"""
import torch
import torch.nn as nn
torch.manual_seed(1)

rnn_layer = nn.RNN(input_size=5, hidden_size=2, num_layers=1, batch_first=True)

w_xh = rnn_layer.weight_ih_l0
w_hh = rnn_layer.weight_hh_l0
b_xh = rnn_layer.bias_ih_l0
b_hh = rnn_layer.bias_hh_l0

print(f'W_xh shape: {w_xh.shape}')
print(f'W_hh shape: {w_hh.shape}')
print(f'b_xh shape: {b_xh.shape}')
print(f'b_hh shape: {b_hh.shape}')

# now we will call the forward pass of the rnn_layer and manually compute
# the outputs at each time step and compare them

x_seq = torch.tensor([[1.0]*5, [2.0]*5, [3.0]*5]).float()
print(f'x_seq:\n {x_seq}')

print(x_seq.size())
# to conform with the rnn_layer's input dimensions, we need to add a new dimension:
# [3,5] -> [1,3,5] => [batch_size, seq_len, input_size]
x_seq = x_seq.reshape(1, 3, 5)
print(x_seq.size())

output, hidden = rnn_layer(x_seq)
# by default when print(output) we will get outputs associated with each time step (corresponding
# to number of input sequences
print(output)

# to get the output for the last time step, we use output[-1,:,:] or
print(output[-1,-1,:])

