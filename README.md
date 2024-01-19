# Learnable positional encoding method
This repo implements a learnable positional encoding method by incorporating both absolute positional encoding and relative positional encoding. 

Typical attention layers are implemented as the following
![alt text](https://github.com/Muhanzhang10/learnable_positional_encoding_method/blob/master/images/attention.png)

In this design, both absolute and relative positions are considered, learnable, and incorporated in the attention layers. We have $p_i and$p_j to be absolute position embedding, $U_Q and $U_K to be projection matrices for position embedding, and $b_j-i to be relative positional embedding.
[design]

## Train
To train the model (with dummy data), run 
`python3 train.py`

To check the model with single outputs (with dummy data), run
`python3 test.py`
