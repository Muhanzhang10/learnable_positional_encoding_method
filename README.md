## Learnable positional encoding method
This repo implements a learnable positional encoding method by incorporating both absolute positional encoding and relative positional encoding. 

Typical attention layers are implemented as the following
![alt text](https://github.com/Muhanzhang10/learnable_positional_encoding_method/blob/master/images/attention.png)

In this design, both absolute and relative positions are considered, learnable, and incorporated in the attention layers. We have p<sub>i</sub> and p<sub>j</sub> to be absolute position embedding, U<sup>Q</sup> and U<sup>K</sup> to be projection matrices for position embedding, and b<sub>j-i</sub> to be relative positional embedding.
![alt text](https://github.com/Muhanzhang10/learnable_positional_encoding_method/blob/master/images/design.png)

## Deliverables
Paragraph response to question 1 is in *deliverables/Question1.pdf*

5 min video to question 2 is in *deliverables/learnable_position_encoding_method.mp4*

## Run
To train the model (with dummy data), run 
`python3 train.py`

To check the model with single outputs (with dummy data), run
`python3 test.py`

## Citations
[Guolin Ke, Di He, Tieyan Liu. "RETHINKING POSITIONAL ENCODING IN LANGUAGE PRE-TRAINING" ICLR 2021, 2021.](https://arxiv.org/pdf/2006.15595.pdf)

[Peter Shaw, Jakob Uszkoreit, Ashish Vaswani. "Self-Attention with Relative Position Representations", 20218.](https://arxiv.org/pdf/1803.02155.pdf)
