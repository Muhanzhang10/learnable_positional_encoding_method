import torch
import torch.nn as nn 
import torch.optim as optim

from transformer import TransformerSentenceEncoder



def train():
    epochs = 5
    batch_size = 256
    
    model = TransformerSentenceEncoder(3000)
    criterion = nn.CrossEntropyLoss() # loss function
    parameters = model.parameters()

    # Set hyperparameters for Adam optimizer
    learning_rate = 0.001
    betas = (0.9, 0.999)  
    eps = 1e-8 
    optimizer = optim.Adam(parameters, lr=learning_rate, betas=betas, eps=eps)

    
    for i in range(epochs):
        # Initiate dummy data
        inputs = torch.rand(500, batch_size).to(torch.long)
        targets = torch.rand(batch_size).to(torch.long)
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()  # Clear the gradients
        loss.backward()  # Compute gradients
        optimizer.step()  # Update weights
        print(loss)
train()
