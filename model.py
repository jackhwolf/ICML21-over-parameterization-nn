import torch
import numpy as np
import json

class ReLUSkipBlock(torch.nn.Module):
    ''' custom module for ReLU block with skip connection '''

    def __init__(self, r_width_in, r_width_out):
        '''
        r_width_in:  int, dimensions of input
        r_width_out: int, dimensions of input
        '''
        super().__init__()
        self.relu_width_in = int(r_width_in)
        self.relu_width_out = int(r_width_out)            
        self.l1 = torch.nn.Linear(self.relu_width_in, self.relu_width_out)
        self.skip_l = torch.nn.Linear(self.relu_width_in, self.relu_width_out)
        if self.relu_width_out == 1:
            self.l2 = torch.nn.Linear(1, 1)

    def forward(self, x):
        out1 = self.l1(x).clamp(min=0)
        skip_out = self.skip_l(x)
        out = out1 + skip_out
        if self.relu_width_out == 1:
            out = self.l2(out)
        return out

    def weights(self):
        ws = [self.l1.weight]
        if self.relu_width_out == 1:
            ws.append(self.l2.weight)
        return ws

class ReLULinearSkipBlock(torch.nn.Module):
    ''' custom module for ReLU-->Linear block/layer with skip '''

    def __init__(self, D, r_width, l_width):
        '''
        D:          int, dimensions of input
        r_width:    int, width of ReLU layer
        l_width:    int, width of linear layer
        '''
        super().__init__()
        self.D = int(D)
        self.relu_width = int(r_width)
        self.linear_width = int(l_width)
        self.W = torch.nn.Linear(self.D, self.relu_width)
        self.V = torch.nn.Linear(self.relu_width, self.linear_width)
        self.skip_l = torch.nn.Linear(self.D, self.linear_width)

    def forward(self, x):
        out1 = self.W(x).clamp(min=0)
        out2 = self.V(out1)
        skip_out = self.skip_l(x)
        return out2 + skip_out

    def weights(self):
        return [self.W.weight, self.V.weight]

class Model(torch.nn.Module):
    ''' model is a sequence of ReLULinearSkipBlocks '''

    def __init__(self, D=2, relu_width=320, linear_width=160, layers=1, 
                        epochs=10, learning_rate=1e-3, 
                        regularization_lambda=0.1, regularization_method=1,
                        modelid=0):
        super().__init__()
        self.D = int(D)
        self.relu_width = int(relu_width)
        self.linear_width = int(linear_width)
        self.layers = int(layers)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.regularization_lambda = float(regularization_lambda)
        self.regularization_method = int(regularization_method)
        if self.regularization_method in [0, 1, 2]:
            self.wd = 0
        elif self.regularization_method == 3:
            self.wd = 1e-3
        self.blocks = torch.nn.Sequential(*self.build_blocks())
        self.modelid = int(modelid)
        self.should_regularize = self.regularization_method in [1, 2]

    def describe(self):
        out = {'relu_width': self.relu_width, 'linear_width': self.linear_width}
        out['layers'] = self.layers
        out['epochs'] = self.epochs
        out['learning_rate'] = self.learning_rate
        out['weight_decay'] = self.wd
        out['regularization_lambda'] = self.regularization_lambda
        out['regularization_method'] = self.regularization_method
        return out

    def build_blocks(self):
        blocks = []
        for bi in range(self.layers):
            if self.linear_width != 0:
                D = self.D if bi == 0 else self.linear_width            
                out = 1 if bi == self.layers-1 else self.linear_width   
                block = ReLULinearSkipBlock(D, self.relu_width, out)
            else:
                D = self.D if bi == 0 else self.relu_width            
                out = 1 if bi == self.layers-1 else self.relu_width   
                block = ReLUSkipBlock(D, out)
            blocks.append(block)
        return blocks

    def learn(self, x, y):
        x, y = torch.FloatTensor(x), torch.FloatTensor(y)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.wd)
        sparsity_epochs = []
        loss_epochs = []
        for e in range(self.epochs):
            pred = self.forward(x)
            loss = criterion(pred, y)
            if self.should_regularize:
                loss += (self.regularization_lambda/2) * self.regularize()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_sparsity = self.sparsity().tolist()
            if e % 200 == 0:
                print(f"ID: {self.modelid}, E: {e}, LOSS: {loss}")
                print(current_sparsity)
            if e % 5000 == 0:
                sparsity_epochs.append([e, current_sparsity])
                loss_epochs.append([e, loss.detach().numpy().item()])
        return loss.detach().numpy().item(), sparsity_epochs, loss_epochs
    
    def forward(self, x):
        x = torch.FloatTensor(x)
        return self.blocks(x)

    def predict(self, x):
        x = torch.FloatTensor(x)
        with torch.no_grad():
            return self.forward(x).detach().numpy()

    def regularize(self):
        r = 0
        for i in range(self.layers):
            r += self.blocks[i].W.weight.pow(2).sum()
            if self.regularization_method == 1:
                r += self.blocks[i].V.weight.pow(2).sum()
            elif self.regularization_method == 2:
                terms = self.blocks[i].V.weight.abs().sum(0).pow(2)
                for c in terms:
                    r += c
        return r

    def sparsity(self):
        def sperc(x):
            x = x.abs()
            threshold = 1e-3*x.max()
            perc = (x > threshold).sum() / x.numel()
            return perc.numpy().item() 
        sparsity = []
        for block in self.blocks:
            for weight in block.weights():
                sparsity.append(sperc(weight))
        return np.array(sparsity)

    def show(self, p=None):
        print(f"=========={len(self.blocks)} layers==========")
        print(json.dumps(self.describe(), indent=4))
        if p is not None:
            print(f"forward pass test: {self.forward(p)}")
        for b in self.blocks:
            print(b)
        print("============================")

    