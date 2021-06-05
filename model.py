import torch
import numpy as np
import json

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
        self.C = torch.nn.Linear(self.D, self.linear_width)

    def forward(self, x):
        out1 = self.W(x).clamp(min=0)
        out2 = self.V(out1)
        skip_out = self.C(x)
        return out2 + skip_out

    def weights(self):
        return [self.W.weight, self.V.weight]
    
    def state_dict_matlab(self, i=0):
        out = {
            f"W_{i}": self.W.weight.detach().numpy(),
            f"V_{i}": self.V.weight.detach().numpy(),
            f"C_{i}": self.C.weight.detach().numpy(),
        }
        return out
    
class DeepBlock(torch.nn.Module):

    def __init__(self, relu_in, relu_out):
        super().__init__()
        self.relu_in = int(relu_in)
        self.relu_out = int(relu_out)
        self.R = torch.nn.Linear(self.relu_in, self.relu_out)
        if self.relu_out == 1:
            self.C = None
        else:
            self.C = torch.nn.Linear(self.relu_in, self.relu_out)

    def forward(self, x):
        if self.relu_out == 1:
            out = self.R(x)
        else:
            out = self.R(x).clamp(min=0) + self.C(x)
        return out

    def weights(self):
        if self.relu_out == 1:
            return [self.R.weight]
        else:
            return [self.R.weight, self.C.weight]
        
    def state_dict_matlab(self, i=0):
        out = {f"R_{i}": self.R.weight.detach().numpy()}
        if self.relu_out != 1:
            out[f"C_{i}"] = self.C.weight.detach().numpy()
        return out

class Model(torch.nn.Module):
    ''' model is a sequence of ReLULinearSkipBlocks '''

    def __init__(self, D=2, relu_width=320, linear_width=160, layers=1, 
                        epochs=10, learning_rate=1e-3,
                        regularization_lambda=0.1, regularization_method=1,
                        block_architecture=False, modelid=0, weight_decay=0):
        super().__init__()
        self.D = int(D)
        self.relu_width = int(relu_width)
        self.linear_width = int(linear_width)
        self.layers = int(layers)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.regularization_lambda = float(regularization_lambda)
        self.regularization_method = regularization_method
        self.block_architecture = bool(block_architecture)
        if not self.block_architecture and self.regularization_method in ['term2']:
            self.weight_decay = 0
            self.regularization_lambda = float(regularization_lambda)
        elif self.regularization_method in ['standard_wd']:
            self.weight_decay = float(regularization_lambda)
            self.regularization_lambda = 0
        elif self.regularization_method in ['none']:
            self.weight_decay = 0
            self.regularization_lambda = 0
        self.modelid = int(modelid)
        self.blocks = torch.nn.Sequential(*self.build_blocks())

    def describe(self):
        out = {'relu_width': self.relu_width, 'linear_width': self.linear_width}
        out['layers'] = self.layers
        out['epochs'] = self.epochs
        out['learning_rate'] = self.learning_rate
        out['weight_decay'] = self.weight_decay
        out['regularization_lambda'] = self.regularization_lambda
        out['regularization_method'] = self.regularization_method
        out['block_architecture'] = self.block_architecture
        return out

    def build_blocks(self):
        blocks = []
        if self.block_architecture:
            blocks.append(DeepBlock(self.D, self.relu_width))
            for l in range(self.layers):
                blocks.append(DeepBlock(self.relu_width, self.relu_width))
            blocks.append(DeepBlock(self.relu_width, 1))
        else:            
            for bi in range(self.layers):
                D = self.D if bi == 0 else self.linear_width            
                out = 1 if bi == self.layers-1 else self.linear_width   
                block = ReLULinearSkipBlock(D, self.relu_width, out)
                blocks.append(block)
        return blocks

    def learn(self, x, y):
        x, y = torch.FloatTensor(x), torch.FloatTensor(y)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        sparsity_epochs = []
        loss_epochs = []
        for e in range(self.epochs):
            pred = self.forward(x)
            loss = criterion(pred, y)
            if not self.block_architecture and self.regularization_method == 'term2':
                loss += self.regularization_lambda * self.regularize_term2()
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

    def regularize_term2(self):
        r = 0
        for i in range(self.layers):
            r += self.blocks[i].W.weight.pow(2).sum()/2
            r += self.blocks[i].V.weight.abs().sum(0).pow(2).sum()/2
            r += self.blocks[i].C.weight.abs().sum()
            r += self.blocks[i].C.bias.abs().sum()
        return r
    
    def l2_norm(self):
        r = 0
        for param in self.parameters():
            if not param.requires_grad:
                continue
            r += param.data.pow(2).sum()
        return r
    
    def regularization_value_metric(self):
        if self.regularization_method == 'term2':
            return self.regularize_term2()
        elif self.regularization_method == 'standard_wd':
            return self.l2_norm()
        else:
            return torch.tensor(0)

    def sparsity(self):
        def sperc(x):
            x = x.abs()
            # threshold = 1e-3*x.max()
            threshold = 10e-3
            perc = (x > threshold).sum() / x.numel()
            return perc.numpy().item() 
        sparsity = []
        for block in self.blocks:
            for weight in block.weights():
                sparsity.append(sperc(weight))
        return np.array(sparsity)

    def norms(self):
        ns = []
        for block in self.blocks:
            ns.append([
                block.W.weight.detach().pow(2).sum().numpy().item(),
                block.V.weight.detach().pow(2).sum().numpy().item()
            ])
        return ns

    def show(self, p=None):
        print(f"=========={len(self.blocks)} layers==========")
        print(json.dumps(self.describe(), indent=4))
        if p is not None:
            print(f"forward pass test: {self.forward(p)}")
        for b in self.blocks:
            print(b)
        print("============================")
        
    def state_dict_matlab(self):
        out = {}
        for i in range(len(self.blocks)):
            mat = self.blocks[i].state_dict_matlab(i)
            out.update(mat)
        return out

    
