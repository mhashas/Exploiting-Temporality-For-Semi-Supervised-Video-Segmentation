import torch
from torch import nn
from torch.autograd import Variable

class ConvLSTMCell(nn.Module):
    """
    Generate a convolutional LSTM cell
    """

    def __init__(self, input_size, hidden_size, kernel_size, padding, bias=True):
        super(ConvLSTMCell,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gates = nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size, padding=padding, bias=bias)
        self._init_weights()

    def forward(self, x, prev_state):
        prev_hidden, prev_cell = prev_state

        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat((x, prev_hidden), 1)
        gates = self.gates(stacked_inputs)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)
        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)

        return (hidden, cell)

    def _init_weights(self):
        torch.nn.init.xavier_normal_(self.gates.weight.data, 0.02)


class ConvLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, kernel_size, padding=1, num_layers=1, batch_first=True, bias=True):
        super(ConvLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = padding
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.layers = self.build_layers()

    def build_layers(self):
        layers = []
        for i in range(self.num_layers):
            curr_input_size = self.input_size if i == 0 else self.hidden_size
            layers.append(ConvLSTMCell(curr_input_size, self.hidden_size, self.kernel_size, self.padding, self.bias))

        return nn.Sequential(*layers)


    def forward(self, x, hidden_state=(None, None)):
        seq_len = x.size(1)
        current_input = x
        output = []

        for i, layer in enumerate(self.layers):
            if i == 0:
                hidden_state = self.init_hidden(x)

            for t in range(seq_len):
                hidden_state = layer(current_input[:, t, :, :, :], hidden_state)
                output.append(hidden_state[0])

            output = torch.stack(output, dim=1)
            current_input = output

        return output

    def init_hidden(self, x):
        batch_size = x.size()[0]
        spatial_size = x.size()[3:]
        state_size = [batch_size, self.hidden_size] + list(spatial_size)

        if torch.cuda.is_available():
            return (Variable(torch.zeros(state_size)).cuda(), Variable(torch.zeros(state_size)).cuda())
        else:
            return (Variable(torch.zeros(state_size)), Variable(torch.zeros(state_size)))


