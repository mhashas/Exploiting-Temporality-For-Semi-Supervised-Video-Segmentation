import torch.nn as nn
import torch

INITIAL_STATE_0 = '0'
INITIAL_STATE_LEARNED = '0-learned'
INITIAL_STATE_CNN_LEARNED = 'cnn-learned'

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, batch_first, bidirectional=False, lstm_initial_state=INITIAL_STATE_0):
        super(LSTM, self).__init__()
        self.num_directions = 2 if bidirectional else 1
        self.hidden_size = int(hidden_size / self.num_directions)
        self.lstm_initial_state = lstm_initial_state
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size, batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, x, hidden_init=None):
        if hidden_init is None:
            hidden_init = self.get_hidden_state(x)

        out, (hn, cn) = self.lstm(x, hidden_init)

        return out, (hn, cn)


    def get_hidden_state(self, x):
        batch_size = x.size(0)

        if self.lstm_initial_state == INITIAL_STATE_0:
            h0 = torch.zeros(self.num_directions, batch_size, self.hidden_size)
            c0 = torch.zeros(self.num_directions, batch_size, self.hidden_size)

        elif self.lstm_initial_state == INITIAL_STATE_LEARNED:
            h0 = torch.zeros(self.num_directions, 1, self.hidden_size)
            c0 = torch.zeros(self.num_directions, 1, self.hidden_size)
            nn.init.xavier_normal_(h0, gain=nn.init.calculate_gain('relu'))
            nn.init.xavier_normal_(c0, gain=nn.init.calculate_gain('relu'))

            h0 = nn.Parameter(h0, requires_grad=True)  # Parameter() to update weights
            c0 = nn.Parameter(c0, requires_grad=True)
            h0 = h0.repeat(1, batch_size, 1)
            c0 = c0.repeat(1, batch_size, 1)

        if torch.cuda.is_available():
            h0 = h0.cuda()
            c0 = c0.cuda()

        return (h0, c0)