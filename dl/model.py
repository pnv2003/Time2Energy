from torch import nn

class CNN_LSTM(nn.Module):
    def __init__(self, 
        input_size=1, 
        cnn_num_filters=64,
        cnn_kernel_size=3,
        cnn_stride=1,
        cnn_padding=1,
        pool_kernel_size=2,
        pool_stride=2,
        lstm_hidden_units=50,
        lstm_num_layers=1,
        dropout=0.2,
        fc_hidden_units=50,
        output_size=1
    ):
        super(CNN_LSTM, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(input_size, cnn_num_filters, kernel_size=cnn_kernel_size, stride=cnn_stride, padding=cnn_padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_stride)
        )
        
        self.lstm = nn.LSTM(
            cnn_num_filters, lstm_hidden_units, num_layers=lstm_num_layers, batch_first=True, dropout=dropout
        )
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_units, fc_hidden_units),
            nn.ReLU(),
            nn.Linear(fc_hidden_units, output_size)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x
