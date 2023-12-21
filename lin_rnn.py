import torch
from torch import nn
from typing import Tuple, List, Dict

class LinRNN(nn.Module):
    """
    Basic linear RNN block. This represents a single layer of linear RNN
    """
    def __init__(self, input_size: int, hidden_size: int) -> None:
        """
        input_size: Number of features of your input vector
        hidden_size: Number of hidden neurons
        output_size: Number of features of your output vector
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, hidden_state) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns computed output and lin(i2h + h2h).
        Returns follows pytorch RNN standard: output contains the hidden_state matrix, hn the last hidden_state.
        Inputs
        ------
        x: Input vector
        hidden_state: Previous hidden state
        Outputs
        -------
        out:  output (without activation because of how pytorch works)
        hidden_state: New hidden state matrix
        """
        x = self.i2h(x)
        hidden_state = self.h2h(hidden_state)
        hidden_state = x + hidden_state
        return hidden_state, hidden_state[-1]