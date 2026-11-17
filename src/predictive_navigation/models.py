
import torch
from torch import nn
from torch.nn import functional as F


class NavigationRNN(nn.Module):
    def __init__(self, hidden_size=16, head_type="linear"):
        super().__init__()

        # 3 observation categories + 4 previous-action categories.
        # The fourth action category means "no previous action".
        self.gru = nn.GRU(
            input_size=7,
            hidden_size=hidden_size,
            batch_first=True,
        )

        # Predict A, B, or C from memory and the next action.

        if head_type == "linear":
            self.head_features = nn.Identity()
            head_input_size = hidden_size + 3

        elif head_type == "mlp":
            self.head_features = nn.Sequential(
                nn.Linear(hidden_size + 3, 32),
                nn.ReLU(),
            )
            head_input_size = 32

        else:
            raise ValueError(f"Unknown head type: {head_type}")

        self.head = nn.Linear(head_input_size, 3)



    def forward(self, observations, actions):
        """
        observations: (batch, steps), current observations
        actions:      (batch, steps), upcoming movement IDs

        Returns:
            logits:        (batch, steps, 3)
            hidden_states: (batch, steps, hidden_size)
        """
        if observations.ndim != 2 or actions.shape != observations.shape:
            raise ValueError(
                "Observations and actions must have shape (batch, steps)"
            )

        # Previous action for each time step.
        # ID 3 represents the initial "no previous action".
        previous_actions = torch.full_like(actions, 3)
        previous_actions[:, 1:] = actions[:, :-1]

        observation_input = F.one_hot(
            observations, num_classes=3
        )
        previous_action_input = F.one_hot(
            previous_actions, num_classes=4
        )

        rnn_input = torch.cat(
            [observation_input, previous_action_input],
            dim=-1,
        ).float()

        hidden_states, _ = self.gru(rnn_input)

        # The movement we are about to make.
        current_action_input = F.one_hot(
            actions, num_classes=3
        ).float()

        prediction_input = torch.cat(
            [hidden_states, current_action_input],
            dim=-1,
        )

        # logits = self.head(prediction_input)
        features = self.head_features(prediction_input)
        logits = self.head(features)

        return logits, hidden_states



class ObservationActionMLP(nn.Module):
    #Predict the next observation without recurrent memory

    def __init__(self, hidden_size=40):
        super().__init__()

        # 3 observation categories + 3 action categories
        self.network = nn.Sequential(
            nn.Linear(6, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, observations, actions):
        if observations.ndim != 2 or actions.shape != observations.shape:
            raise ValueError(
                "Observations and actions must have shape (batch, steps)"
            )

        obs_input = F.one_hot(observations, num_classes=3)
        action_input = F.one_hot(actions, num_classes=3)

        x = torch.cat([obs_input, action_input], dim=-1).float()

        logits = self.network(x)

        # No recurrent hidden state
        return logits, None

