import torch
from torch import nn

# Create a model instance
model = torch.nn.Sequential(
    nn.Linear(in_features=512, out_features=1024),
    nn.ReLU(),
    nn.Linear(in_features=1024, out_features=1024),
    nn.ReLU(),
    nn.Linear(in_features=1024, out_features=1)
)

# Load the state dictionary into the model
model.load_state_dict(torch.load('C:\\Users\\Suryansh\\VS_Codes\\drone_practicum\\model_weights.pth'))

# Now you can use the model for predictions
def final_model(x):
    test_logits = model(x).squeeze()
    test_preds = torch.round(torch.sigmoid(test_logits))
    test_preds = int(test_preds)
    return test_preds
