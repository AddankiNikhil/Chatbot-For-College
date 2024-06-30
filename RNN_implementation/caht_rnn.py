import json
import torch
import random

from torch import nn
from torch.utils.data import Dataset, DataLoader
from nltk_utils import tokenize, bag_of_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('your_file_modified.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_size).to(x.device)
        out, _ = self.rnn(x.unsqueeze(0), h0)
        out = self.fc(out[-1, :, :])
        return out


model = RNN(input_size=data["input_size"], hidden_size=data["hidden_size"], output_size=data["output_size"]).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"


def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(tag)
                if "links" in intent.keys():
                    return random.choice(intent['responses']), intent["links"][0]
                else:
                    return random.choice(intent['responses'])
    return "I do not understand..."


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)
