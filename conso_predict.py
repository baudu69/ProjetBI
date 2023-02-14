import argparse
from datetime import datetime
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(torch.nn.Module):
    def __init__(self, in_dim, caches, out_dim=1):
        super(MLP, self).__init__()

        assert out_dim==1, 'out_dim must be 1'
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.couches = torch.nn.ModuleList()
        for i in range(len(caches)):
            self.couches.append(torch.nn.Linear(self.in_dim if i == 0 else caches[i-1], caches[i]))
        self.couches.append(torch.nn.Linear(caches[-1], self.out_dim))

    def forward(self, x):
        for i in range(len(self.couches)):
            x = torch.relu(self.couches[i](x))
        x=torch.squeeze(x)
        return x



def predict(model, heure, temperature, jour_semaine, jour_mois, mois):
    toPredict = np.array([[heure, mois, temperature, jour_semaine, jour_mois]])
    toPredict = torch.tensor(toPredict, dtype=torch.float, device=device)
    y_pred = model(toPredict)
    y_pred = y_pred.data.cpu().numpy()
    return y_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict the consumption of a date")
    parser.add_argument('-d', '--date', type=str, help='Date voulu yyy-mm-dd', required=True)
    parser.add_argument('-t', '--temperature', type=float, help='Température', required=True)
    parser.add_argument('-hr', '--heure', type=int, help='Heure', required=True)
    args = parser.parse_args()
    model = torch.load('model_non_normalise.model')
    heure = args.heure
    temperature = args.temperature
    date = args.date
    date_time = datetime.strptime(date, '%d/%m/%y')
    jour_semaine = date_time
    jour_mois = date_time.day
    mois = date_time.month
    print(predict(model, heure, temperature, jour_semaine, jour_mois, mois))
