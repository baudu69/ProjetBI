import argparse
import pickle
from datetime import datetime

from matplotlib import pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict the consumption of a date")
    parser.add_argument('-d', '--date', type=str, help='Date voulu yyy-mm-dd', required=True)
    parser.add_argument('-t', '--temperature', type=list[float], help='Température', required=True)
    parser.add_argument('-hr', '--heure', type=list[int], help='Heure', required=True)
    parser.add_argument('-di', '--display', type=bool, help='Display the model', required=False, default=False)
    args = parser.parse_args()
    ss = pickle.load(open('standard_scaler.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
    date = args.date
    temperature = args.temperature
    data = []
    jour_semaine = datetime.strptime(date, '%d/%m/%Y').weekday()
    jour_mois = datetime.strptime(date, '%d/%m/%Y').day
    mois = datetime.strptime(date, '%d/%m/%Y').month
    heures = args.heure
    print(heures)
    for heure in args.heure:
        data.append([heure, jour_semaine, jour_mois, mois, temperature[heure]])
    data_norm = ss.transform(data)
    pred = model.predict(data_norm)
    if args.display:
        plt.plot(pred)
        plt.legend(['Consommation estimée'])
        plt.xlabel('Heure')
        plt.ylabel('Consommation (mWh)')
        plt.show()
        print(pred)
