# importowanie niezbędnych bibliotek
import numpy as np
import pandas_datareader.data as pdr
from pandas_datareader._utils import RemoteDataError
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import json

plt.style.use('fivethirtyeight')

# utworzenie zmiennych typu datetime do przechowania zakresu czasu
date = [2021, 1, 1]  # podanie przez użytkownika daty od której mają być pobrane dane (rok, miesiąc, dzień)
end_time = dt.datetime.now()  # aktualny czas
start_time = dt.datetime(date[0], date[1], date[2])  # czas początkowy
time_data = (start_time, end_time,
             start_time - end_time)  # krotka przechowująca informacje o czasie (czas początkowy, aktualny czas, roznica)
json_file_path = "config.json"

with open(json_file_path) as json_file:
    config = json.load(json_file)


# utworzenie zmiennych przechowujących listę akcji do analizy
# stock_list = ['TSLA', 'AMZN', 'AAPL', 'META', 'GOOG']
# możliwe jest też odczytanie z pliku z wykorzystaniem przygotowanej funkcji get_stock_data_from_file(file_path)

def predict_stock_moves_ML(stock_input_array)
    #możliwe jest przekazanie tylko jednej akcji na raz!!!
    data = stock_input_array.Close
    dataset = data.values
    # wykorzystanie 80% danych do uczenia maszynowego
    training_data_len = math.ceil(len(dataset) * .8)
    # normalizacja danych z wykorzystaniem MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    # przygotowanie danych do trenowania modelu
    train_data = scaled_data[0:training_data_len, :]
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])

def get_data_from_yahoo(stock_input_array):
    try:
        stock_data = pdr.get_data_yahoo(stock_input_array, start_time, end_time)
        return stock_data
    except RemoteDataError:
        file_path_temp = config['stockFilePath']
        print(f'No data found for {file_path_temp}')


def get_stock_data_from_file(file_path):
    try:
        lines_object = open(file_path, "r")
        lines = lines_object.read().splitlines()
        lines_object.close()
        return lines
    except:
        print(f'An error occured while trying to read {file_path}')


def show_graphs(input_data_frame, save_to_files):
    count = 0
    input_data_frame = input_data_frame.Close
    input_data_frame.plot()
    if save_to_files == True:
        file_name = "graph" + str(count) + ".png"
        plt.savefig(file_name)
        count += 1
    input_data_frame.plot.box()
    if save_to_files == True:
        file_name = "graph" + str(count) + ".png"
        plt.savefig(file_name)
        count += 1
    input_data_frame.plot.area(stacked=False)
    if save_to_files == True:
        file_name = "graph" + str(count) + ".png"
        plt.savefig(file_name)
        count += 1
    plt.show()


def main():
    data_frame = get_data_from_yahoo(get_stock_data_from_file(config['stockFilePath']))
    # uzyskiwanie dostępu do danych ze statusem Close
    show_graphs(data_frame, True)
    predict_stock_moves_ML(data_frame)


if __name__ == "__main__":
    main()
