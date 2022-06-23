# aby dane były jakkolwiek poprawnie przewidywane, niezbędne będzie pobranie wartości akcji sprzed minimum dwóch lat!
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

def predict_stock_moves_ML(stock_input_array):
    # możliwe jest przekazanie tylko jednej akcji na raz!!!
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
        x_train.append(train_data[i - 60:i, 0])
        y_train.append(train_data[i, 0])
    # konwersja x_train oraz y_train na np
    x_train, y_train = np.array(x_train), np.array(y_train)
    # przygotowanie macierzy trójwymiarowej (wymagane przez silnik neuronowy LSTM)
    print(f"Aktualna wielkość danych treningowych wynosi {x_train.shape[0]}x{x_train.shape[1]}")
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    # budowanie modelu
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    # kompilowanie modelu
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    # trenowanie modelu
    model.fit(x_train, y_train, batch_size=1, epochs=15)
    model_output_path = config['model_path']
    model.save(model_output_path, save_format='tf')
    # utworzenie danych do testowania
    test_data = scaled_data[training_data_len - 60:, :]
    # utworzenie x_test oraz y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])
    # konwersja x_test na np
    x_test = np.array(x_test)
    # przygotowanie macierzy trójwymiarowej (wymagane przez silnik neuronowy LSTM)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    # przewidywanie
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    # uzyskiwanie RMSE
    RMSE = np.sqrt(np.mean(predictions - y_test) ** 2)
    print(f"Współczynnik RMSE wynosi: {RMSE}")
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    output = (train, valid)
    return output


def visualize_predicted_data(train, valid):
    plt.figure()
    plt.title('Model')
    plt.xlabel('Date')
    plt.ylabel('Close price')
    # TODO: do ogarnięcia wyświetlanie plotów z valid i train, coś nie działa
    plt.plot(train)
    plt.plot(valid)
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')


def get_data_from_yahoo(stock_input_array):
    try:
        stock_data = pdr.get_data_yahoo(stock_input_array, start_time, end_time)
        return stock_data
    except RemoteDataError as e:
        file_path_temp = config['stockFilePath']
        print(f'No data found for {file_path_temp}. Exception message: {e}')
        print(f"Check your internet connection or firewall settings!")


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
    plt.legend(loc='lower right')


def main():
    # pobieranie danych z Yahoo Finances
    data_frame = get_data_from_yahoo(get_stock_data_from_file(config['stockFilePath']))
    # Przygotowywanie wizualizacji pobranych danych
    show_graphs(data_frame, True)
    # Wizualizacja przewidywanego zachowania giełdy na 60 dni do przodu
    train, valid = predict_stock_moves_ML(data_frame)
    visualize_predicted_data(train, valid)
    # Wyświetlenie wykresów
    plt.show()


if __name__ == "__main__":
    main()
