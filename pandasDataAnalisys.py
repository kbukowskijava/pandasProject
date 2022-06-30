# aby dane były jakkolwiek poprawnie przewidywane, niezbędne będzie pobranie wartości akcji sprzed minimum dwóch lat!
# importowanie niezbędnych bibliotek
import datetime as dt
import json
import math
import os

import absl.logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as pdr
import tensorflow as tf
from keras.layers import Dense, LSTM
from keras.models import Sequential
from pandas_datareader._utils import RemoteDataError
from sklearn.preprocessing import MinMaxScaler

CURRENT_STOCK = ""
RMSE_VALUES = []

# wyłączenie errorów tensorflow
absl.logging.set_verbosity(absl.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 8

# utworzenie zmiennych typu datetime do przechowania zakresu czasu
date = [2021, 1, 1]  # podanie przez użytkownika daty od której mają być pobrane dane (rok, miesiąc, dzień)
end_time = dt.datetime.now()  # aktualny czas
start_time = dt.datetime(date[0], date[1], date[2])  # czas początkowy
time_data = (start_time, end_time,
             start_time - end_time)  # krotka przechowująca informacje o czasie (czas początkowy, aktualny czas, roznica)
json_file_path = "config.json"  # ścieżka do pliku konfiguracyjnego

# załadowanie pliku JSON do dict'a config
with open(json_file_path) as json_file:
    config = json.load(json_file)


# możliwe jest też odczytanie z pliku z wykorzystaniem przygotowanej funkcji get_stock_data_from_file(file_path)

# funkcja wykorzystująca biblioteki tensorflow do utworzenia modelu ML oraz przewidywanie
def predict_stock_moves_ML(stock_input_array):
    try:
        # możliwe jest przekazanie tylko jednej akcji na raz!!!
        data = stock_input_array.filter(['Close'])  # filtrowanie data frame w celu uzyskania tylko kolumny Close
        dataset = data.values  # konwersja na np.array
        # wykorzystanie 80% danych do uczenia maszynowego (nie ma sensu więcej)
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
        # uzyskiwanie RMSE - Root Mean Square Error (jest standardowym sposobem pomiaru błędu modelu w przewidywaniu danych ilościowych)
        RMSE = np.sqrt(np.mean(predictions - y_test) ** 2)
        RMSE_VALUES.append(str("RMSE dla " + CURRENT_STOCK + " = " + str(RMSE)))
        print(f"Współczynnik RMSE wynosi: {RMSE}")
        list_of_y_test = y_test.tolist()
        list_of_predictions = predictions.tolist()
        print_confusion_matrix(flatten_list(list_of_y_test), flatten_list(list_of_predictions))
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
        output = (train, valid)
        return output
    except Exception as e:
        print(f'Exception message: {e}')


def flatten_list(list_to_flatten):
    flat_list = []
    # Iteracja po każdym elemencie głównej listy
    for element in list_to_flatten:
        if type(element) is list:
            # Jeśli element jest listą iteruj po elementach tego elementu
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


def print_confusion_matrix(y_true, y_pred):
    cm = tf.math.confusion_matrix(y_true, y_pred)
    cm_temp = [cm[0][0], cm[0][1], cm[1][0], cm[1][1]]
    save_matrix_to_xlsx(cm_temp, config['confusionMatrixSavePath'])
    print('True positive = ', cm[0][0])
    print('False positive = ', cm[0][1])
    print('False negative = ', cm[1][0])
    print('True negative = ', cm[1][1])


def save_matrix_to_xlsx(matrix, savePath):
    df = pd.DataFrame(matrix).T
    df.to_excel(excel_writer=str(savePath + CURRENT_STOCK + "_CM.xlsx"))


# Zapisanie wszystkich RMSE do pliku
def save_rmse_to_txt(savePath):
    with open(savePath, 'w') as fp:
        for item in RMSE_VALUES:
            # write each item on a new line
            fp.write("%s\n" % item)


# funkcja wizualizująca przewidywane dane
def visualize_predicted_data(train, valid, save_to_files, count, current_stock_name):
    try:
        plt.figure()
        plt.title(current_stock_name + " ML Predicitions")
        plt.xlabel('Date')
        plt.ylabel('Close price')
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
        if save_to_files:
            file_name = "graphs/graph" + str(count) + ".png"
            plt.savefig(file_name)
            count += 1
        return count
    except Exception as e:
        print(f'Exception message: {e}')
        return count


# pobieranie danych z wykorzystaniem API z Yahoo Finances
def get_data_from_yahoo(stock_input_array):
    try:
        stock_data = pdr.get_data_yahoo(stock_input_array, start_time, end_time)
        return stock_data
    except RemoteDataError as e:
        file_path_temp = config['stockFilePath']
        print(f'No data found for {file_path_temp}. Exception message: {e}')
        print(f"Check your internet connection or firewall settings!")


# pobieranie danych z pliku txt
def get_stock_data_from_file(file_path):
    try:
        lines_object = open(file_path, "r")
        lines = lines_object.read().splitlines()
        lines_object.close()
        return lines
    except RemoteDataError as e:
        print(f'An error occured while trying to read {file_path}. Error message: {e}')


# wizualizacja wszystkich akcji na jednym wykresie
def show_graphs(input_data_frame, save_to_files, count):
    try:
        input_data_frame = input_data_frame.Close
        input_data_frame.plot()
        if save_to_files:
            file_name = "graphs/graph" + str(count) + ".png"
            plt.savefig(file_name)
            count += 1
        input_data_frame.plot.box()
        if save_to_files:
            file_name = "graphs/graph" + str(count) + ".png"
            plt.savefig(file_name)
            count += 1
        input_data_frame.plot.area(stacked=False)
        if save_to_files:
            file_name = "graphs/graph" + str(count) + ".png"
            plt.savefig(file_name)
            count += 1
        plt.legend(loc='lower right')
        plt.show()
        return count
    except Exception as e:
        print(f'Exception message: {e}')


def main():
    # pobieranie danych z Yahoo Finances
    stocks = get_stock_data_from_file(config['stockFilePath'])
    counter = 0
    data_frame = get_data_from_yahoo(stocks)
    # zapisanie do pliku CSV
    data_frame.to_csv("stock.csv")
    # Przygotowywanie wizualizacji pobranych danych
    counter = show_graphs(data_frame, bool(config['performGraphSave']), counter)
    for stock in stocks:
        CURRENT_STOCK = stock
        data_frame = get_data_from_yahoo(stock)
        # Wizualizacja przewidywanego zachowania giełdy na 60 dni do przodu
        train, valid = predict_stock_moves_ML(data_frame)
        counter = visualize_predicted_data(train, valid, bool(config['performGraphSave']), counter, stock)
        # Wyświetlenie wykresów
        plt.show()
        counter += 1
    save_rmse_to_txt(config['rmseFileName'])


if __name__ == "__main__":
    main()
