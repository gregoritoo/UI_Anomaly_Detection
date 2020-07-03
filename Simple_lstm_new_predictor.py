
import numpy as np
from Functions.functions import write_predictions, modifie_df_for_fb, make_sliced_request, decoupe_dataframe
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout,Activation
import matplotlib.pyplot as plt
from Simple_lstm_predictor import Simple_lstm_predictor


class Simple_lstm_new_predictor(Simple_lstm_predictor):

    def __init__(self, df, host, measurement, look_back, nb_layers, loss, metric, nb_features, optimizer, nb_epochs,
                 nb_batch, form, freq_period):
        Predictor.__init__(self)
        self.df = df
        self.host = host
        self.measurement = measurement
        self.form = form
        self.freq_period = freq_period
        trend_x, trend_y= self.prepare_data(df, look_back,self.freq_period,self.form)
        model_trend = self.make_models(nb_layers, loss, metric, nb_features, optimizer, True)
        self.model_trend = self.train_model(model_trend, trend_x, trend_y, nb_epochs, nb_batch, "trend")
        self.model_save(self.model_trend,"trend")




    def make_models(self, nb_layers, loss, metric, nb_features, optimizer, trend):
        '''
        Create an LSTM model depending on the parameters selected by the user

        Parameters
        ----------
        nb_layers : int
            nb of layers of the lstm model.
        loss : str
            loss of the model.
        metric : str
            metric to evaluate the model.
        nb_features : int
            size of the ouput (one for regression).
        optimizer : str
            gradient descend's optimizer.
        trend : bool
              Distinguish trend signal from others (more complicated to modelise).

        Returns
        -------
        model : Sequential object
            model
        '''
        if trend:
            nb_layers = int(nb_layers / 1)
        model = Sequential()
        model.add(LSTM(nb_layers, return_sequences=True, activation='softmax', input_shape=(nb_features, self.look_back)))
        model.add(Dropout(0.2))
        model.add(LSTM(nb_layers))
        model.add(Dropout(0.2))
        model.add(Activation('softmax'))
        model.add(Dense(int(nb_layers / 2), activation='softmax'))
        model.add(Dense(1))
        model.compile(loss=loss, optimizer=optimizer, metrics=['mse'])
        print("model_made")
        return model

    def prediction_eval(prediction, real_data):
        '''
        This functino compute and print four differents metrics (mse ,mae ,r2 and median) to evaluate accuracy of the model
        prediction and real_data need to have the same size

        Parameters
        ----------
        prediction : array
            predicted values.
        real_data : array
            real data.

        Returns
        -------
        None.

        '''
        from sklearn.metrics import mean_absolute_error as mae
        from sklearn.metrics import mean_squared_error as mse
        from sklearn.metrics import median_absolute_error as medae
        from sklearn.metrics import r2_score as r2

        print("mean_absolute_error : ", mae(real_data, prediction))
        print("mean_squared_error : ", mse(real_data, prediction))
        print("median_absolute_error : ", medae(real_data, prediction))
        print("r2_score : ", r2(real_data, prediction))

    def plot_training(self, model, df_a):
        '''
        Print one step ahead preidctidn during the training period ,  not working yet

        Parameters
        ----------
        model : Sequential object
            model
        df_a : dataframe
            dataframe of historical data before any processing.

        Returns
        -------
        None.

        '''
        if model == "trend":
            model = self.model_trend
            x_train, y_train = decoupe_dataframe(self.trend, self.look_back)
        elif model == "residual":
            model = self.model_residual
            x_train, y_train = decoupe_dataframe(self.residual, self.look_back)
        elif model == "seasonal":
            model = self.model_seasonal
            x_train, y_train = decoupe_dataframe(self.seasonal, self.look_back)

        x_train = np.reshape(x_train, (int(len(x_train) / self.look_back), 1, self.look_back))
        train_predict = model.predict(x_train)
        plt.plot(train_predict)
        plt.plot(df_a[: -self.look_back])  # plot tout df_a si on souhaite le mettre en prod
        plt.show()




