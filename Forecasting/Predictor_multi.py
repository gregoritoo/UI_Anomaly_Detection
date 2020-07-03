
import numpy as np
import os
from prediction_functions import decoupe_dataframe
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping,Callback


class EarlyStoppingByUnderVal(Callback):
    '''
    Class to stop model's training earlier if the value we monitor(monitor) goes under a threshold (value)
    replace usual callbacks functions from keras 
    '''
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            print("error")

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True


class Predictor ():
    
    def __init__(self):
        print("Creating object")


    def prepare_data(self,df,look_back,freq_period):
        '''  
        Parameters
        ----------
        df : DataFrame
            datafrmae contening historical data .
        look_back : int
            length entry of the model .
        
        Decompose the signal in three sub signals, trend,seasonal and residual in order to work separetly on each signal

        Returns
        -------
        trend_x : array
             values of the trend of the signal, matrix of dimention X array of dimension (1,length entry of model) X= length(dataframe)/look_back.
        trend_y : array
            vaklues to be predicted during the training
        seasonal_x : array
            same as trend_x but with the seasonal part of the signal.
        seasonal_y : array
            same as trend_y but with the seasonal part of the signal.
        residual_x : array
            same as trend_x but with the residual part of the signal.
        residual_y : array
            same as trend_y but with the residual part of the signal.
        '''
        self.look_back=look_back
        df=df.dropna()
        decomposition = seasonal_decompose(df["y"], period = freq_period)
        df.loc[:,'trend'] = decomposition.trend
        df.loc[:,'seasonal'] = decomposition.seasonal
        df.loc[:,'residual'] = decomposition.resid
        df["trend"]= df['trend']
        df["seasonal"]=df['seasonal']
        df["residual"]=df['residual']
        df_a=df
        df=df.dropna()
        self.shift=len(df_a)-len(df)
        df=df.reset_index(drop=True)
        df["trend"]= df["trend"].fillna(method="bfill")        
        self.trend=np.asarray( df.loc[:,'trend'])
        self.seasonal=np.asarray( df.loc[:,'seasonal'])
        self.residual=np.asarray( df.loc[:,'residual'])
        trend_x,trend_y=decoupe_dataframe(df["trend"], look_back)
        seasonal_x,seasonal_y=decoupe_dataframe(df["seasonal"], look_back)
        residual_x,residual_y=decoupe_dataframe(df["residual"], look_back)
        print("prepared")
        return trend_x, trend_y,seasonal_x,seasonal_y,residual_x,residual_y
    
    
    def train_model(self,model,x_train,y_train,nb_epochs,nb_batch,name):
        '''   
        Train the model and save it in the right file
        
        Parameters
        ----------
        model : Sequential object
            model.
        x_train : array
            training data inputs.
        y_train : array
            training data ouputs.
        nb_epochs : int.
            nb of training repetitions.
        nb_batch : int
            size of batch of element which gonna enter in the model before doing a back propagation.
        trend : bool
            Distinguish trend signal from others (more complicated to modelise).
        
        Returns
        -------
        model : Sequential object
            model.

        '''
        if name=="trend" : 
            nb_epochs=nb_epochs*7
            x_train=x_train.reshape(x_train.shape[0],1,x_train.shape[1])            	
            es = EarlyStopping(monitor='loss', mode='min', min_delta=0.1 ,patience = 200)
            hist=model.fit(x_train,y_train,epochs=nb_epochs,batch_size=nb_batch,verbose=0,callbacks=[es])
            i=0
            while hist.history["loss"][-1] > 2 and i <5:    
                i=i+1
                epochs=50
                hist=model.fit(x_train,y_train,epochs=epochs,batch_size=100,verbose=0,callbacks=[es])
            print("model_trained")
            self.model_save(model,name)
        elif  name=="residual" : 
            nb_epochs=nb_epochs*2
            x_train=x_train.reshape(x_train.shape[0],1,x_train.shape[1])         	
            es = EarlyStopping(monitor='loss', mode='min', min_delta=0.1 ,patience = 200)
            hist=model.fit(x_train,y_train,epochs=nb_epochs,batch_size=nb_batch,verbose=0,callbacks=[es])
            i=0
            self.model_save(model,name)
        else :
            es=EarlyStoppingByUnderVal(monitor="loss", value=0.0000005, verbose=0)
            x_train=x_train.reshape(x_train.shape[0],1,x_train.shape[1])
            model.fit(x_train,y_train,epochs=nb_epochs,batch_size=nb_batch,verbose=0)
            print("model_trained")
            self.model_save(model,name)
        return model
    
    def model_save(self,model,name) :
       file=""
       print(self.form)
       if type(self.form) != list :
           form=self.form[1 :].split(",")
       else :
           form = self.form
       try :
           for element in form :
               value=element.split("=")
               file=file+'_'+value[1]
           file=file[1 :].replace(":","")
       except Exception :
           file=self.host
       file=file.replace(" ","")
       path="Modeles/"+file+"_"+self.measurement
       if not os.path.isdir(path) :
           os.makedirs(path)
       path="Modeles/"+file+"_"+self.measurement+"/"+name+".h5"
       save_model(model,path)


    def make_prediction(self,len_prediction):
        lower,upper=self.frame_prediction(self.prediction)
        return prediction,lower,upper
    
    def frame_prediction(self,prediction):
        '''
        This function compute the 95% confident interval by calculating the standard deviation and the mean of the residual(Gaussian like distribution)
        and return yestimated +- 1.96*CI +mean (1.96 to have 95%)

        Parameters
        ----------
        prediction : array
            array contening the perdicted vector (1,N) size.

        Returns
        -------
        lower : array
            array contening the lowers boundry values (1,N) size.
        upper : array
            array contening the upper boundry values (1,N) size.

        '''
        mae=-1*np.mean(self.residual)
        std_deviation=np.std(self.residual)
        sc = 1.96       #1.96 for a 95% accuracy
        margin_error = mae + sc * std_deviation
        lower = prediction - margin_error
        upper = prediction + margin_error
        return lower,upper
