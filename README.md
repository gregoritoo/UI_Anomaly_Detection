# UI_Anomaly_Detection

This project goal is to create a UI for one click multivariate data analysis, forecasting and anomalies detection for time series on Influxdb.
The structure is divided in three main parts :

Anomaly detection and time series forecating
--------------------------------------------
The first page is for anomaly detection and forecasting. <br/>
The statistical's models used are from the adtk library. You can have have more infomation here : [https://adtk.readthedocs.io/en/stable/]. <br/>
The ML based approach for anomaly detection is a LSTM. <br/>
For forecasting, the prediction is made by stacking the results of three LSTMs. Each LSTM treat a sub signal obtained after a STL decomposition. More informations on this blog []. <br/>
Once the model saved, a kapacitor UDF and a alert are created in order to make real time anomaly detection on your database. <br/>
![First_page_1](/IMG/UI_1.JPG)
![First_page_2](/IMG/UI_2.JPG)

Same measurement but differents tags analysis
---------------------------------------------
This second page show some basic plots for several time series at the time to be able to compare values from different class in the same measurement.<br/>
![First_page_1](/IMG/UI_3.JPG)
![First_page_2](/IMG/UI_4.JPG)
![First_page_1](/IMG/UI_5.JPG)


Multivariate Analysis and  multivariate anomaly detection
---------------------------------------------------------

This third page allows the user to compare two variable from the all database by ploting some basic plots and correlation.<br/>
Some more advance models for mutlivariate alerts will be available soon.<br/>
![First_page_1](/IMG/UI_6.JPG)
