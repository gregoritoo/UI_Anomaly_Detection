import os
import numpy as np
import pandas as pd
from influxdb import InfluxDBClient
import configparser
import matplotlib.pyplot as plt
from datetime import datetime

INFLUX_NAME = os.environ['INFLUX_NAME']
INFLUX_PASSWORD = os.environ['INFLUX_PASSWORD']
INFLUX_PORT = 443
INFLUX_HOST = "influxdb-monitoring.supralog.com"
INFLUX_SSL = True
INFLUX_VERIFY_SSL = True

class Query():
    '''
    Mother class for requests
    '''

    def __init__(self, field, db):
        self.field = field
        self.db = db
        self.client = self.connect_database()

    def connect_database(self):
        client = InfluxDBClient(host=INFLUX_HOST, port=INFLUX_PORT, username=INFLUX_NAME, password=INFLUX_PASSWORD,
                                ssl=INFLUX_SSL, verify_ssl=INFLUX_VERIFY_SSL)
        client.switch_database(self.db)
        self.client = client

    def make_query(self, measurement, every, cond, gb, period, client):
        '''
            Parameters

            ----------

            typo : str

                way to aggregate values during the window considered  ex mean,sum, max.

            db : str

                Influx database's name.

            measurement : str

                Database's measurement to study.

            every : str

                frequency at which the request is made (duration in influxdb syntax ex "5m").

            period : str

                window to be concidered (duration in influxdb syntax ex "5m").

            gb : str

               group by condition of the requeste to write

                ex "host,element_to_groupby," => separte element with "y" and always end with ","

            cond : str

                condition to add to the request .

                Start with "AND " condition in influxdb syntax without the WHERE

                    ex AND host="GSCA"

            Returns

            -------

            results : request.response

               result of the request

        '''
        if measurement == "healthCheck" :
            query = 'SELECT ' + self.field + ' FROM "' + self.db + '"."autogen".' + measurement + ' WHERE "time" > ' + every + cond + ' GROUP BY ' + gb[: -1]
        elif measurement == "appErrors" :
            query = 'SELECT ' + "(err"+') FROM "' + self.db + '"."autogen".' + measurement + ' WHERE "time" > ' + every + cond + ' GROUP BY ' +  gb[: -1]
        else :
            query = 'SELECT ' + self.typo + "(" + self.field + ') FROM "' + self.db + '"."autogen".' + measurement + ' WHERE "time" > ' + every + cond + ' GROUP BY ' + gb + 'time(' + period + ') fill(previous)'
        print(query)
        self.gb = gb
        results = client.query(query)
        return pd.DataFrame(results.get_points(measurement=measurement))


    def request(self, every, period, cond, gb):
        client = self.connect_database()
        df = self.make_query(self.measurement, every, cond, gb, period, self.client)
        return df, self.client



class Query_all(Query):
    '''
        Child class for requesting memory's data
        '''

    def __init__(self, db, host, typo, field, measurement):
        self.host = host
        self.measurement = measurement
        self.field = field
        self.typo = typo
        Query.__init__(self, self.field, db)



