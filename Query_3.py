
import pandas as pd
from Functions.functions_interface import connect_database

class Query():
    '''
    Mother class for requests
    '''

    def __init__(self, field, db):
        self.field = field
        self.db = db
        self.client = connect_database(db)

    def make_query(self, measurement, every, cond, gb, period, client, dic):
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
            query = 'SELECT ' + self.typo + "(" + self.field + ') FROM "' + self.db + '"."autogen".' + measurement + ' WHERE "time" > ' + every + cond + ' GROUP BY ' + gb + 'time(' + period + ') fill(0)'
        print(query)
        self.gb = gb
        results = client.query(query)
        return pd.DataFrame(results.get_points(measurement=measurement))


    def request(self, every, period, cond, gb, dic):
        client = connect_database(self.db)
        df = self.make_query(self.measurement, every, cond, gb, period, self.client, dic)
        return df, client



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

