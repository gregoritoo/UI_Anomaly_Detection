from kapacitor.udf.agent import Agent, Handler
from kapacitor.udf import udf_pb2
import sys
import json
import pandas as pd
import numpy as np
from  adtk.src.adtk.data import validate_series
import pickle
import math
import adtk.src.adtk.detector
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger()


class Anomalies_detection(Handler):
    def __init__(self, agent):
        """Constructor"""
        logger.info('__init__ trigger')
        self._agent = agent
        self._field = ''
        self._size = 25
        self._points = []
        self._time=[]
        self._state = {}
        self.model=""

    def info(self):
        """info: Define what your UDF wants and what will it provide in the end"""
        logger.info('info trigger')
        response = udf_pb2.Response()
        response.info.wants = udf_pb2.BATCH
        response.info.provides = udf_pb2.STREAM
        response.info.options['field'].valueTypes.append(udf_pb2.STRING)
        response.info.options['size'].valueTypes.append(udf_pb2.INT)
        response.info.options['model'].valueTypes.append(udf_pb2.STRING)
        return response

    def init(self, init_req):
        """init: Define what your UDF expects as parameters when parsing the TICKScript"""
        logger.info('INIT trigger')
        for opt in init_req.options:
            if opt.name == 'field':
                self._field = opt.values[0].stringValue
            elif opt.name == 'size':
                self._size = opt.values[0].intValue
            elif opt.name == 'model':
                self.model = opt.values[0].stringValue
        success = True
        msg = ' must provides info'
        if self._field == '':
            success = False
            msg = 'must provide field name'
        response = udf_pb2.Response()
        response.init.success = success
        response.init.error = msg.encode()
        return response

    def begin_batch(self, begin_req):
        """begin_batch: Do something at the beginning of the batch"""
        self._batch=AD_model(self._size,self.model)
        response = udf_pb2.Response()
        response.begin.CopyFrom(begin_req)
        self._begin_response = response
        logger.info('begin_batch trigger')

    def point(self, point):
        """point: process each point within the batch"""
        logger.info('point trigger')
        self._batch.update(point.fieldsDouble[self._field],point.time)

    def snapshot(self):
        """snapshot: take a snapshot of the current data, if the task stops for some reason """
        data = {}
        for group, state in self._state.items():
            data[group] = state.snapshot()
        response = udf_pb2.Response()
        response.snapshot.snapshot = json.dumps(data).encode()
        return response

    def restore(self, restore_req):
        response = udf_pb2.Response()
        response.restore.success = True
        return response

        
    def end_batch(self, batch_meta):
        """end_batch: do something at the end of the batch"""
        results=self._batch.test()
        point=results[-1]
        resp = udf_pb2.Response()
        resp.point.name = batch_meta.name
        resp.point.time = self._batch.get_time(len(results)-1)
        resp.point.group = batch_meta.group
        resp.point.tags.update(batch_meta.tags)
        resp.point.fieldsString['metrics'] = "anomalies"
        resp.point.fieldsDouble["value"] = self._batch.get_value(len(results)-1)
        if point ==  1.0 or point == True :
            resp.point.fieldsDouble['val_anomalies'] = 100
        else :
            resp.point.fieldsDouble['val_anomalies'] = 1
        if type(resp != None) :
            self._agent.write_response(resp)
        else :
            logger.info('pbbbbbb',resp)
        logger.info('end_batch')

    def end_batch_3(self, batch_meta):
        """end_batch: do something at the end of the batch"""
        results=self._batch.test()
        p=0
        for point in results:            
            response = udf_pb2.Response()
            response.point.name = batch_meta.name
            response.point.time = self._batch.get_time(p)
            response.point.group = batch_meta.group
            response.point.tags.update(batch_meta.tags)
            response.point.fieldsString['metrics'] = "anomalies"
            response.point.fieldsDouble["value"] = self._batch.get_value(p)
            if point ==  1.0 or point == True :
                response.point.fieldsDouble['val_anomalies'] = 100
            else :
                response.point.fieldsDouble['val_anomalies'] = 1
            p=p+1
            self._agent.write_response(response)   
        logger.info('end_batch')


class AD_model(object):

    def __init__(self,size,model):
        self.size=size
        self._date=[]
        self._value=[]
        self.model=model
        with open(self.model, 'rb') as f:
            self.model = pickle.load(f)
            
    def update(self,value,time) :
        self._value.append(value)
        self._date.append(time)
       

    def test(self):
        self.df=pd.DataFrame({'time':self._date,'y': self._value})
        self.df=self.df.fillna(0)
        self.df=self.df.reset_index(drop=True)
        self.df["time"]=self.df["time"].astype('Int64')        
        self.df=self.df.set_index("time")
        self.df.index=pd.to_datetime(self.df.index)        
        s = validate_series(self.df["y"])                   
        val=self.model.detect(s, return_list=False)
        logger.info("les res",val)
        return val

    def get_value(self,p):
        return self._value[p]
    
    def get_time(self,p):
        return self._date[p]
    
if __name__ == '__main__':
    # Create an agent
    agent = Agent()

    # Create a handler and pass it an agent so it can write points
    h = Anomalies_detection(agent)

    # Set the handler on the agent
    agent.handler = h

    # anything printed to STDERR from a UDF process gets captured into the logs
    logger.info("Starting agent for Anomalies_detection")
    agent.start()
    agent.wait()
    logger.info("Agent finished")
