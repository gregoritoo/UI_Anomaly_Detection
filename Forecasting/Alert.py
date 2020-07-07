# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:02:06 2020

@author: GSCA
"""




import os 

path_to_kap = os.environ['kapacitor']
path_to_script = os.environ['script']


class Alert():
    def __init__(self,host,measurement) :
        self.host=host
        self.measurement=measurement       
        self.texte=""

        
    def create(self,message,form,period,typo,field):
        self.form=form
        '''
        Create the tick alert 
        Note : NEED TO DEFINE the path of the script, which will be launched when an alert is trigged, as a variable environnement

        Parameters
        ----------
        message : str
            Message to be shown as an alert on slack etc ; need to be written with kapacitor syntax
                
        Returns
        -------
        None.

        '''
        where_condition=""
        where=[[element.split("=") for element in form[1 :].split(",")][i][0]for i in range(len(form[1 :].split(",")))]
        for ele in where :
            where_condition=where_condition+ele+"="+ele +" AND "
        texte=""
        cond=["var "+(form[1 :].replace(","," AND").split("AND")[i]).replace("=","='")+"'" for i in range(len(form[1 :].replace(","," AND").split("AND")))]
        for element in cond :
            texte=texte+element+"\n"
        texte = texte +"\n\n"+ """var realtime = batch
                |query('SELECT mean(yhat) as real_value FROM "telegraf"."autogen".pred_"""+self.measurement+""" WHERE """+where_condition[: -5]+"""')
                    .period(5m)
                    .every(5m)
                    .align()
                |last('real_value')
                    .as('real_value')
                |log()
                    .prefix('P0-1')
                    .level('DEBUG')

            
            var predicted = batch
                |query('SELECT mean(yhat) as prediction FROM "telegraf"."autogen".pred_3"""+self.measurement+""" WHERE """+where_condition[: -5]+"""')
                    .period(5m)
                    .every(1h)
                    .align()
                |last('prediction')
                    .as('prediction')
                |log()
                    .prefix('P0-2')
                    .level('DEBUG')
            
            
            var joined_data = realtime
                |join(predicted)
                    .as('realtime', 'predicted')
                    .tolerance(20m)
            
            
            var performance_error = joined_data
                |eval(lambda: abs("realtime.real_value" - "predicted.prediction"))
                    .as('performance_error')
                |alert()
                    .crit(lambda: "performance_error" > 10 )
                    .message('""" +message+"""')
                    .slack()
                    .exec('"""+path_to_script+"""', '\""""+self.host+"\"'"+""", '\""""+self.measurement+"\"'"+""", '\""""+str(form[1 :])+"\"'"+""", '\""""+period+"\"'"+""", '\""""+typo+"\"'"+""", '\""""+field+"\"'"+""")
                |log()
                    .prefix('P0-3')
                    .level('DEBUG')"""
            
        self.texte=texte 
    def save_alert(self):
        self.form=self.form[1 :].replace("=",".")
        self.form=self.form.replace(",","_")
        self.form=self.form.replace(":","")
        self.path=r"Forecasting_Alerte/alerte_"+self.measurement+"_"+self.form+".tick"
        self.path=self.path.replace(" ","")
        print(self.path)
        with open(self.path,"w") as f :
            f.write(self.texte)
        f.close()
    
    
        
    def define_alert(self):
        self.form=self.form.replace("=",".")
        self.form=self.form.replace(",","_")
        self.form=self.form.replace(":","")
        cmd_define_alert=path_to_kap+" define "+"alerte_"+self.measurement+"_"+self.form+" -type batch -tick "+self.path+" -dbrp telegraf.autogen"
        print(cmd_define_alert)
        os.system('cmd /c '+cmd_define_alert)
        
        
    def enable_alert(self):
        self.form=self.form.replace("=",".")
        self.form=self.form.replace(",","_")
        self.form=self.form.replace(":","")
        cmd_enable_alert=path_to_kap+" enable "+"alerte_"+self.measurement+"_"+self.form
        os.system('cmd /c '+cmd_enable_alert)
        
    def launch(self):
        self.define_alert()
        self.enable_alert()

