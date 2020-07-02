import os

path_to_kap = os.environ['kapacitor']


class Alert():
    def __init__(self, host, measurement):
        self.host = host
        self.measurement = measurement
        self.texte = ""

    def create(self, message, form, pas, path_to_model, field, typo, db):
        self.form = form
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
        where_condition = ""
        where = [[element.split("=") for element in form[1:].split(",")][i][0] for i in range(len(form[1:].split(",")))]
        for ele in where:
            where_condition = where_condition + ele + "=" + ele + " AND "
        texte = ""
        tags = ""
        cond = ["var " + (form[1:].replace(",", " AND").split("AND")[i]).replace("=", "='") + "'" for i in
                range(len(form[1:].replace(",", " AND").split("AND")))]
        for element in cond:
            texte = texte + element + "\n"
            tags = tags + ".tag('" + element.split("=")[0] + "'," + element.split("=")[1] + ")\n"
        texte = texte + "\n\n" + """var data = batch
                |query('SELECT """ + typo + """(*) FROM """ + db + """."autogen".""" + self.measurement + """ WHERE """ + where_condition[
                                                                                                                          : -5] + """')
                    .period(""" + str(int(25 * pas)) + """m)
                    .every(""" + str(int(pas)) + """m)
                    .groupBy(time(""" + str(int(pas)) + """m))

                data
                  @udf_test()
                    .field('""" + typo + """_""" + field + """')
                    .size(25)
                    .model('""" + path_to_model + """')
                  |alert()
                    .crit(lambda: "val_anomalies" > 10)
                    .message('""" + message + """')
                    .slack()
                  |InfluxDBOut()
                        .database('""" + db + """')
                        .retentionPolicy('autogen')
                        .measurement('anomalies')
                        .tag('measurement','""" + self.measurement + """')\n
                        """ + tags.replace("var ", "")

        self.texte = texte

    def save_alert(self):
        self.form = self.form[1:].replace("=", ".")
        self.form = self.form.replace(",", "_")
        self.form = self.form.replace(":", "")
        self.path = r"Alerte/alerte_" + self.measurement + "_" + self.form + ".tick"
        self.path = self.path.replace(" ", "")
        print(self.path)
        with open(self.path, "w") as f:
            f.write(self.texte)
        f.close()

    def define_alert(self):
        self.form = self.form.replace("=", ".")
        self.form = self.form.replace(",", "_")
        self.form = self.form.replace(":", "")
        cmd_define_alert = path_to_kap + " define " + "alerte_AD_" + self.measurement + "_" + self.form + " -type batch -tick " + self.path + " -dbrp telegraf.autogen"
        print(cmd_define_alert)
        os.system('cmd /c ' + cmd_define_alert)

    def enable_alert(self):
        self.form = self.form.replace("=", ".")
        self.form = self.form.replace(",", "_")
        self.form = self.form.replace(":", "")
        cmd_enable_alert = path_to_kap + " enable " + "alerte_AD_" + self.measurement + "_" + self.form
        os.system('cmd /c ' + cmd_enable_alert)

    def launch(self):
        self.define_alert()
        self.enable_alert()

