import tensorflow as tf
from flask import *
import numpy
import pandas as pd
numpy.random.seed(10)
import os
import fileinput

new_model = tf.keras.models.load_model('spectrum_milk_freshness_detection_0929.h5')
app = Flask(__name__)

default_data=pd.read_excel("default_data.xlsx")
default_data_array=default_data.values

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', **locals())

@app.route('/predict', methods=['GET', 'POST'])
def predict():
     if request.method == 'POST':
             file = request.files['inputfile']
             userdata = pd.read_csv(file)
             userdata_array=userdata.values
             inputdata=numpy.vstack([default_data_array,userdata_array])
             x_max=inputdata.max(axis=0)
             x_min=inputdata.min(axis=0)
             userdata_reshape=numpy.reshape(userdata_array,(121))
             normal_data=userdata_reshape
             for i in range(121):
                 std_data=(userdata_reshape[i]-x_min[i])/(x_max[i]-x_min[i])
                 normal_data[i]=std_data

             normal_data=numpy.reshape(normal_data,(1,121))
             predictions = new_model.predict(normal_data)
             print(predictions)
             if (predictions[0][0]>0.5): 
                 results='fresh'
             else:
                 results='unfresh'
    
     return render_template('index.html',results=results)

if __name__ == '__main__':
    app.run(debug=True)



##wording table
#df is the default data
#userdata is the uploaded file
#inputdata is the userdata and df stacked