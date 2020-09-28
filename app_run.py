import tensorflow as tf
from flask import *
import numpy
import pandas as pd
numpy.random.seed(10)
import os
import fileinput

new_model = tf.keras.models.load_model('spectrum_milk_freshness_detection_0919v1.h5')
app = Flask(__name__)

df=pd.read_excel("test_data.xlsx")
df1=df.values

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', **locals())

@app.route('/predict', methods=['GET', 'POST'])
def predict():
     if request.method == 'POST':
             file = request.files['inputfile']
             userdata = pd.read_excel(file)
             userdata1=userdata.values
             inputdata=numpy.vstack([df1,userdata1])
             x_max=inputdata.max(axis=0)
             x_min=inputdata.min(axis=0)
             userdata1=numpy.reshape(userdata1,(121))
             for i in range(121):
                 std_data=(userdata1[i]-x_min[i])/(x_max[i]-x_min[i])
                 userdata1[i]=std_data
             userdata1=numpy.reshape(userdata1,(1,121))
             predictions = new_model.predict(userdata1)
             if (predictions>0.5): 
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
