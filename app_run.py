import tensorflow as tf
from flask import *
import numpy
import pandas as pd
from sklearn import preprocessing
numpy.random.seed(10)
import os
import fileinput

new_model = tf.keras.models.load_model('spectrum_milk_freshness_detection_0919v1.h5')
app = Flask(__name__)

def Standardize(Features):
    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaledFeatures=minmax_scale.fit_transform(Features) 
    return scaledFeatures

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
             test_df = pd.read_excel(file)
             test1=numpy.vstack([df1,test_df])
             std_test1=Standardize(test1)
             predictions = new_model.predict(std_test1)
             if (predictions[10]>0.5): 
                 results='fresh'
             else:
                 results='unfresh'
        
     return render_template('index.html',results=results)

if __name__ == '__main__':
    app.run(debug=True)
