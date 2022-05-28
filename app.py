#Import necessary libraries

from flask import Flask, render_template, request
 
import numpy as np
import os
 
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
 
#load model
model =load_model("model/v3_pred_cott_dis.h5")
 
print('@@ Model loaded')

flag = True
def startPrediction(plant):
  global flag

  test_image = load_img(plant, target_size = (150, 150)) # load image 
  print("@@ Got Image for prediction")
   
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
   
  result = model.predict(test_image).round(3) # predict diseased palnt or not
  print('@@ Raw result = ', result)
   
  pred = np.argmax(result) # get the index of max value

  flag = not flag
  if pred == 0:
    if flag:
      return "Diseased Leaf / रोगग्रस्त पत्ता", 'disease_plant_leaf_1.html' 
    else:
      return "Diseased Leaf / रोगग्रस्त पत्ता", 'disease_plant_leaf_2.html' 
  elif pred == 1:
    if flag:
      return 'Diseased Plant / रोगग्रस्त पौधा', 'disease_plant_1.html' 
    else:
      return 'Diseased Plant / रोगग्रस्त पौधा', 'disease_plant_2.html'
  elif pred == 2:
    if flag:
      return 'Healthy Leaf / स्वस्थ पत्ता', 'healthy_plant_leaf_1.html' 
    else:
      return 'Healthy Leaf / स्वस्थ पत्ता', 'healthy_plant_leaf_2.html'  
  else:
    if flag:
      return 'Healthy Plant / स्वस्थ पौधा', 'healthy_plant_1.html'  
    else:
      return 'Healthy Plant / स्वस्थ पौधा', 'healthy_plant_2.html'  
 
#------------>>startPrediction<<--end
     
 
# Create flask instance
app = Flask(__name__)
 
# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
     
  
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fetch input
        filename = file.filename        
        print("@@ Input posted = ", filename)
         
        file_path = os.path.join('static/user uploaded images', filename)
        file.save(file_path)
 
        print("@@ Predicting class......")
        pred, output_page = startPrediction(plant=file_path)
               
        return render_template(output_page, pred_output = pred, user_image = file_path)
     
# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False) 

    