from flask import Flask,render_template,url_for,request,redirect
from keras.models import load_model
import base64
from PIL import Image
import cv2
import numpy as np
model1=load_model(r"D:\Data Science\Model_Details\VGGfilter.h5")
app=Flask(__name__)
@app.route('/predict',methods=['POST','GET'])
def page1():
    if(request.method=='POST'):
        decoded=base64.b64decode(request.form['64'].split(',')[1])
        im_arr = np.frombuffer(decoded, dtype=np.uint8)  # im_arr is one-dim Numpy array
        print(im_arr)
        img = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)#
        img=cv2.resize(img,(300,300))
        img=np.array(img).astype('float32')/255.0
        img=np.expand_dims(img,axis=0)
        return render_template('modelpage.html',result1=np.argmax(model1.predict(img),axis=1).astype('int'))
    else:
        return 'No Image'
@app.route('/',methods=['GET','POST'])
def landingpage():
    if(request.method=='POST'):
        pass
    else:
        return render_template('modelpage.html')



if __name__=='__main__':
    app.run(debug=True)