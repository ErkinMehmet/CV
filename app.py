import os
import cv2
from funcs import read_image, read_video, read_webcam
from flask import Flask,render_template,request,url_for,redirect
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input

# flask setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True )

# load the trained model
MODEL_PATH = 'model/cnn_model.h5'
model = load_model(MODEL_PATH)

# replace with your actual class names in order of model outputs
class_names=['Oblique fracture','Spiral fracture']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in {'png','jpg','jpeg','gif'}

@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        file=request.files.get('file')
        if file and allowed_file(file.filename):
            filepath=os.path.join(app.config['UPLOAD_FOLDER'],file.filename)
            file.save(filepath)

            # preprocess the image
            image=load_img(filepath,target_size=(224,224))
            image=img_to_array(image)
            image=np.expand_dims(image,axis=0)
            image=preprocess_input(image)

            # make prediction
            preds=model.predict(image)
            idx=np.argmax(preds[0])
            pred_class=class_names[idx]
            confidence=preds[0][idx]
            return render_template('index.html', filename=file.filename,label=pred_class, confidence=f" {confidence*100:.2f}")
        return redirect(request.url)
    return render_template('index.html')        

@app.route('/uploads/<filename>')
def send_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    # test reading images, videos, webcam
    # read_image("resources/image.jpg")
    # read_video("resources/video.mp4")
    #read_webcam()
    app.run(debug=True,host='0.0.0.0',port=8080)

