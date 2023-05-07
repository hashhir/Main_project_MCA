from cProfile import label
from pyexpat.errors import messages
from flask import Flask, render_template, request, redirect, url_for, send_file, abort
from werkzeug.utils import secure_filename
import os
from pridiction import predict

app = Flask(__name__)


@app.route("/")
def FUN_root():
    return render_template("index.html")


@app.route("/google")
def google():
    return render_template("in.html")


@app.route('/image')
def get_image():
    if request.args.get('name') != '':
        filename = request.args.get('name')
    return send_file(filename, mimetype='image/jpg')


@app.route('/', methods=['POST'])
def upload_files():
    uploaded_file = request.files['image_file']
    filename = secure_filename(uploaded_file.filename)
    if filename == '':
        abort(400)
        # file_ext = os.path.splitext(filename)[1]
        # if file_ext not in app.config['UPLOAD_EXTENSIONS']:
        #     abort(400)
    file = "./upload/" + filename
    if not os.path.exists(file):
        uploaded_file.save(file)

    label = predict(file)

    # print(dir)
    # return redirect(url_for('index.htm?dir={}&{}={}&{}={}&{}={}&'.format(
    # dir,labels[0],sums[0],labels[1],sums[1],labels[2],sums[2])))
    
    return render_template("result.html",
                           file=file,
                           labels=label)


def getPrediction(image='./upload/BloodImage.jpg'):
    return run(weights='./models/bccd.pt', source=image, imgsz=(640, 480), view_img=False, line_thickness=1,
               project='./Detections', max_det=30)


if __name__ == "__main__":
    app.run(debug=True,
            # host="0.0.0.0"
        )
