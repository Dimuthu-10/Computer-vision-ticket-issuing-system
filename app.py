from flask import Flask, render_template,request

app = Flask(__name__)

@app.route('/', methods=['GET'])
def helloWorld():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def prediction():
    predict_img = request.files['predict_image']
    save_path = "./images/"+ predict_img.filename
    predict_img.save(save_path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(port=3000, debug=True)