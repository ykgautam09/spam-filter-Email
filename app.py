from flask import Flask, request, render_template, jsonify
from model import predict_mail
from configparser import ConfigParser

config = ConfigParser()
config.read('./config.ini')
app = Flask(__name__)

DEFAULT_EMAIL = config['DEFAULT']['DEFAULT_EMAIL']
AS_API = bool(config['DEFAULT']['AS_API'])


@app.route('/')
def nlp_route():
    return render_template('spamDetection.html', size=0)


@app.route('/', methods=['POST'])
def cosine_model():
    title = request.form.get('email', DEFAULT_EMAIL)
    out = predict_mail(str(title))
    print(out)
    if out:
        result = 'SPAM'
    else:
        result = 'Not-SPAM'

    if AS_API:
        return jsonify(result)
    print(result, len(result))
    return render_template('spamDetection.html', size=len(out), spam=result)


if __name__ == '__main__':
    app.run(port='5000', debug=True)
