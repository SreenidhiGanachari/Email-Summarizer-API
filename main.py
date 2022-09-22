from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig ,BartModel
bart = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-3')
tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-3')

app = Flask(__name__)
api = Api(app)


@app.route('/', methods=['GET', 'POST'])
def home():
    if (request.method == 'GET'):
        data = "summary"
        return jsonify({'summary': data})


@app.route('/home/<string:text>', methods=['GET'])
def disp(text):
    inputs = tokenizer.batch_encode_plus([text], return_tensors='pt')
    summary_ids = bart.generate(inputs['input_ids'], num_beams=int(55), length_penalty=float(55), max_length=int(1042), min_length=int(50), no_repeat_ngram_size=int(10))
    summary = tokenizer.decode(summary_ids.squeeze(), skip_special_tokens=True)
    #print(summary)
    return jsonify({'summary': summary})


if __name__ == '__main__':
    app.run(debug=True)
