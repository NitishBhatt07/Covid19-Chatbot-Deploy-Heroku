from flask import Flask, render_template, request
import random
import json
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model


app = Flask(__name__)


lematizer = WordNetLemmatizer()

intents = json.loads(open("intents.json").read())

words = pickle.load(open("words.pkl",'rb'))
classes = pickle.load(open("classes.pkl",'rb'))


model = load_model("chatbotModel.h5")


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = (lematizer.lemmatize(word) for word in sentence_words)
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i , word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    result = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    result = [[i,r] for i , r in enumerate(result) if r > ERROR_THRESHOLD]
    
    result.sort(key=lambda x : [1] , reverse=True)
    return_list = []
    for r in result:
        return_list.append({'intent':classes[r[0]],"probibility":str(r[1])})
    
    return return_list


def get_response(intnets_list ,inent_json):
    tag = intnets_list[0]['intent']
    list_of_intents = inent_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

 
@app.route('/')
def student():
   return render_template('index.html')

@app.route('/predict',methods = ['POST'])
def predict():
       message = [str(x) for x in request.form.values()]
       ints = predict_class(message[0])
       res = get_response(ints, intents)
       return render_template('index.html',predicted_text='Bot: {}'.format(res))
       
if __name__ == '__main__':
   app.run()
