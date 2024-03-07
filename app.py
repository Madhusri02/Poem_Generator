import streamlit as st 
import tensorflow as tf
import numpy as np
import random

st.title("POEM GENERATOR ")

prom = st.chat_input('Enter how the poem should be started')
st.write(prom)

file_p = tf.keras.utils.get_file('shakespeare.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(file_p , 'r',encoding = "utf-8").read().lower()
ch = sorted(set(text))
sentence = [] #feature data
next_char = [] #targeted data


ch_to_val = dict((c,i) for i ,c in enumerate(ch))
val_to_char = dict((i,c) for i,c in enumerate(ch))
# print(b_to_char)

l = 50
line_size = 10

model = tf.keras.models.load_model('poemgenerator.keras')

def sample(preds, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature 
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    prob = np.random.multinomial(1, preds , 1)
    return np.argmax(prob)

def generate_text(length , temp):
    index = random.randint(0, len(text) - line_size - 1) #here removing first 40 characters
    poem = ''
    sentence = text[index : index + line_size]
    #prediction part
    poem += sentence
    for i in range(length):
        x = np.zeros((1 , line_size ,len(ch)),dtype=bool)
        for t , char in enumerate (sentence):
            x[0,t , ch_to_val[char]] = 1
        predict = model.predict(x , verbose = 0) [0]
        next_index = sample(predict , temp)
        next_char = val_to_char[next_index]
        poem +=next_char
        sentence = sentence[1 : ] + next_char
    return poem

print(generate_text(300 , 0.6))

