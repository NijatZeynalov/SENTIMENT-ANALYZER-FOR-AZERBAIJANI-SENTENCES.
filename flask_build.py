from flask import Flask, render_template,request
import pickle
app = Flask(__name__)

model = pickle.load(open('sent_azerbaijani.pkl', 'rb'))

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
   review =request.form['experience']
   tokens = clean_doc(review)
   tokens = [w for w in tokens if w in vocab]
   # convert to line
   line = ' '.join(tokens)
   # encode
   encoded = tokenizer.texts_to_matrix([line], mode='binary')
   yhat = model.predict(encoded, verbose=0)
   # retrieve predicted percentage and label
   percent_pos = yhat[0, 0]
   if round(percent_pos) == 0:
      return render_template('index.html', prediction_text='Sentiment value {}'.format('NEGATIVE'), second_text = 'Sentiment percent: {}'.format(1-percent_pos))
   else:
      return render_template('index.html', prediction_text='Sentiment value {}'.format('POSITIVE'),
                             second_text='Sentiment percent: {}'.format(percent_pos))
if __name__ == '__main__':
   app.run( threaded=False)