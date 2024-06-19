from flask import Flask,request,url_for,redirect,render_template
import pickle
import numpy as np

app=Flask(__name__,template_folder='templates')

model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    # int_features=[float(x) for x in request.form.values()]
    a_val=request.form["lp"]
    b_val=request.form["dp"]
    c_val=request.form["port_outgoing"]
    final=np.array([[a_val,b_val,c_val]],dtype=np.float64)
    # print(int_features)
    # print(final)
    pred=float(model.predict(final))
    # op='{0:,{1}f}'.format(pred[0][1],2)

    return render_template('index.html',prediction_text='The Predicted value is: {}'.format(pred))

if __name__ == '__main__':
    app.run(debug=True)