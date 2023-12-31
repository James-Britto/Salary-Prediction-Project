from flask import Flask, render_template, request
import pickle
import pandas as pd
app = Flask(__name__)
processor=pickle.load(open('model_transformer.pickle','rb'))
model = pickle.load(open('salary_prediction.pickle','rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        Gender= request.form['gender']
        Education_Level= request.form['education']
        Job_Title= request.form['job_title']
        Years_of_experience= request.form['yoe']
        df=pd.DataFrame({'Years_of_Experience':[Years_of_experience],'Education_Level':[Education_Level],'Gender':[Gender],'Job_Title':[Job_Title]})
        df=processor.transform(df)
        prediction=model.predict(df)
        return render_template('index.html',prediction_output=f"Your expected salary is Rs. {round(prediction[0],2)}")
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)
