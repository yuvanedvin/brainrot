from flask_cors import CORS
from flask import Flask, request, jsonify, render_template
import requests
import os
from transformers import BertTokenizer, BertForSequenceClassification
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
import os


load_dotenv()  # Load environment variables from .env

MODEL_PATH = "distilbert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=3)
app = Flask(__name__, template_folder='templates')
CORS(app)

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY") # Ensure GOOGLE_API_KEY is in .env, not GEMINI_API_KEY
genai.configure(api_key="AIzaSyDhpYZU45rGgVao4cshIamSHkjAd7DKvyw")
gemini_model = genai.GenerativeModel("gemini-2.0-flash")


AGE_QUESTIONS = {
    "10-18": ["How many hours do you spend on social media daily?", "Do you feel addicted to online content?"],
    "19-30": ["Do you feel mentally exhausted after work?", "Do you have trouble focusing?"],
    "31-50": ["Do you experience forgetfulness?", "Do you struggle with conversations?"],
    "50+": ["Do you engage in mental exercises?", "Do you prefer TV over interactions?"]
}

@app.route("/get_questions", methods=["POST"])
def get_questions():
    data = request.get_json()
    age_group = data.get("age_group", "")

    if age_group not in AGE_QUESTIONS:
        return jsonify({"error": "Invalid age group"}), 400

    return jsonify({"questions": AGE_QUESTIONS[age_group]})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    age_group = data.get("age_group", "")
    user_response = data.get("text", "")

    if not age_group or not user_response:
        return jsonify({"error": "Missing age group or response"}), 400

    # Assume compare_with_dataset and predict_severity are defined in utils.py
    from utils import compare_with_dataset, predict_severity
    dataset_match = compare_with_dataset(user_response, age_group)
    if dataset_match:
        severity = dataset_match
    else:
        severity = predict_severity(user_response)

    recommendations = {
        "Mild": "Your cognitive health seems good. Stay engaged in learning activities!",
        "Intense": "You may be experiencing fatigue. Try taking breaks and reducing screen time.",
        "Severe": "Your cognitive health may need attention. Consider mindfulness exercises."
    }

    val=save(recommendations[severity])
    print(val)


   

    return jsonify({
        "severity": severity,
        "recommendation": recommendations[severity],
        "label": severity
    })

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/page2', methods=['POST'])
def next_page():
    global name 
    global age
    name = request.form.get('name')
    age = int(request.form.get('age'))

    if age>=10 and age<=18:
        ques1="How many hours do you spend on social media daily?"
        ques2="Do you feel addicted to online content?"
    elif age>=19 and age<=30:
        ques1="Do you feel mentally exhausted after work?"
        ques2="Do you have trouble focusing?"
    elif age>=31 and age<=50:
        ques1="Do you experience forgetfulness?"
        ques2="Do you struggle with conversations?"
    elif age>50:
        ques1="Do you engage in mental exercises?"
        ques2="Do you prefer TV over interactions?"


    # Pass the data to the next page
    return render_template('Page2.html', name=name, question1=ques1,question2=ques2)

@app.route('/page3', methods=['POST'])
def page3():
    global answer1,answer2
    answer1 = request.form.get('answer1')
    answer2 = request.form.get('answer2')
    prompt=f"Hey Gemini, I have a client who have issue regarding brain rot. His name {name} and his {age}, He feels {answer1} and {answer2}. now give me some suggestion for brainrot issue as a doctor in 180 words dont include subheading,heading and points"

    try:
        response = gemini_model.generate_content(prompt)
        gemini_response = response.text.strip()
    except Exception as e:
        gemini_response = f"Error fetching response: {str(e)}"

   
    return render_template('Page3.html', response=gemini_response)

def save(severe):
    new_data = {
            "Age_Group": age,
            "Response": answer1,
            "Severity": severe
        }
    new_data2={
            "Age_Group": age,
            "Response": answer2,
            "Severity": severe
        }
        
    try:
            # Load existing data if CSV exists, otherwise create a new DataFrame
        try:
            dataset_path = os.path.join('dataset', 'brain_rot_data.csv')
            df = pd.read_csv(dataset_path)
        except FileNotFoundError:
            df = pd.DataFrame(columns=["Age_Group", "Response", "Severity"])
        #data frame for answer1
        new_df1= pd.DataFrame([new_data])
        #dataframe for answer2
        new_df2= pd.DataFrame([new_data2])

        # to store answer1
        df = pd.concat([df, new_df1], ignore_index=True)
        # to store answer2
        df = pd.concat([df, new_df2], ignore_index=True)
        df.to_csv(dataset_path, index=False)
    except Exception as e:
        print(f"Error saving response to CSV: {e}")
        return 0




if __name__ == "__main__":
    app.run(debug=True)
