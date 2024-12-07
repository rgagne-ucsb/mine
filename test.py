from flask import Flask, render_template, request, send_from_directory,jsonify
import os
import google.generativeai as genai
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

df = pd.read_csv('data/cleanin.csv')

# Configure the Gemini API
os.environ["GOOGLE_API_KEY"] = "AIzaSyA57pNgX_pIeLpZcyu58elYo2e3b6bp7d0"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

@app.route('/skill/<path:filename>')
def serve_skill(filename):
    return send_from_directory('skill', filename)

@app.route("/")
def home():
    return render_template("test.html")

@app.route("/rhyme/<rhyme_id>")
def show_rhyme(rhyme_id):
    model = genai.GenerativeModel('gemini-1.5-flash')
    chat = model.start_chat(
    history=[
        {"role": "user", "parts": "Hello, I would like to learn new skill."},
        {"role": "model", "parts": "Great to meet you. What would you like to learn?"},
    ]
    )
    response = chat.send_message(["Give me a 60 days plan for learning the following skill, make it 2 months strictly,no more than 8 weeks, no need to include anything else apart from week plan not even a single word, make it in week wise manner, with days in half manner like in week 1, days 1-4 do this in bullet points(make 2 bullet points per week, mention days), start with week 1, career name: ", rhyme_id], stream=True)
    for chunk in response:
        print(chunk.text)
        print("_" * 200)

    rhyme = response.text
    print(chat.history)
    return render_template("rhyme.html", rhyme=rhyme)

@app.route("/job-match-ai", methods=["GET", "POST"])
def job_match_ai():
    if request.method == "POST":
        # Get user input from form
        user_input = request.form["skills"]
        user_skills = set(map(str.strip, user_input.lower().split(",")))  # Convert to lowercase and split

        # Prepare job skills data (assuming 'df' contains job descriptions and 'job_skills' column)
        df = pd.read_csv('data/cleanin.csv')

        # Vectorize job descriptions and user skills
        tfidf_vectorizer = TfidfVectorizer(stop_words="english")

        # Combine job skills with user skills for vectorization
        job_skills = df['job_skills'].fillna("")  # Handling missing job skills
        user_skills_str = ', '.join(user_skills)

        # Create the TF-IDF matrix
        all_skills = job_skills.tolist() + [user_skills_str]
        tfidf_matrix = tfidf_vectorizer.fit_transform(all_skills)

        # Calculate cosine similarity between user skills and job descriptions
        cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])

        # Add cosine similarity to the DataFrame
        df['similarity'] = cosine_sim.flatten()

# Function to calculate matched and missing skills
    #     def calculate_skills_match(row, user_skills):
    # # Ensure job_skills is a string, handle NaN values gracefully
    #         job_skills_set = set(map(str.strip, str(row["job_skills"]).lower().split(",")))
    #         matched_skills = user_skills.intersection(job_skills_set)
    #         missing_skills = job_skills_set.difference(user_skills)
    #         return pd.Series({
    #             "matched_skills": ", ".join(matched_skills),
    #             "missing_skills": ", ".join(missing_skills)
    #         })

        def calculate_skills_match(row, user_skills):
    # Ensure job_skills is a string, handle NaN values gracefully
            job_skills_set = set(map(str.strip, str(row["job_skills"]).lower().split(",")))
            matched_skills = user_skills.intersection(job_skills_set)
            missing_skills = job_skills_set.difference(user_skills)
            return pd.Series({
                "matched_skills": ", ".join(matched_skills),
                "missing_skills": ", ".join(missing_skills)
            })




        # Apply the function to the DataFrame to calculate matched and missing skills
        df[["matched_skills", "missing_skills"]] = df.apply(calculate_skills_match, user_skills=user_skills, axis=1)

        # Filter and sort results
        matched_jobs = df[df['similarity'] > 0].sort_values(by="similarity", ascending=False).head(5)

        # Prepare results for rendering
        results = matched_jobs[["job_link", "job_skills", "matched_skills", "missing_skills", "similarity"]].to_dict("records")
        
        return render_template("money.html", results=results)

    return render_template("job_match.html")


if __name__ == "__main__":
    app.run(debug=True)
