from flask import Flask, render_template, request,redirect,url_for
import numpy as np
import faiss
import pandas as pd
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate 
from sentence_transformers import SentenceTransformer
import re

app = Flask(__name__)
model=SentenceTransformer("all-MiniLM-L6-v2")
index=faiss.read_index("email_embeddings_index.faiss")
all_embeddings=np.load("email_embeddings.npy")
# Clean text function
def clean_input_text(text):
    # Remove any non-alphanumeric characters (except spaces)
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return cleaned_text

# Initialize the LLM
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    groq_api_key="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",
    temperature=0,
)

# Create the PromptTemplate for email generation
def prom(subject,context):
    email_with_context_prompt = PromptTemplate(
        input_variables=["subject", "context"],
        template=(
            "You are an AI email generator. Based on the subject and additional context provided below, "
            "generate a professional and engaging email body in a formal tone. Make sure to integrate the context seamlessly into the email body.\n\n"
            "Subject: {subject}\n\n"
            "Dear [Recipient],\n\n"
            "I hope this email finds you well.\n\n"
            "{context}\n\n"
            "Please feel free to reach out if you have any questions or need further information. I look forward to your response.\n\n"
            "Best regards,\n"
            "[Your Name]"
        ),
    )
    return email_with_context_prompt

@app.route('/', methods=['GET', 'POST'])
def index():
    generated_email = ""
    if request.method == 'POST':
        subject = request.form.get('subject')
        context = request.form.get('context')
        
        # Clean the inputs
        subject_cleaned = clean_input_text(subject)
        context_cleaned = clean_input_text(context)
        #print("cont clean",context_cleaned)

        
        #Embed the input query (subject + context)
        query_text = subject_cleaned + " " + context_cleaned
        query_embedding = model.encode([query_text])  # Use SentenceTransformer to generate embedding
        query_embedding = np.array(query_embedding, dtype="float32").reshape(1, -1)

        index=faiss.read_index("email_embeddings_index.faiss")
        
       # Perform similarity search in FAISS index
        D, I = index.search(query_embedding, k=5)  # 'k=5' means returning top 5 closest results

        closest_bodies = []
        df = pd.read_csv("preprocess_final_text.csv")  # Assuming 'pf' contains your email data
        for i in I[0]:
            if i != -1:  # Exclude invalid results
                closest_bodies.append(df.iloc[i]['Body'])  # Assuming 'pf' contains the email data

        
        # Combine the retrieved bodies into the context
        context_combined = context_cleaned + "\n\n" + "\n".join(closest_bodies)

        input_data = {
            "subject": subject_cleaned,
            "context": context_combined
        }

        try:
            # Chain the prompt with the LLM
            email_with_context_prompt = prom(subject_cleaned, context_combined)
            chain_extract = email_with_context_prompt | llm
            result = chain_extract.invoke(input_data)
            generated_email = result.content

            return redirect(url_for('generated_email', subject=subject_cleaned, body=generated_email))
        
        except Exception as e:
            generated_email = f"Error generating email: {e}"
            return render_template('index.html', generated_email=generated_email)
    
    return render_template('index.html')

@app.route('/generated_email')
def generated_email():
    subject=request.args.get('subject')
    body=request.args.get('body')

    return render_template('generated_email.html', subject=subject, body=body)

if __name__ == "__main__":
    app.run(debug=True,port=5000)
