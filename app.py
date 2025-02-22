import streamlit as st
import requests
from transformers import pipeline

# Configure Streamlit
st.set_page_config(page_title="AI Healthcare Assistant", page_icon="🤖🏥", layout="wide")

# Load a medically reliable model
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-large")

model = load_model()

def fetch_medical_context(query: str) -> str:
    """
    Fetch reliable medical information using Google Custom Search API (Better than MedlinePlus)
    """
    try:
        GOOGLE_API_KEY = "AIzaSyBxphvqMeyttf8GOOKfYEgcMBH00bHnT0A"
        SEARCH_ENGINE_ID = "91c0096f6f54c48db"
        search_url = f"https://www.googleapis.com/customsearch/v1?q={query}+health&cx={SEARCH_ENGINE_ID}&key={GOOGLE_API_KEY}"
       
        response = requests.get(search_url)
        if response.status_code == 200:
            data = response.json()
            if "items" in data:
                for item in data["items"]:
                    title = item["title"]
                    snippet = item["snippet"]
                    link = item["link"]
                    return f"**{title}**\n\n{snippet}\n\nRead more: {link}"
       
        return "I'm sorry, I couldn't find precise medical data for your question. Please consult a medical professional."
   
    except Exception as e:
        return f"Error fetching medical data: {e}"

# Streamlit App Layout
st.title("🤖🏥 AI-Powered Healthcare Assistant")
st.write("Enter a medical query, and I'll provide an in-depth response.")

# Input query
user_input = st.text_area("Enter your question:", placeholder="e.g., How to maintain blood pressure?")

# Toggle for short or detailed answers
detailed_response = st.checkbox("Provide a detailed response")

if st.button("Get Answer"):
    if user_input:
        with st.spinner("Fetching reliable medical data..."):
            context = fetch_medical_context(user_input)

        if "Error" in context:
            st.error(context)
        else:
            with st.spinner("Generating response..."):
                try:
                    prompt = (
                        f"You are a medical AI assistant. Provide a detailed, well-structured response to this question:\n\n"
                        f"**Question:** {user_input}\n\n"
                        f"**Medical Context:**\n{context}\n\n"
                        f"Your response should be clear, informative, and medically accurate."
                    )

                    response = model(
                        prompt, 
                        max_length=1500 if detailed_response else 500, 
                        min_length=300,  # Ensures a detailed response
                        do_sample=True, 
                        top_p=0.95,  # Allows more diverse responses
                        temperature=0.9  # Encourages a bit more variation
                    )  

                    answer = response[0]['generated_text'].strip()

                    # Display enhanced response
                    st.success("**Healthcare Assistant Response:**")
                    st.markdown(f"### 🏥 {answer}")  # Styled response for clarity
                except Exception as e:
                    st.error(f"Error generating response: {e}")
    else:
        st.warning("Please enter a medical query.")
