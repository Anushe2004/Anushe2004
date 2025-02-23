import streamlit as st
import requests
from transformers import pipeline
import re
import speech_recognition as sr

# Configure Streamlit
st.set_page_config(page_title="AI Healthcare Assistant", page_icon="🤖🏥", layout="wide")

# Load a medically reliable model
@st.cache_resource
def load_model():
    return pipeline("text2text-generation", model="google/flan-t5-large")

st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")
model = load_model()
summarizer = load_summarizer()


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
def clean_response(response: str):
    response = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', response, flags=re.IGNORECASE)

    # Remove excessive new lines and spaces
    response = re.sub(r'\n+', '\n', response).strip()
    response = re.sub(r'\s+', ' ', response)

    # Remove certain filler phrases (customize based on output)
    unwanted_phrases = [
        "As an AI language model,", "I'm here to help,", "I can provide information, but", 
        "However,", "It is important to note that", "Please consult a professional"
    ]
    for phrase in unwanted_phrases:
        response = response.replace(phrase, '')

    return response.strip()
#st.subheader("📚 Medication Information")
#medicine_name = st.text_input("Enter a medicine name:", placeholder="e.g., Aspirin")



def get_medicine_info(medicine_name):
    url = f"https://api.fda.gov/drug/label.json?search=openfda.brand_name:{medicine_name}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        if "results" in data:
            drug_info = data["results"][0]  # Take the first result

            medicine_details = {
                "Brand Name": drug_info["openfda"].get("brand_name", ["N/A"])[0],
                "Generic Name": drug_info["openfda"].get("generic_name", ["N/A"])[0],
                "Manufacturer": drug_info["openfda"].get("manufacturer_name", ["N/A"])[0],
                "Purpose": drug_info.get("purpose", ["N/A"])[0],
                "Warnings": drug_info.get("warnings", ["N/A"])[0],
                "Dosage": drug_info.get("dosage_and_administration", ["N/A"])[0],
                "Side Effects": drug_info.get("adverse_reactions", ["N/A"])[0],
                "Interactions": drug_info.get("drug_interactions", ["N/A"])[0],
            }
            
            return medicine_details
    return None


#if st.button("Get Medication Info"):
    #if medicine_name:
        #with st.spinner("Fetching medication details..."):
            #med_info = get_medicine_info(medicine_name)

        #if isinstance(med_info, dict):
            #st.success(f"**Medication Information for {medicine_name}:**")
            #for key, value in med_info.items():
                #st.write(f"**{key}:** {value}")
        #else:
            #st.error(med_info)
    #else:
        #st.warning("Please enter a valid medicine name.")
col1, col2 = st.columns([2, 1])
with col1:
    
    
    # Voice Input Button
    st.title("🤖🏥 Welcome to MediBot!")
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
                         answer = clean_response(answer)

                    # Display enhanced response
                         st.success("**Healthcare Assistant Response:**")
                         st.markdown(f"### 🏥 {answer}")  # Styled response for clarity
                         
                    except Exception as e:
                         st.error(f"Error generating response: {e}")
       else:
            st.warning("Please enter a medical query.")


with col2:
    st.subheader("💊 Medicine Information")

    medicine_name = st.text_input("Enter a medicine name:", placeholder="e.g., Paracetamol")

    if st.button("Get Medicine Info"):
        if medicine_name:
            with st.spinner("Fetching medicine details..."):
                info = get_medicine_info(medicine_name)
            
            if info:
                st.success(f"**Medicine Details for {medicine_name}:**")
                for key, value in info.items():
                    st.write(f"**{key}:** {value}")
                if "Summary" in info:
                    st.subheader("📌 Summary")
                    st.write(info["Summary"])
            else:
                st.error("No information found. Try another medicine.")
        else:
            st.warning("Please enter a valid medicine name.")


    
