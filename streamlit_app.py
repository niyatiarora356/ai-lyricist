import streamlit as st
import requests
import json

# Page config
st.set_page_config(
    page_title="AI Lyricist",
    page_icon="✍️",
    layout="centered"
)

# Custom CSS for Shakespearean vibe
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5dc;
        color: #2c1e1e;
    }
    .main {
        background: rgba(255, 255, 255, 0.8);
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        font-family: 'Garamond', serif;
        color: #4a3728;
    }
    .sonnet-container {
        font-family: 'Garamond', serif;
        font-size: 1.2rem;
        white-space: pre-wrap;
        line-height: 1.6;
        padding: 20px;
        border-left: 5px solid #4a3728;
        background-color: #fffaf0;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

st.title("✍️ AI Lyricist")
st.subheader("Your Personal Shakespearean Bard")

st.markdown("""
Welcome, traveler! Enter a theme, and let the digital spirits weave a sonnet in the immortal style of the Bard of Avon.
""")

theme = st.text_input("Enter a Theme (e.g., Love, Time, Jealousy, Nature):", "Love")

if st.button("Weave a Sonnet"):
    if theme:
        with st.spinner("The Bard is thinking..."):
            try:
                # Call FastAPI backend
                response = requests.post(
                    "http://localhost:8000/generate",
                    json={"theme": theme}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    sonnet = result["sonnet"]
                    
                    st.markdown("### Thy Sonnet:")
                    st.markdown(f'<div class="sonnet-container">{sonnet}</div>', unsafe_allow_html=True)
                    
                    st.download_button(
                        label="Download this Sonnet",
                        data=sonnet,
                        file_name=f"sonnet_{theme.lower()}.txt",
                        mime="text/plain"
                    )
                else:
                    st.error(f"Error from backend: {response.text}")
            except Exception as e:
                st.error(f"Could not connect to the backend server. Make sure the FastAPI app is running! (Error: {e})")
    else:
        st.warning("Pray, enter a theme first!")

st.sidebar.markdown("---")
st.sidebar.info("""
**About AI Lyricist**
This application uses a fine-tuned TinyLlama model to generate Shakespearean sonnets. 
The model was trained on the complete works of William Shakespeare to internalize his unique vocabulary, rhythm, and rhyme schemes.
""")

st.sidebar.markdown("© 2026 The Digital Bard")
