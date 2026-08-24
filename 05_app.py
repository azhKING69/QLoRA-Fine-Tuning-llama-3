import streamlit as st
import subprocess
import json

# --- CONFIGURATION ---
MODEL_NAME = "osha-1k-model"
# ---------------------

st.set_page_config(page_title="OSHA Extraction AI", layout="wide")

# Inject Custom HTML & CSS (Google Fonts + Sleek Theming)
st.markdown("""
<style>
    /* Import modern Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    /* Apply font to everything in Streamlit */
    html, body, [class*="css"], .stMarkdown, p, h1, h2, h3, h4, h5, h6 {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Main container padding */
    .block-container {
        padding-top: 3rem !important;
        padding-bottom: 3rem !important;
        max-width: 1400px;
    }
    
    /* Main title styling */
    h1 {
        font-weight: 700 !important;
        margin-bottom: 0px !important;
        padding-bottom: 5px !important;
        color: #ffffff !important;
        letter-spacing: -0.5px;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #94a3b8;
        font-size: 1.15rem;
        margin-top: 0px !important;
        margin-bottom: 2rem !important;
        font-weight: 400;
    }

    /* Horizontal Config Row */
    .config-row {
        display: flex;
        gap: 30px;
        background-color: #111827;
        padding: 18px 25px;
        border-radius: 12px;
        border: 1px solid #1f2937;
        margin-bottom: 2.5rem;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        align-items: center;
        flex-wrap: wrap;
    }
    
    .config-item {
        display: flex;
        gap: 12px;
        color: #d1d5db;
        font-size: 0.95rem;
        align-items: center;
        font-weight: 500;
    }
    
    .config-label {
        color: #6b7280;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 1px;
        font-weight: 700;
    }
    
    .config-value {
        color: #3b82f6; /* Modern Blue */
        font-weight: 600;
        background-color: rgba(59, 130, 246, 0.15);
        padding: 4px 10px;
        border-radius: 6px;
    }
    
    /* Style Text Area */
    .stTextArea textarea {
        border-radius: 8px !important;
        border: 1px solid #374151 !important;
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 1px #3b82f6 !important;
    }
    
    /* JSON output box styling */
    .stCodeBlock {
        border-radius: 8px !important;
        border: 1px solid #1f2937 !important;
    }
</style>
""", unsafe_allow_html=True)

# Main Content
st.title("OSHA Construction Incident AI")
st.markdown('<p class="subtitle">Automated structured data extraction from unstructured workplace incident reports.</p>', unsafe_allow_html=True)

# Horizontal Config Row 
st.markdown(f"""
<div class="config-row">
    <div class="config-item"><span class="config-label">Model</span> <span class="config-value">{MODEL_NAME}</span></div>
    <div class="config-item"><span class="config-label">Engine</span> <span class="config-value">Ollama Local</span></div>
    <div class="config-item"><span class="config-label">Base</span> <span class="config-value">Llama 3.1 8B</span></div>
    <div class="config-item"><span class="config-label">Quantization</span> <span class="config-value">4-bit GGUF</span></div>
</div>
""", unsafe_allow_html=True)

system_prompt = """You are a construction safety analyst. Given an OSHA incident narrative, extract the following structured fields.
Respond with ONLY a valid JSON object. Do not include markdown formatting or explanation.

Fields:
- event_type
- injury_nature
- body_part
- source_equipment
- hospitalized: true or false
- amputation: true or false"""

# Add a gap column between the two main columns for breathing room
col1, space, col2 = st.columns([1, 0.05, 1])

with col1:
    st.markdown("#### Input Narrative")
    user_input = st.text_area(
        "Paste the official OSHA incident report below:", 
        height=280, 
        label_visibility="collapsed",
        placeholder="e.g., Worker fell 15 feet from scaffolding, breaking his right leg..."
    )
    
    st.write("") # small visual spacing before the button
    extract_btn = st.button("Extract JSON Data", type="primary", use_container_width=True)

with col2:
    st.markdown("#### Structured Extraction")
    if extract_btn:
        if user_input:
            with st.spinner("Processing locally via Ollama..."):
                process = subprocess.Popen(
                    ['ollama', 'run', MODEL_NAME],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                full_prompt = f"{system_prompt}\n\nNarrative: {user_input}"
                stdout, stderr = process.communicate(input=full_prompt)
                
                try:
                    parsed_json = json.loads(stdout.strip())
                    st.success("Extraction Complete")
                    st.json(parsed_json)
                except Exception:
                    st.error("Model returned invalid JSON format")
                    st.code(stdout, language="json")
        else:
            st.warning("Please enter a narrative first.")
