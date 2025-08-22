import streamlit as st
import pandas as pd
import json
import io
import requests

# (paste your cleaned Streamlit code here, same as I showed before)
# --- Backend Logic (Copied from your original code and adapted) ---

def load_data_from_uploaded_file(uploaded_file):
    """Loads data from a Streamlit uploaded file object into a pandas DataFrame."""
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

def get_data_schema_and_sample(df, num_samples=5):
    """Analyzes the DataFrame to extract its schema and a small sample."""
    buffer = io.StringIO()
    df.info(buf=buffer)
    schema_info = buffer.getvalue()
    data_sample = df.head(num_samples).to_string()
    return schema_info, data_sample

def create_analysis_prompt(schema_info, data_sample):
    """Creates the main prompt for the data readiness assessment."""
    # MODIFIED: Added instruction for a concise summary
    return f"""
    Based on the data schema and sample provided below, perform a comprehensive enterprise data readiness analysis.
    Your analysis should include:
    1. A detailed data profiling for each column.
    2. A governance assessment identifying PII and other quality issues.
    3. A data readiness score (0-100) with a clear justification.
    4. A **concise executive summary** (3–4 sentences maximum).
    5. Actionable recommendations: Provide a list (array) of clear, standalone recommendations, each as a separate string. Avoid numbering within the text.

    --- Data Schema ---
    {schema_info}
    -------------------

    --- Data Sample ---
    {data_sample}
    -----------------

    Provide your analysis in a structured JSON format according to the schema I define.
    """

def call_gemini_api(api_key, prompt, is_json_output=False, json_schema=None):
    """Calls the Gemini API with the given prompt and API key."""
    if not api_key:
        st.error("Gemini API Key is missing. Please enter it in the sidebar.")
        return None

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={api_key}"

    payload = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}

    if is_json_output and json_schema:
        payload["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseSchema": json_schema
        }

    try:
        response = requests.post(
            api_url,
            headers={'Content-Type': 'application/json'},
            json=payload,
            timeout=180 # Set a timeout for the request
        )
        response.raise_for_status()
        result = response.json()

        if not result.get('candidates'):
            raise ValueError("API response is missing 'candidates'. Full response: " + response.text)

        content_part = result['candidates'][0]['content']['parts'][0]
        return json.loads(content_part['text']) if is_json_output else content_part['text']

    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Gemini API: {e}")
        return None
    except (KeyError, IndexError, ValueError) as e:
        st.error(f"Error parsing API response: {e}. Check if the API key is valid.")
        return None

def get_llm_analysis(api_key, schema_info, data_sample):
    """Calls the Gemini API to get the full data readiness analysis."""
    analysis_prompt = create_analysis_prompt(schema_info, data_sample)

    analysis_schema = {
        "type": "OBJECT",
        "properties": {
            "data_profiling": {
                "type": "ARRAY",
                "items": { "type": "OBJECT", "properties": {
                    "column_name": {"type": "STRING"}, "data_type": {"type": "STRING"},
                    "missing_percentage": {"type": "NUMBER"}, "summary": {"type": "STRING"}
                }}
            },
            "governance_assessment": {
                "type": "OBJECT", "properties": {
                    "pii_detection": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "quality_issues": {"type": "STRING"}
                }
            },
            "readiness_score": {
                "type": "OBJECT", "properties": {
                    "score": {"type": "INTEGER"}, "justification": {"type": "STRING"}
                }
            },
            # "summary_and_recommendations": {
            #     "type": "OBJECT", "properties": {
            #         "summary": {"type": "STRING"}, "recommendations": {"type": "STRING"}
            #     }
            "summary_and_recommendations": {
              "type": "OBJECT",
              "properties": {
                  "summary": {"type": "STRING"},
                  "recommendations": {
                      "type": "ARRAY",
                      "items": {"type": "STRING"}
                    }
            }
        }
    }
  }
    return call_gemini_api(api_key, analysis_prompt, is_json_output=True, json_schema=analysis_schema)

def display_report_st(analysis_result):
    """MODIFIED: Prints the final analysis in a new dashboard-style format."""
    if not analysis_result:
        st.warning("Analysis could not be completed.")
        return

    st.header("GenAI Data Readiness Advisor Report")

    # --- Key Metrics Row ---
    score_data = analysis_result.get('readiness_score', {})
    score_value = score_data.get('score', 0)

    gov_data = analysis_result.get('governance_assessment', {})
    pii_cols = gov_data.get('pii_detection', [])

    profiling_data = analysis_result.get('data_profiling', [])
    high_missing_cols = [
        col['column_name'] for col in profiling_data if col.get('missing_percentage', 0) > 30
    ]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Overall Readiness Score", value=f"{score_value}/100")
    with col2:
        st.metric(label="PII Columns Detected", value=len(pii_cols))
    with col3:
        st.metric(label="High Missing Data Columns (>30%)", value=len(high_missing_cols))

    st.info(f"**Justification:** *{score_data.get('justification', 'No justification provided.')}*")
    st.divider()

    # --- Summary and Recommendations ---
    summary_data = analysis_result.get('summary_and_recommendations', {})
    st.subheader("Executive Summary")
    st.markdown(summary_data.get('summary', 'No summary provided.'))

    # st.subheader("Actionable Recommendations")
    # recommendations = summary_data.get('recommendations', 'No recommendations provided.')
    # for rec in recommendations.split('.'):
    #     if rec.strip():
    #         st.markdown(f"- {rec.strip()}.")
    recommendations = summary_data.get('recommendations', [])

    st.subheader("Actionable Recommendations")
    if isinstance(recommendations, list) and recommendations:
       for rec in recommendations:
          st.markdown(f"- {rec}")
    else:
        st.warning("No structured recommendations returned.")
    st.divider()

    # --- Detailed Tabs ---
    tab1, tab2 = st.tabs(["Governance & Quality Assessment", "Detailed Data Profile"])

    with tab1:
        st.subheader("Governance Assessment")
        st.markdown("**Potential PII Columns Detected:**")
        if pii_cols:
            st.warning(', '.join(f"`{col}`" for col in pii_cols))
        else:
            st.success("No obvious PII columns detected.")

        st.subheader("Data Quality Issues")
        st.error(gov_data.get('quality_issues', 'No issues reported.'))

    with tab2:
        st.subheader("Column-by-Column Profile")
        if not profiling_data:
            st.write("No data profiling information was returned.")
        else:
            for col_profile in profiling_data:
                st.markdown(f"#### `{col_profile.get('column_name', 'N/A')}`")

                c1, c2 = st.columns(2)
                c1.markdown(f"**Type:** {col_profile.get('data_type', 'N/A')}")
                # Use a progress bar for missing data
                missing_pct = col_profile.get('missing_percentage', 0)
                c2.progress(int(missing_pct), text=f"Missing Data: {missing_pct}%")

                st.markdown(f"**Summary:** {col_profile.get('summary', 'N/A')}")
                st.markdown("---")


# --- Streamlit UI ---

st.set_page_config(layout="wide", page_title="Data Readiness Advisor")
st.title("GenAI Powered Enterprise Data Readiness Advisor 🤖")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter your Gemini API Key", type="password", help="Get your key from Google AI Studio.")
    st.markdown("---")
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    analyze_button = st.button("Analyze Data Readiness", type="primary", use_container_width=True)

# --- Main Content Area ---
if analyze_button:
    if uploaded_file is not None and api_key:
        with st.spinner('Analyzing your data... This may take a minute.'):
            try:
                # 1. Load Data
                df = load_data_from_uploaded_file(uploaded_file)
                if df is not None:
                    # 2. Get Schema and Sample
                    schema, sample = get_data_schema_and_sample(df)

                    # 3. Run full analysis using Gemini
                    final_analysis = get_llm_analysis(api_key, schema, sample)

                    # 4. Display the report
                    display_report_st(final_analysis)

            except Exception as e:
                st.error(f"An unexpected error occurred during the analysis: {e}")
    elif not api_key:
        st.warning("Please enter your Gemini API Key in the sidebar.")
    else:
        st.warning("Please upload a CSV file.")
else:
    st.info("Upload a CSV file and enter your Gemini API Key to begin the analysis.")
