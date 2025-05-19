import streamlit as st
import pandas as pd
import openai
import os
import json
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv

st.set_page_config(page_title="Investor Dashboard", layout="wide")

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

UPLOAD_DIR = "data/uploads"
CLEANED_DIR = "data/cleaned"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)

st.title("📊 Investor Analysis Dashboard")
uploaded_files = st.file_uploader("Upload investor spreadsheets", type=["xlsx", "csv"], accept_multiple_files=True)

@st.cache_data
def load_data(file):
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file, header=0)

        df.dropna(how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)

        if df.empty or len(df.columns) == 0:
            return None

        return df
    except Exception:
        return None

def clean_and_standardize(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    rename_map = {}
    col_map = {
        'firm': ['firm', 'company', 'organization', 'org', 'synergy_list_of_investors_(20_of_61)'],
        'notes': ['notes', 'note', 'investor_note', 'bd_notes', 'comments'],
        'status': ['status', 'stage', 'interest', 'response']
    }

    for target, aliases in col_map.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = target
                break

    df.rename(columns=rename_map, inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def gpt_tag_status(notes):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
#            model="gpt-4",
            messages=[
                {"role": "system", "content": "Categorize the investor status."},
                {"role": "user", "content": f"Based on the following investor note, categorize their status as one of: [active, interested, cold, passed, unclear].\n\nNote: {notes}\nStatus:"}
            ],
            max_tokens=10,
            temperature=0
        )
        return response.choices[0].message['content'].strip().lower()
    except Exception:
        return "unclear"

def gpt_summarize_notes(notes):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
#            model="gpt-4",
            messages=[
                {"role": "system", "content": "Summarize investor intent and background."},
                {"role": "user", "content": f"Summarize the key intent and background from the following investor note:\n{notes}\nSummary:"}
            ],
            max_tokens=100,
            temperature=0.3
        )
        return response.choices[0].message['content'].strip()
    except Exception:
        return ""




combined_df = None
firm_col = 'firm'
status_col = 'status'
notes_col = 'notes'

if uploaded_files:
    all_dfs = []
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        df = load_data(file)
        if df is None:
            st.warning(f"{file.name} contains no usable data. Please check formatting.")
            continue

        df = clean_and_standardize(df)
        st.info(f"**{file.name}** loaded with columns: {', '.join(df.columns)}")
        all_dfs.append(df)

    if not all_dfs:
        st.stop()

    combined_df = pd.concat(all_dfs, ignore_index=True)

    required_columns = {'firm', 'status', 'notes'}
    missing_columns = required_columns - set(combined_df.columns)

    if missing_columns:
        st.warning(f"Missing expected columns: {', '.join(missing_columns)}")
        firm_col = st.selectbox("Select firm column", combined_df.columns, key='firm')
        status_col = st.selectbox("Select status column", combined_df.columns, key='status')
        notes_col = st.selectbox("Select notes column", combined_df.columns, key='notes')

        if len({firm_col, status_col, notes_col}) < 3:
            st.error("Column selections must be unique. Please choose different columns for firm, status, and notes.")
            st.stop()

        combined_df.rename(columns={firm_col: 'firm', status_col: 'status', notes_col: 'notes'}, inplace=True)
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

    if 'status' not in combined_df.columns and 'notes' in combined_df.columns:
        st.info("No status column found — tagging using GPT...")
        combined_df['status'] = combined_df['notes'].apply(gpt_tag_status)

    summarize_notes = st.sidebar.checkbox("Generate Note Summaries (GPT)", value=False)
    if summarize_notes and 'notes' in combined_df.columns:
        st.info("Generating note summaries via GPT...")
        combined_df['summary'] = combined_df['notes'].apply(gpt_summarize_notes)

    st.sidebar.subheader("Filter Investors")
    if 'status' in combined_df.columns:
        status_filter = st.sidebar.multiselect("Status", options=combined_df['status'].dropna().unique())
        if status_filter:
            combined_df = combined_df[combined_df['status'].isin(status_filter)]

    if 'firm' in combined_df.columns:
        firm_filter = st.sidebar.multiselect("Firm", options=combined_df['firm'].dropna().unique())
        if firm_filter:
            combined_df = combined_df[combined_df['firm'].isin(firm_filter)]

    if 'status' in combined_df.columns:
        st.subheader("Summary by Status")
        status_counts = combined_df['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        st.bar_chart(status_counts.set_index('Status'))

    st.subheader("Investor Table")
    st.dataframe(combined_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cleaned_filename = f"cleaned_investors_{timestamp}.csv"
    cleaned_path = os.path.join(CLEANED_DIR, cleaned_filename)
    combined_df.to_csv(cleaned_path, index=False)

    csv = combined_df.to_csv(index=False).encode('utf-8')
    json_data = combined_df.to_json(orient='records', indent=2).encode('utf-8')

    st.download_button("Download Cleaned CSV", csv, "cleaned_investors.csv", "text/csv")
    st.download_button("Download JSON", json_data, "cleaned_investors.json", "application/json")

if combined_df is not None:
    st.subheader("💬 Ask GPT about your investors")

    if {'firm', 'status', 'notes'}.issubset(combined_df.columns):
        chat_query = st.chat_input("Ask something like: Who are our active leads from Sequoia?")

        if chat_query:
            context_sample = combined_df[['firm', 'status', 'notes']].fillna('').to_dict(orient='records')
            flat_context = json.dumps(context_sample[:25], indent=2)

            messages = [
                {"role": "system", "content": ("You are an analyst assistant. "
                                      "The user will ask questions about investor data. "
                                      "Use the following data context to help answer. " 
                                      + flat_context)},
                {"role": "user", "content": chat_query}
            ]

            try:
                reply = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
#                    model="gpt-4",
                    messages=messages,
                    max_tokens=300,
                    temperature=0.2
                )
                st.write(reply.choices[0].message['content'])

            except Exception as e:
                st.error(f"GPT request failed: {e}")
#            except Exception as e:
#                st.error("GPT request failed. Check your API key and usage limits.")
    else:
        st.warning("Cannot generate GPT chat context. Required columns are missing even after mapping.")
