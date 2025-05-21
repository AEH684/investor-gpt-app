import streamlit as st
import pandas as pd
import os
import json
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import openpyxl
import yfinance as yf



st.set_page_config(page_title="Investor Dashboard", layout="wide")

load_dotenv()
client = OpenAI()

UPLOAD_DIR = "data/uploads"
CLEANED_DIR = "data/cleaned"
ASSISTANT_EXPORT_XLSX = r"C:\\assistant_data\\paul_data_latest.xlsx"
ASSISTANT_EXPORT_JSON = r"C:\\assistant_data\\paul_data_latest.json"
COMMON_PROFILE_PATH   = os.path.join("data", "profiles.json")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(r"C:\\assistant_data", exist_ok=True)

st.title("📊 Investor Analysis Dashboard")

# 👤 User selector: Paul vs Gene
user_role = st.sidebar.radio("Who's using the dashboard?", ["Paul (Capital Strategist)", "Gene (AL – Dev)"])

# utils.py
import pandas as pd
import streamlit as st

COLUMN_ALIASES = {
    "organization": "firm",
    "company": "firm",
    "firm_name": "firm",
    "comments": "notes",
    "remarks": "notes",
    "status_1": "status",
    "statuses": "status"
}

@st.cache_data
def load_data(file, header_row):
    print(f"Loading file: {file.name}")
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            xlsx = pd.ExcelFile(file)
            print("Available sheets:", xlsx.sheet_names)

            # Try each sheet until one has data
            for sheet in xlsx.sheet_names:
                raw = pd.read_excel(xlsx, sheet_name=sheet, header=None, nrows=30)
                if not raw.dropna(how="all").empty:
                    print(f"Using sheet: {sheet}")
                    break
            else:
                st.warning(f"No usable sheets found in {file.name}")
                return None

            # Load actual data using provided header row
            df = pd.read_excel(xlsx, sheet_name=sheet, header=header_row)

        # Normalize and alias
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns}, inplace=True)
        df.dropna(how='all', inplace=True)
        df.dropna(axis=1, how='all', inplace=True)

        if df.empty or len(df.columns) == 0:
            st.warning(f"{file.name} contains no usable data. Please check formatting.")
            return None

        print(f"Loaded and normalized {file.name} with shape: {df.shape}")
        return df

    except Exception as e:
        st.error(f"Error loading {file.name}: {e}")
        return None

# --- MAIN APP LOGIC ---
uploaded_files = st.file_uploader("To Upload New Participants:", type=["xlsx", "csv"], accept_multiple_files=True)


def fetch_yahoo_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "symbol": info.get("symbol"),
            "shortName": info.get("shortName"),
            "currentPrice": info.get("currentPrice"),
            "marketCap": info.get("marketCap"),
            "sector": info.get("sector"),
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
        }
    except Exception as e:
        return {"error": str(e)}


def clean_and_standardize(df):
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    rename_map = {}
    col_map = {
        'firm': ['firm', 'company', 'organization', 'organizations', 'org', 'synergy_list_of_investors_(20_of_61)'],
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
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Categorize the investor status."},
                {"role": "user", "content": f"Based on the following investor note, categorize their status as one of: [active, interested, cold, passed, unclear].\n\nNote: {notes}\nStatus:"}
            ],
            max_tokens=10,
            temperature=0
        )
        return response.choices[0].message.content.strip().lower()
    except Exception:
        return "unclear"

def gpt_summarize_notes(notes):
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "Summarize investor intent and background."},
                {"role": "user", "content": f"Summarize the key intent and background from the following investor note:\n{notes}\nSummary:"}
            ],
            max_tokens=100,
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return ""

if "combined_df" not in st.session_state:
    st.session_state["combined_df"] = None

combined_df = st.session_state["combined_df"]
firm_col = 'firm'
status_col = 'status'
notes_col = 'notes'

# 🛠️ Advanced Options in Sidebar
with st.sidebar.expander("🛠️ Advanced Upload Options"):
    header_row = st.number_input("Header row (0-indexed)", min_value=0, max_value=100, value=2)

# --- File Upload and Processing ---
if uploaded_files:
    all_dfs = []
    for file in uploaded_files:
        file_path = os.path.join(UPLOAD_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())

        df = load_data(file, header_row)
        if df is None:
            st.warning(f"{file.name} contains no usable data. Please check formatting.")
            continue

        df = clean_and_standardize(df)

        if 'status' not in df.columns and 'notes' in df.columns:
            with st.spinner(f"Tagging statuses for {file.name} using GPT..."):
                df['status'] = df['notes'].apply(gpt_tag_status)

        st.info(f"**{file.name}** loaded with columns: {', '.join(df.columns)}")
        all_dfs.append(df)

    if not all_dfs:
        st.stop()

    combined_df = pd.concat(all_dfs, ignore_index=True)
    st.session_state["combined_df"] = combined_df

    # Load & merge profiles
    if os.path.exists(COMMON_PROFILE_PATH):
        with open(COMMON_PROFILE_PATH, "r") as f:
            existing_profiles = json.load(f)
        profiles_df = pd.DataFrame(existing_profiles)
    else:
        profiles_df = pd.DataFrame(columns=combined_df.columns)

    combined_profiles = pd.concat([profiles_df, combined_df], ignore_index=True)
    combined_profiles.drop_duplicates(subset=['firm', 'notes'], inplace=True)

    combined_profiles.to_json(COMMON_PROFILE_PATH, orient="records", indent=2)



    required_columns = {'firm', 'status', 'notes'}
    missing_columns = required_columns - set(combined_df.columns)

    if missing_columns:
        st.warning(f"Missing expected columns: {', '.join(missing_columns)}")
        available_columns = combined_df.columns.tolist()
        firm_col = st.selectbox("Select firm column", available_columns, index=available_columns.index('firm') if 'firm' in available_columns else 0, key='firm')
        status_col = st.selectbox("Select status column", available_columns, index=available_columns.index('status') if 'status' in available_columns else 1, key='status')
        notes_col = st.selectbox("Select notes column", available_columns, index=available_columns.index('notes') if 'notes' in available_columns else 2, key='notes')

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cleaned_filename = f"cleaned_investors_{timestamp}.csv"
    cleaned_path = os.path.join(CLEANED_DIR, cleaned_filename)
    combined_df.to_csv(cleaned_path, index=False)

    try:
        combined_df.to_excel(ASSISTANT_EXPORT_XLSX, index=False, engine='openpyxl')
    except Exception as e:
        st.warning(f"Failed to write Excel to assistant path: {e}")

    try:
        combined_df.to_json(ASSISTANT_EXPORT_JSON, orient='records', indent=2)
    except Exception as e:
        st.warning(f"Failed to write JSON to assistant path: {e}")

    csv = combined_df.to_csv(index=False).encode('utf-8')
    json_data = combined_df.to_json(orient='records', indent=2).encode('utf-8')

    st.download_button("Download Cleaned CSV", csv, "cleaned_investors.csv", "text/csv")
    st.download_button("Download JSON", json_data, "cleaned_investors.json", "application/json")

#st.write("Debug: uploaded_files =", uploaded_files)

if not uploaded_files:
    st.subheader("📂 Investment Profiles")

    if st.session_state["combined_df"] is None:
        if st.button("Load Profiles"):
            if os.path.exists(COMMON_PROFILE_PATH):
                with open(COMMON_PROFILE_PATH, "r") as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

                if 'status' not in df.columns and 'notes' in df.columns:
                    st.info("No status column found — tagging using GPT...")
                    with st.spinner("Tagging status using GPT This will take a minute..."):
                        df['status'] = df['notes'].apply(gpt_tag_status)

                st.session_state["combined_df"] = df
                st.success("Loaded profiles from common store.")
            else:
                st.warning("No profiles file found yet.")

combined_df = st.session_state["combined_df"]

if combined_df is None:
    st.warning("no data to analyze.")
else:
    if 'status' in combined_df.columns:
        st.subheader("Summary by Status")
        status_counts = combined_df['status'].value_counts().reset_index()
        status_counts.columns = ['Status', 'Count']
        st.bar_chart(status_counts.set_index('Status'))

    # Show investor table
    st.subheader("Investor Table")
    st.dataframe(combined_df)

    # Show debug and GPT block
    st.write("Note: active columns =", combined_df.columns.tolist())
    st.subheader("💬 Consult GPT about your profiles")
    if "chat_query" not in st.session_state:
        st.session_state["chat_query"] = None
    new_chat = st.chat_input("Ask something like: Who are our active leads from Sequoia?")
    if new_chat:
        st.session_state["chat_query"] = new_chat
    chat_query = st.session_state["chat_query"]

    with st.sidebar:
        if st.button("🔄 Clear Chat Session"):
            st.session_state["chat_query"] = None
            st.session_state["ticker_input"] = None
            st.session_state["yahoo_data"] = {}
            st.rerun()

    if {'firm', 'status', 'notes'}.issubset(combined_df.columns):
        ticker_input = None
        yahoo_data = {}
        needs_web_lookup = False

        # Detect if Paul wants live data

        if chat_query and user_role == "Paul (Capital Strategist)":
            needs_web_lookup = any(
                kw in chat_query.lower() for kw in [
                    "market", "valuation", "real-time", "price", "fund", "investment", "capital",
                    "raise", "aum", "multiple", "trend", "public", "stock", "equity",
                    "pe", "p/e", "eps", "earnings", "revenue", "profit"
                ]
            )

        if "ticker_input" not in st.session_state:
            st.session_state["ticker_input"] = None
        if "yahoo_data" not in st.session_state:
            st.session_state["yahoo_data"] = {}

        if needs_web_lookup:
            st.info("🔍 Real-time lookup triggered for Paul — integrating Yahoo Finance data...")
            new_ticker = st.text_input("Enter a stock ticker (e.g., MSFT, TSLA, AAPL):")
            if new_ticker and new_ticker != st.session_state["ticker_input"]:
                st.session_state["ticker_input"] = new_ticker
                st.session_state["yahoo_data"] = fetch_yahoo_data(new_ticker.upper())

            ticker_input = st.session_state["ticker_input"]
            yahoo_data = st.session_state["yahoo_data"]

            if ticker_input:
                st.subheader("📈 Yahoo Finance Snapshot")
                st.json(yahoo_data)

            # ✅ Move this into the GPT block — don't put st.stop() or fetch logic twice

        # Run GPT only when ready
        if chat_query is not None and chat_query.strip() != "":
            context_sample = combined_df[['firm', 'status', 'notes']].fillna('').to_dict(orient='records')
#            flat_context = json.dumps(context_sample[:25], indent=2)
            flat_context = "\n".join(
                f"- {entry['firm']}: {entry.get('status', 'unknown')} (Notes: {entry.get('notes', '').strip() or 'none'})"
                for entry in context_sample[:25]
            )
            if needs_web_lookup and ticker_input and yahoo_data:
                summary = ", ".join(f"{k}: {v}" for k, v in yahoo_data.items() if k != "error")
                flat_context += f"\n\n📈 Real-time data for {ticker_input.upper()}: {summary}"
                st.subheader("🧾 Final context sent to GPT")
                st.text(flat_context)

            # System prompt
            if user_role == "Paul (Capital Strategist)":

                system_prompt = (
                    "You are Capital Strategist, a GPT built for Paul Harrison. Paul is a CEO with a proprietary formula-based framework to evaluate investment structures, business opportunities, and strategic alignment."
                    "You operate in real-time, integrating internet-sourced data when needed, and use structured investor data to produce actionable insight."
                    "You have access to recently retrieved real-time market data, including stock performance, valuation metrics, and sector benchmarks."
                    "When Paul asks for 'the latest', 'current value', or anything similar about a company, assume it refers to the most recent data already provided. Do not disclaim lack of internet access. Instead, confidently analyze the Yahoo Finance data and present it as live insight, unless explicitly told otherwise."
                    "Stay concise, strategic, and finance-driven."
            )
            else:
                system_prompt = (
                    "You are AL, a development assistant for Gene. "
                    "Help with debugging, Python, pandas, Streamlit integration, and anything related to code maintenance or data logic."
                )

            messages = [
                {"role": "system", "content": system_prompt + "\n\nContext: " + flat_context},
                {"role": "user", "content": chat_query}
            ]
            # ✅ Debug just before GPT call
            st.subheader("🚀 READY TO RUN GPT")
            st.text("Message Summary Sent to GPT:")
            st.text(f"System prompt length: {len(system_prompt)}")
            st.text(f"Context length: {len(flat_context)}")
            st.text(f"User prompt: {chat_query}")

            try:
                reply = client.chat.completions.create(
                    model="gpt-4-turbo",
                    messages=messages,
                    max_tokens=300,
                    temperature=0.2
                )
                st.write(reply.choices[0].message.content)

            except Exception as e:
                import traceback
                st.error("❌ GPT request failed!")
                st.text(traceback.format_exc())
    else:
        st.warning("Cannot generate GPT chat context. Required columns are missing even after mapping.")

