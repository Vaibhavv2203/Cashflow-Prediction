# Interactive AI-Based Cashflow Predictor with Sidebar Enhancements
import streamlit as st
import pandas as pd
import json
import re
from datetime import datetime
import google.generativeai as genai
import plotly.express as px

# --- SETUP ---
st.set_page_config(page_title="Cashflow Predictor AI", layout="wide")
st.title("📊 AI-Based Cashflow Predictor")

# --- GEMINI API KEY ---
genai.configure(api_key="YOUR_GEMINI_KEY")  # Replace with your actual Gemini API key

# --- SESSION STATE ---
for key in ["future_df", "chatbot_df", "chatbot_generated_df", "chat_history"]:
    if key not in st.session_state:
        if key == "chatbot_df":
            st.session_state[key] = pd.DataFrame(columns=["User", "Bot"])
        elif key == "chat_history":
            st.session_state[key] = []
        else:
            st.session_state[key] = None

# --- SIDEBAR: FILE UPLOAD ---
st.sidebar.header("Upload & Preview")
uploaded_file = st.sidebar.file_uploader("Upload past 1-year cashflow Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date')

    st.sidebar.subheader("📁 Uploaded Data Preview")
    st.sidebar.dataframe(df.head(5), use_container_width=True)

    # --- SIDEBAR: PREDICTION SETTINGS ---
    st.sidebar.header("Prediction Settings")
    months_to_predict = st.sidebar.slider("Months to predict", 1, 6, 2)

    # --- PREDICT CASHFLOW ---
    sample_data = df.tail(60).copy()
    sample_data['date'] = sample_data['date'].dt.strftime('%Y-%m-%d')
    data_dict = sample_data.to_dict(orient='records')

    prediction_prompt = f"""
    You are a financial assistant. Given the following company's daily cashflow transactions for the past two months, predict a realistic cashflow sequence for the next {months_to_predict} months.
    Each row includes: date, amount (positive=credit, negative=debit), event, description, party.
    Maintain realistic patterns (rent, salary, client payments, loans). Return data as a JSON list with the same schema.
    Past Transactions:
    {json.dumps(data_dict, indent=2)}
    """

    if st.sidebar.button("🔮 Predict Future Cashflows"):
        with st.spinner("Generating predictions using Gemini..."):
            try:
                model = genai.GenerativeModel("models/gemini-2.0-flash")
                response = model.generate_content(prediction_prompt)
                raw_response = response.text.strip()
                match = re.search(r"\[.*\]", raw_response, re.DOTALL)
                json_data = json.loads(match.group(0))
                future_df = pd.DataFrame(json_data)
                future_df['date'] = pd.to_datetime(future_df['date'], errors='coerce')
                st.session_state.future_df = future_df
            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # Show predicted data if available
    if st.session_state.future_df is not None:
        st.sidebar.subheader("📈 Predicted Cashflows")
        st.sidebar.dataframe(st.session_state.future_df, use_container_width=True)

# --- INTERACTIVE CHATBOT ---
st.markdown("### 🤖 Interactive Cashflow Assistant")

chat_input_container = st.container()
chat_response_container = st.container()

with chat_input_container:
    user_input = st.text_input("Ask a question or wait for a suggestion...", key="top_input")

if uploaded_file:
    combined_df = df.copy()
    if st.session_state.future_df is not None:
        combined_df = pd.concat([combined_df, st.session_state.future_df], ignore_index=True)
    combined_df = combined_df.sort_values('date')
    combined_data = combined_df.tail(150).to_dict(orient='records')
    for rec in combined_data:
        if isinstance(rec.get('date'), pd.Timestamp):
            rec['date'] = rec['date'].strftime('%Y-%m-%d')

    chat_context = json.dumps(combined_data, indent=2)

    if user_input:
        st.session_state.chat_history.append({"role": "user", "text": user_input})
        chat_prompt = f"""
        You are a financial assistant. The user is asking questions based on the following cashflow data:

        Data:
        {chat_context}

        Conversation so far:
        {st.session_state.chat_history}

        Reply in a helpful tone. If possible, ask a relevant follow-up question to keep the conversation going. If the answer can be in the form of table, then store it in the form of json.
        If it is in the form of json, then suggest a possible chart type and x-axis and y-axis in the form of:
        eg: Recommended chart type: bar
            X-axis: Party
            Y-axis: Number of expenses
        (Mandatory to suggest if it is in the json format)
            
        If the answer is not in json format of which dataframe can not be made then only answer the question asked in a simple sentence format.

        User: {user_input}
        """
        with st.spinner("Thinking..."):
            try:
                model = genai.GenerativeModel("models/gemini-2.0-flash")
                response = model.generate_content(chat_prompt)
                bot_response = response.text.strip()
                st.session_state.chat_history.append({"role": "bot", "text": bot_response})

                # Try parsing JSON from bot
                match_df = re.search(r"\[.*\]", bot_response, re.DOTALL)
                if match_df:
                    try:
                        extracted_data = json.loads(match_df.group(0))
                        chatbot_generated_df = pd.DataFrame(extracted_data)
                        st.session_state.chatbot_generated_df = chatbot_generated_df
                    except:
                        st.warning("Gemini returned invalid JSON table.")

            except Exception as e:
                bot_response = f"Error: {e}"
                st.session_state.chat_history.append({"role": "bot", "text": bot_response})

    with chat_response_container:
        for msg in st.session_state.chat_history:
            role = "user" if msg["role"] == "user" else "assistant"
            st.chat_message(role).markdown(msg["text"])

        if st.session_state.chatbot_generated_df is not None:
            st.subheader("📊 Data Extracted from Response")
            st.dataframe(st.session_state.chatbot_generated_df, use_container_width=True)

            # Extract and suggest chart
            bot_response = st.session_state.chat_history[-1]["text"]
            match = re.search(r"Recommended chart type:\s*(\w+)\s+X-axis:\s*(.*?)\s+Y-axis:\s*(.*)", bot_response, re.IGNORECASE)
            if match:
                chart_type, x_axis, y_axis = match.groups()
                chart_type = chart_type.lower()
                #st.write(f"{chart_type},{x_axis},{y_axis}")
                st.success(f"📌 Suggested Chart: {chart_type}")
                st.info(f"X-axis: {x_axis}, Y-axis: {y_axis}")

                plot_df = st.session_state.chatbot_generated_df
                if x_axis in plot_df.columns and y_axis in plot_df.columns:
                    try:
                        fig = None
                        if chart_type == "line":
                            fig = px.line(plot_df, x=x_axis, y=y_axis, markers=True)
                        elif chart_type == "bar":
                            fig = px.bar(plot_df, x=x_axis, y=y_axis)
                        elif chart_type == "area":
                            fig = px.area(plot_df, x=x_axis, y=y_axis)
                        elif chart_type == "pie":
                            fig = px.pie(plot_df, names=x_axis, values=y_axis)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Plotting failed: {e}")
else:
    st.info("👈 Upload a valid cashflow file to start using the assistant.")
