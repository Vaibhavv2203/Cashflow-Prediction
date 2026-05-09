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
if "future_df" not in st.session_state:
    st.session_state.future_df = None
if "chatbot_df" not in st.session_state:
    st.session_state.chatbot_df = pd.DataFrame(columns=["Question", "Response", "Chart Type", "X-axis", "Y-axis"])
if "chatbot_generated_df" not in st.session_state:
    st.session_state.chatbot_generated_df = None

# --- FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload your past 1-year cashflow Excel or CSV file", type=["xlsx", "csv"])

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("📁 Uploaded Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

    # --- CLEAN DATA ---
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    df = df.sort_values('date')

    # --- SIDEBAR SETTINGS ---
    st.sidebar.header("Prediction Settings")
    months_to_predict = st.sidebar.slider("Months to predict", min_value=1, max_value=6, value=2)

    # --- PREPARE PROMPT FOR GEMINI ---
    sample_data = df.tail(60).copy()
    sample_data['date'] = sample_data['date'].dt.strftime('%Y-%m-%d')
    data_dict = sample_data.to_dict(orient='records')

    prediction_prompt = f"""
You are a financial assistant. Given the following company's daily cashflow transactions for the past two months, predict a realistic cashflow sequence for the next {months_to_predict} months.

Each row includes:
- date
- amount (positive for credit, negative for debit)
- event (e.g. Loan Sanction, Salary Credit, Client Payment)
- description
- party (bank, client, vendor)

Maintain a consistent style with realistic patterns like monthly rent, recurring salaries, random client payments, and possible loan sanctions. Return the data in JSON format as a list of dicts with the same schema.

Past Transactions:
{json.dumps(data_dict, indent=2)}
"""

    if st.button("🔮 Predict Future Cashflows"):
        with st.spinner("Generating predictions using Gemini..."):
            try:
                model = genai.GenerativeModel("models/gemini-2.0-flash")
                response = model.generate_content(prediction_prompt)
                raw_response = response.text.strip()

                match = re.search(r"\[.*\]", raw_response, re.DOTALL)
                if not match:
                    raise ValueError("No valid JSON list found in response.")

                json_data = json.loads(match.group(0))
                future_df = pd.DataFrame(json_data)
                future_df['date'] = pd.to_datetime(future_df['date'], errors='coerce')

                st.session_state.future_df = future_df

            except Exception as e:
                st.error(f"Something went wrong: {e}")

    # --- Always show predicted data if available ---
    if st.session_state.future_df is not None:
        st.subheader("📈 Predicted Cashflows")
        st.dataframe(st.session_state.future_df, use_container_width=True)

        csv = st.session_state.future_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Predictions as CSV", csv, "predicted_cashflows.csv", "text/csv")

    # --- CHATBOT SECTION ---
    st.markdown("### 🤖 Ask a Question About Your Cashflows")
    user_query = st.text_input("Type your question here", key="cashflow_q")

    if user_query:
        with st.spinner("Analyzing your actual and predicted data..."):
            try:
                # Combine actual and future data
                combined_df = df.copy()
                if st.session_state.future_df is not None:
                    future_df = st.session_state.future_df.copy()
                    future_df['date'] = pd.to_datetime(future_df['date'], errors='coerce')
                    combined_df = pd.concat([combined_df, future_df], ignore_index=True)

                combined_df = combined_df.sort_values('date')
                combined_data = combined_df.to_dict(orient='records')

                for record in combined_data:
                    if isinstance(record.get('date'), pd.Timestamp):
                        record['date'] = record['date'].strftime('%Y-%m-%d')

                context_json = json.dumps(combined_data[-150:], indent=2)

                chat_prompt = f"""
You are a financial assistant. Answer the following question based only on the provided cashflow data, which includes both past transactions and future predictions.

1. If the question needs tabular data (e.g. breakdown of expenses per company), return it as a JSON list of dictionaries.
2. Recommend the best type of chart to visualize the data if the first condition is met, along with the x-axis and y-axis labels. I want the data in the following format:
   eg; Recommended chart type: bar (Single word)
       X-axis: Party
       Y-axis: Number of expenses
    If the first condition is not met, then return a single line answer without the tabular format.

Cashflow Data:
{context_json}

User Question:
{user_query}
"""

                model = genai.GenerativeModel("models/gemini-2.0-flash")
                response = model.generate_content(chat_prompt)
                raw_response = response.text.strip()
                st.markdown("**💬 Gemini's Answer:**")

                # Try extracting a dataframe
                match_df = re.search(r"\[.*\]", raw_response, re.DOTALL)
                chatbot_generated_df = None
                if match_df:
                    try:
                        extracted_table = json.loads(match_df.group(0))
                        chatbot_generated_df = pd.DataFrame(extracted_table)
                        st.session_state.chatbot_generated_df = chatbot_generated_df

                        st.subheader("📊 Generated Dataframe from Chatbot")
                        st.dataframe(chatbot_generated_df, use_container_width=True)
                    except Exception:
                        st.warning("Gemini returned invalid JSON table.")
                else:
                    st.write(raw_response)

                # Extract chart suggestion
                chart_type, x_axis, y_axis = None, None, None
                match = re.search(r"Recommended chart type:\s*(\w+)\s+X-axis:\s*(.*?)\s+Y-axis:\s*(.*)", raw_response, re.IGNORECASE)

                if match:
                    chart_type = match.group(1).lower()
                    x_axis = match.group(2).strip()
                    y_axis = match.group(3).strip()

                    st.success(f"📌 Suggested Chart: {chart_type}")
                    if x_axis and y_axis:
                        st.info(f"X-axis: {x_axis}, Y-axis: {y_axis}")

                # Save to chatbot history
                new_entry = pd.DataFrame([{
                    "Question": user_query,
                    "Response": raw_response,
                    "Chart Type": chart_type or "",
                    "X-axis": x_axis,
                    "Y-axis": y_axis
                }])
                st.session_state.chatbot_df = pd.concat([st.session_state.chatbot_df, new_entry], ignore_index=True)

                # Plotting
                if chart_type and x_axis and y_axis:
                    plot_df = st.session_state.chatbot_generated_df
                    if plot_df is not None and x_axis in plot_df.columns and y_axis in plot_df.columns:
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
                            st.write("🔁 Try a different chart type or verify column names.")

            except Exception as e:
                st.error(f"Chatbot failed: {e}")

        # --- Display chatbot history ---
        st.markdown("### 📜 Chat History")
        st.dataframe(st.session_state.chatbot_df, use_container_width=True)

else:
    st.info("👆 Upload a valid Excel or CSV file to get started.")
