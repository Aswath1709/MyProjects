import streamlit as st
import pandas as pd
import plotly.express as px
import os
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from llm import response_from_llm



st.set_page_config(page_title='InsightForge - AI BI Assistant', layout='wide')


st.markdown(
    """
    <style>
        body {
            background-color: #f4f4f4;
        }
        .stTabs [data-baseweb="tab-list"] {
            display: flex;
            justify-content: space-evenly;
            gap: 30px;
            border-bottom: 4px solid #3498DB;
            padding-bottom: 20px;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 24px;
            font-weight: bold;
            padding: 20px 40px;
            color: #FFFFFF;
            background-color: #2980B9;
            border-radius: 12px 12px 0 0;
            transition: background-color 0.3s ease, color 0.3s ease;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #1F618D;
            color: white;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1B4F72 !important;
            color: white !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2980B9;
        }
        .stButton>button {
            background-color: #3498DB;
            color: white;
            border-radius: 8px;
        }
        .stMetric {
            background-color: #2980B9;
            padding: 15px;
            border-radius: 12px;
            color: white;
            font-weight: bold;
        }
        .stMetric .stIncrease {
            color: #00FF00 !important; /* Neon Green for positive */
            font-weight: bold;
        }
        .stMetric .stDecrease {
            color: #E74C3C !important; /* Bright Red for negative */
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True
)

def analyze_csv(csv_path):
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
        
        if df.empty:
            raise ValueError("CSV file is empty or could not be loaded correctly.")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Year"] = df["Date"].dt.year

        insights = "### Yearly Business Insights\n"

        
        yearly_sales = df.groupby("Year")["Sales"].sum()
        if len(yearly_sales) > 1:
            insights += "#### Yearly Sales Comparison\n"
            for year, sales in yearly_sales.items():
                insights += f"- {year}: ${sales:,}\n"
            insights += "\n"

            
            sales_growth = yearly_sales.pct_change().dropna() * 100
            insights += "#### Year-over-Year (YoY) Sales Growth\n"
            for year, growth in sales_growth.items():
                insights += f"- {year} vs {year-1}: {growth:.2f}%\n"
            insights += "\n"

        yearly_satisfaction = df.groupby("Year")["Customer_Satisfaction"].mean()
        insights += "#### Customer Satisfaction by Year\n"
        for year, score in yearly_satisfaction.items():
            insights += f"- {year}: {score:.2f}/5\n"
        insights += "\n"

        
        region_sales = df.groupby(["Year", "Region"])["Sales"].sum().reset_index()
        insights += "#### Sales by Region\n"
        for year in region_sales["Year"].unique():
            region_data = region_sales[region_sales["Year"] == year]
            for _, row in region_data.iterrows():
                insights += f"- {year}, {row['Region']}: ${row['Sales']:,}\n"
        insights += "\n"

        
        top_products_by_region = df.groupby(["Year", "Region", "Product"])["Sales"].sum().reset_index()
        top_products_by_region = top_products_by_region.sort_values(by=["Year", "Region", "Sales"], ascending=[True, True, False])
        insights += "#### Top-Selling Products by Region\n"
        for year in top_products_by_region["Year"].unique():
            for region in top_products_by_region["Region"].unique():
                top_product = top_products_by_region[(top_products_by_region["Year"] == year) & (top_products_by_region["Region"] == region)].iloc[0]
                insights += f"- {year}, {region}: {top_product['Product']} (${top_product['Sales']:,})\n"
        insights += "\n"

        return insights

    except Exception as e:
        print(f"Error processing CSV: {e}")
        return ""




def load_and_process_pdfs(pdf_paths):
    all_texts = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        texts = text_splitter.split_documents(pages)
        all_texts.extend(texts)
    
    return all_texts

import chromadb

chromadb.api.client.SharedSystemClient.clear_system_cache()

embeddings = OpenAIEmbeddings()



csv_summary = analyze_csv("sales_data.csv")
pdf_texts = load_and_process_pdfs(["AI business model innovation.pdf", "BI approaches.pdf", "Time-Series-Data-Prediction-using-IoT-and-Machine-Le_2020_Procedia-Computer-.pdf", "Walmarts sales data analysis.pdf"])
csv_text = str(csv_summary)
#print("csv text = ",csv_text)
pdf_texts_extracted = [doc.page_content for doc in pdf_texts]
all_texts = [csv_text] + pdf_texts_extracted
vector_db = Chroma.from_texts(all_texts, embeddings)


retriever = vector_db.as_retriever()


df = pd.read_csv('sales_data.csv',encoding="utf-8",dtype=str)


st.title("📊 InsightForge - AI Business Intelligence")
tabs = st.tabs(["Dashboard", "Visualizations", "Chat Assistant", "Model Evaluation"])


with tabs[0]:
    df["Sales"] = pd.to_numeric(df["Sales"], errors="coerce")

    
    total_revenue = df["Sales"].sum()
    
    df['Sales Growth Rate (%)'] = df['Sales'].pct_change() * 100

    
    sales_growth = df['Sales Growth Rate (%)'].iloc[-1] if len(df) > 1 else 0
    
    customer_retention = df["Customer_Satisfaction"].astype(float).mean() 
    
    
    st.title("📊 Business Intelligence Dashboard")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${total_revenue:,.0f}")
    col2.metric("Sales Growth Rate", f"{sales_growth:.2f}%")
    col3.metric("Customer Retention", f"{(customer_retention/5)*100:.2f}%")
    
    df['Date'] = pd.to_datetime(df['Date'])

    
    df_weekly = df.resample('W', on='Date').sum()  # Weekly
    df_monthly = df.resample('M', on='Date').sum()  # Monthly
    
    
    if "selected_tab" not in st.session_state:
        st.session_state.selected_tab = "Monthly"

   
    selected_tab = st.radio("Select Timeframe:", ["Weekly", "Monthly"], 
                            index=["Weekly", "Monthly"].index(st.session_state.selected_tab))

    
    st.session_state.selected_tab = selected_tab

   
    if selected_tab == "Weekly":
        st.subheader("📅 Weekly Sales Trends")
        fig_weekly = px.line(df_weekly, x=df_weekly.index, y='Sales', 
                            title="Weekly Sales Performance",
                            color_discrete_sequence=["#3498DB"])
        st.plotly_chart(fig_weekly, use_container_width=True)

    elif selected_tab == "Monthly":
        st.subheader("📆 Monthly Sales Trends")
        fig_monthly = px.line(df_monthly, x=df_monthly.index, y='Sales', 
                            title="Monthly Sales Performance",
                            color_discrete_sequence=["#2ECC71"])
        st.plotly_chart(fig_monthly, use_container_width=True)

    


with tabs[1]:
    
    st.title("Data Visualizations")


    
    
    
    viz_option = st.selectbox("Select a visualization:", [
        "Regional Analysis", "Product Performance", "Customer Demographics"])
    
    
    if viz_option == "Product Performance":
        st.subheader("Product Performance Comparison")
        df['Date'] = pd.to_datetime(df['Date'])
        df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
        df.dropna(subset=['Sales'], inplace=True)
        df['Year'] = df['Date'].dt.year
        product_sales = df.groupby(["Year", "Product"])['Sales'].sum().reset_index()
        fig = px.bar(product_sales, x="Year", y="Sales", color="Product", title="Product Performance Over the Years")
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Regional Analysis":
        st.subheader("Regional Sales Analysis")
        df['Date'] = pd.to_datetime(df['Date'])
        df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce') 
        df.dropna(subset=['Sales'], inplace=True)
        df['Year'] = df['Date'].dt.year  

        
        region_year_sales = df.groupby(["Year", "Region"])["Sales"].sum().reset_index()

        min_sales = float(region_year_sales["Sales"].min())
        max_sales = float(region_year_sales["Sales"].max())

        
        y_min = min_sales * 0.9  
        y_max = max_sales * 1.1  

        
        fig = px.bar(region_year_sales, 
                    x="Year", 
                    y="Sales", 
                    color="Region", 
                    title="Yearly Sales by Region",
                    barmode="group",  
                    color_discrete_sequence=["#27AE60", "#E74C3C", "#F1C40F", "#9B59B6"])

        
        fig.update_layout(
            yaxis=dict(
                tickformat=",",  
                title="Total Sales",
                range=[y_min, y_max],  
                zeroline=True,  
                zerolinecolor='gray',  
                zerolinewidth=1  
            ),
            xaxis=dict(
                type="category",
                title="Year"
            )
        )

    
        st.plotly_chart(fig, use_container_width=True)
    
    elif viz_option == "Customer Demographics":
        st.subheader("Customer Demographics & Segmentation")
        
        
        fig_age = px.box(df, x="Customer_Age", title="Customer Age Distribution")
        st.plotly_chart(fig_age, use_container_width=True)
        
        
        if "Gender" in df.columns:
            gender_counts = df["Gender"].value_counts().reset_index()
            gender_counts.columns = ["Gender", "Count"]
            fig_gender = px.bar(gender_counts, x="Gender", y="Count", title="Gender Distribution", color="Gender")
            st.plotly_chart(fig_gender, use_container_width=True)
        
        
        if "Customer_Segment" in df.columns:
            segment_counts = df["Customer_Segment"].value_counts().reset_index()
            segment_counts.columns = ["Segment", "Count"]
            fig_segment = px.bar(segment_counts, x="Segment", y="Count", title="Customer Segmentation", color="Segment")
            st.plotly_chart(fig_segment, use_container_width=True)



if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

memory = st.session_state.memory  

with tabs[2]:
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.title("💬 AI Chat Assistant")
    st.write("🧠 Ask the AI about business insights and trends!")

    
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            role = "user" if isinstance(message, HumanMessage) else "assistant"
            with st.chat_message(role):
                st.markdown(message.content)
    
    
    
    user_input = st.text_input("Type your message and press Enter...", key="user_input", placeholder="Ask me anything...")

    if user_input:
        
        memory.chat_memory.add_user_message(user_input)

        st.session_state.chat_history.append(HumanMessage(content=user_input))
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)

        
        relevant_docs = retriever.invoke(user_input)
        context = "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else "No relevant information found."

        
        chat_history = memory.load_memory_variables({})["chat_history"]

        
        with chat_container:
            with st.chat_message("assistant"):
                streamed_response = st.write_stream(response_from_llm(user_input, chat_history, context))

        
        memory.chat_memory.add_ai_message(streamed_response)
        st.session_state.chat_history.append(AIMessage(content=streamed_response))
        
        

from langchain.evaluation.qa import QAEvalChain
from langchain_openai import ChatOpenAI


df['Date'] = pd.to_datetime(df['Date'], errors='coerce')


df['Year'] = df['Date'].dt.year



qa_pairs = [
    {
        "question": "What were the total sales in the year 2024?",
        "answer": "In 2024, the total sales were{}.".format(df[df["Year"] == 2024]["Sales"].sum())
    },
    {
        "question": "How much sales did the west region do in 2026?",
        "answer": "In 2026, the West region had sales of ${}.".format(df[(df["Region"] == "West") & (df["Year"] == 2026)]["Sales"].sum())
    }
]
from openai import OpenAI
client = OpenAI()

def get_relevant_context(question):
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else "No relevant information found."
    return context

def get_openai_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",  
        input=text,
    )
   
    return response.data[0].embedding

from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import OpenAIEmbeddings

def eval_model(qa_pairs):
    llm = ChatOpenAI(model_name='gpt-4o-mini', temperature=0.7)
    eval_chain = QAEvalChain.from_llm(llm=llm)
    embeddings = OpenAIEmbeddings()

    predictions = []

    
    for qa_pair in qa_pairs:
        question = qa_pair['question']
        expected_answer = qa_pair['answer']
        
       
        context = get_relevant_context(question)

        
        generated_answer = response_from_llm(question, chat_history=[], context=context)
        generated_answer_str = ''.join(generated_answer)
        
        
        predictions.append({
            "question": question,
            "answer": generated_answer_str  
        })

    
    eval_results = eval_chain.evaluate(
        examples=qa_pairs,  
        predictions=predictions,  
        prediction_key="answer"
    )

    
    results = []
    for i, result in enumerate(eval_results):
        predicted_answer = predictions[i]["answer"]
        actual_answer = qa_pairs[i]["answer"]
        
        
        predicted_embedding = get_openai_embedding(predicted_answer)
        actual_embedding = get_openai_embedding(actual_answer)
        
        
        similarity = cosine_similarity([predicted_embedding], [actual_embedding])[0][0]
        
        
        is_correct = similarity > 0.8
        
        results.append({
            "question": qa_pairs[i]["question"],
            "predicted": predicted_answer,  
            "actual": actual_answer,  
            "correct": is_correct  
        })

    return results


with tabs[3]:
    st.title("🤖 Model Evaluation")
    st.write("Evaluating the AI's performance based on sample Q&A pairs.")

    
    eval_results = eval_model(qa_pairs)

    
    for result in eval_results:
        st.write(f"### Question: {result['question']}")
        st.write(f"Predicted Answer: {result['predicted']}")
        st.write(f"Actual Answer: {result['actual']}")
        st.write(f"Correct: {'Yes' if result['correct'] else 'No'}")
        st.write("---")