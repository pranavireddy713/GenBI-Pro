import streamlit as st
import pandas as pd
import plotly.express as px
from google import genai
from sklearn.ensemble import RandomForestRegressor

client = genai.Client(
    api_key="AQ.Ab8RN6IqrWjvysQzfnNfZmPjmysV52mzKff0UOP97nLONp5xOA"
)

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="GenBI Dashboard",
    layout="wide"
)

# ---------------- TITLE ----------------

st.title("📊 GenBI - Generative AI Business Intelligence Assistant")

st.write("Upload any CSV dataset and generate analytics automatically.")

# ---------------- FILE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Load dataset
    df = pd.read_csv(uploaded_file)

    st.success("✅ Dataset Loaded Successfully!")

    # ---------------- DATASET PREVIEW ----------------

    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head())

    # ---------------- DATASET INFORMATION ----------------

    st.subheader("📌 Dataset Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    # ---------------- STATISTICS ----------------

    st.subheader("📈 Statistics")

    st.dataframe(df.describe(include="all"))

    # ---------------- MISSING VALUES ----------------

    st.subheader("❌ Missing Values")

    missing_values = df.isnull().sum()

    st.dataframe(missing_values)

    # ---------------- GENERIC KPIs ----------------

    st.subheader("📊 Dataset KPIs")

    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) > 0:

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Rows",
                df.shape[0]
            )

        with col2:
            st.metric(
                "Columns",
                df.shape[1]
            )

        with col3:
            st.metric(
                f"Average {numeric_cols[0]}",
                round(df[numeric_cols[0]].mean(), 2)
            )

        with col4:
            st.metric(
                f"Maximum {numeric_cols[0]}",
                round(df[numeric_cols[0]].max(), 2)
            )

    # ---------------- SCATTER PLOT ----------------

    if len(numeric_cols) >= 2:

        st.subheader("📈 Automatic Data Visualization")

        fig1 = px.scatter(
            df,
            x=numeric_cols[0],
            y=numeric_cols[1],
            title=f"{numeric_cols[0]} vs {numeric_cols[1]}"
        )

        st.plotly_chart(fig1, use_container_width=True)

    # ---------------- HISTOGRAM ----------------

    if len(numeric_cols) >= 1:

        st.subheader("📊 Distribution Analysis")

        fig2 = px.histogram(
            df,
            x=numeric_cols[0],
            title=f"Distribution of {numeric_cols[0]}"
        )

        st.plotly_chart(fig2, use_container_width=True)

    # ---------------- CORRELATION HEATMAP ----------------

    if len(numeric_cols) >= 2:

        st.subheader("🔥 Correlation Heatmap")

        corr = df[numeric_cols].corr()

        fig3 = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            title="Correlation Matrix"
        )

        st.plotly_chart(fig3, use_container_width=True)

    # ---------------- BOX PLOT ----------------

    if len(numeric_cols) >= 1:

        st.subheader("📦 Outlier Detection")

        fig4 = px.box(
            df,
            y=numeric_cols[0],
            title=f"Box Plot of {numeric_cols[0]}"
        )

        st.plotly_chart(fig4, use_container_width=True)

    # ---------------- DATA TYPES ----------------

    st.subheader("🧾 Column Data Types")

    dtype_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str)
    })

    st.dataframe(dtype_df)
    if uploaded_file is not None:
    
# ==========================
# MACHINE LEARNING PREDICTION
# ==========================

        st.markdown("---")
        st.header("🤖 Machine Learning Prediction")

        numeric_cols = df.select_dtypes(include="number").columns

        if len(numeric_cols) > 1:

            target = st.selectbox(
                "Select Target Column",
                numeric_cols
            )

            if st.button("Train ML Model"):

                X = df[numeric_cols].drop(columns=[target])
                y = df[target]

                model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                )

                model.fit(X, y)

                score = model.score(X, y)

                st.success(
                    f"Model Accuracy: {round(score*100,2)}%"
                )


        # ==========================
        # AI DATA ASSISTANT
        # ==========================

        st.markdown("---")

        st.header("🤖 AI Data Assistant")
        
        if st.button("📊 Generate Business Insights"):

            prompt = f"""
            Analyze this dataset and provide:

            1. Key business insights
            2. Important trends
            3. Risks
            4. Recommendations

            Dataset Columns:
            {list(df.columns)}
            """

            with st.spinner("🤖 Generating Business Insights... Please wait"):
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )

            st.success(response.text)
            
            st.download_button(
                label="📥 Download Report",
                data=response.text,
                file_name="Business_Insights_Report.txt",
                mime="text/plain"
            )

        question = st.text_input(
            "Ask anything about your dataset:"
            
        )

        if st.button("Ask AI"):

            if question:

                prompt = f"""
                Dataset Columns:
                {list(df.columns)}

                Question:
                {question}
                """

                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )

                st.success(response.text)

