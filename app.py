"""Streamlit frontend for multilingual legal semantic retrieval."""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.evaluator import evaluate_all_queries
from src.retriever import LegalSemanticRetriever


st.set_page_config(page_title="Indian Legal Semantic Retrieval", page_icon=":scales:", layout="wide")


LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Gujarati": "gu",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Bengali": "bn",
    "Punjabi": "pa",
    "Kannada": "kn",
    "Malayalam": "ml",
}


@st.cache_resource
def get_retriever() -> LegalSemanticRetriever:
    """Create and cache the retrieval pipeline for the app session."""
    return LegalSemanticRetriever()


def main() -> None:
    """Render the main Streamlit application."""
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #f7f4ed 0%, #ffffff 35%, #eef3f8 100%);
                color: #17202a;
            }
            .block-container {
                padding-top: 2rem;
            }
            .translated-query {
                background: #eff6ff;
                color: #12324a;
                border: 1px solid #bfdbfe;
                border-radius: 14px;
                padding: 0.9rem 1rem;
                font-size: 1rem;
                line-height: 1.5;
            }
            .case-card {
                background: rgba(255, 255, 255, 0.96);
                padding: 1.15rem 1.2rem;
                border-radius: 18px;
                border: 1px solid #d7dee7;
                box-shadow: 0 12px 28px rgba(60, 69, 79, 0.10);
                color: #16212c !important;
                margin-bottom: 1rem;
            }
            .case-title {
                color: #142433 !important;
                font-size: 1.35rem;
                font-weight: 700;
                margin-bottom: 0.55rem;
                line-height: 1.35;
            }
            .case-meta {
                color: #34516b !important;
                font-size: 0.95rem;
                margin-bottom: 0.85rem;
            }
            .case-summary {
                color: #1d2b38 !important;
                font-size: 1rem;
                line-height: 1.65;
                margin-bottom: 0.9rem;
            }
            .score-label {
                color: #31495f !important;
                font-size: 0.94rem;
                font-weight: 600;
                margin-bottom: 0.35rem;
            }
            .score-track {
                width: 100%;
                height: 12px;
                border-radius: 999px;
                background: #dbe6f3;
                overflow: hidden;
                margin-bottom: 0.9rem;
            }
            .score-fill {
                height: 100%;
                border-radius: 999px;
                background: linear-gradient(90deg, #1d4ed8 0%, #38bdf8 100%);
            }
            .metric-chip {
                padding: 0.42rem 0.8rem;
                background: #efe7d8;
                color: #5b4632 !important;
                border-radius: 999px;
                display: inline-block;
                font-size: 0.92rem;
                line-height: 1.45;
                border: 1px solid #e3d6bf;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Democratizing Legal Information in India")
    st.caption("Multilingual semantic retrieval inspired by the MEDCOM 2025 IEEE paper.")

    retriever = get_retriever()
    col1, col2 = st.columns([2, 1])

    with col1:
        language_label = st.selectbox("Choose query language", list(LANGUAGES.keys()), index=2)
        query_text = st.text_area(
            "Enter your legal query",
            height=130,
            placeholder="\u0a89\u0aa6\u0abe\u0ab9\u0ab0\u0aa3: \u0a9b\u0ac7\u0aa4\u0ab0\u0aaa\u0abf\u0a82\u0aa1\u0ac0 \u0aae\u0abe\u0a9f\u0ac7 \u0aa8\u0a95\u0ab2\u0ac0 \u0ab0\u0ab8\u0ac0\u0aa6\u0acb \u0ab5\u0aa1\u0ac7 \u0aaa\u0ac8\u0ab8\u0abe \u0ab5\u0ab8\u0ac2\u0ab2\u0ab5\u0abe\u0aa8\u0abe \u0a95\u0ac7\u0ab8",
        )

    with col2:
        top_k = st.slider("Top-K results", 1, 10, 5)
        run_eval = st.button("Run Evaluation", use_container_width=True)
        search_now = st.button("Search Cases", type="primary", use_container_width=True)

    if search_now and query_text.strip():
        response = retriever.retrieve(query_text, LANGUAGES[language_label], top_k)
        st.subheader("Translated Query")
        st.markdown(
            f"<div class='translated-query'>{response['translated_query']}</div>",
            unsafe_allow_html=True,
        )
        result_df = pd.DataFrame(response["results"])

        st.download_button(
            "Download JSON",
            data=json.dumps(response, ensure_ascii=False, indent=2),
            file_name="legal_search_results.json",
            mime="application/json",
        )
        st.download_button(
            "Download CSV",
            data=result_df.to_csv(index=False),
            file_name="legal_search_results.csv",
            mime="text/csv",
        )

        st.subheader("Top Relevant Cases")
        for result in response["results"]:
            score_percent = max(0.0, min(result["score"], 1.0)) * 100
            citations = ", ".join(result["citations"]) if result["citations"] else "No citations available"
            st.markdown(
                f"""
                <div class="case-card">
                    <div class="case-title">{result['rank']}. {result['title']}</div>
                    <div class="case-meta"><strong>Case ID:</strong> {result['case_id']}</div>
                    <div class="case-summary">{result['summary']}</div>
                    <div class="score-label">Relevance Score: {result['score']:.4f}</div>
                    <div class="score-track">
                        <div class="score-fill" style="width: {score_percent:.2f}%"></div>
                    </div>
                    <span class="metric-chip">Citations: {citations}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if run_eval:
        with st.spinner("Evaluating semantic search against BM25..."):
            metrics = evaluate_all_queries(retriever)
        st.subheader("Evaluation Metrics")
        left, right = st.columns(2)
        left.json(metrics["semantic"])
        right.json(metrics["bm25"])
        st.image(metrics["plot_path"], caption="BM25 vs Semantic Retrieval")


if __name__ == "__main__":
    main()
