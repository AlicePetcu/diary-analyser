import glob
import streamlit as st
import plotly.express as px
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

filepaths = sorted(glob.glob("diary/*.txt"))

analyzer = SentimentIntensityAnalyzer()
pos = []
neg = []
days = []
for filepath in filepaths:
    days.append(filepath[6:-4])
    with open(filepath, "r") as file:
        content = file.read()
    score = analyzer.polarity_scores(content)
    pos.append(score["pos"])
    neg.append(score["neg"])

st.title("Diary Analyser")

st.subheader("Positivity")
figure_pos = px.line(x=days, y=pos, labels={"x": "Days", "y": "Positivity"})
st.plotly_chart(figure_pos)

st.subheader("Negativity")
figure_neg = px.line(x=days, y=neg, labels={"x": "Days", "y": "Negativity"})
st.plotly_chart(figure_neg)

