"""
eda_visualizer.py – Exploratory Data Analysis (EDA) Generator for Complaints (B5W6)
-----------------------------------------------------------------------------------
Generates interactive EDA visualizations for cleaned consumer complaint datasets.
This module is decoupled from data loading/cleaning for modularity and reuse.

Core Features:
  • Product distribution bar chart
  • Narrative length histogram
  • Missing value heatmap
  • Monthly complaint volume time series (if available)
  • WordCloud of common complaint keywords

Author: Nabil Mohamed
"""

# ------------------------------------------------------------------------------
# 📦 Imports
# ------------------------------------------------------------------------------
import pandas as pd  # Data handling
import matplotlib.pyplot as plt  # Plotting
import seaborn as sns  # Visualization styling
from collections import Counter  # Word counting
from wordcloud import WordCloud  # WordCloud generation
import nltk  # NLP utilities

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords


# ------------------------------------------------------------------------------
# 🧠 Class: ComplaintEDAVisualizer
# ------------------------------------------------------------------------------
class ComplaintEDAVisualizer:
    """
    Generates exploratory data analysis (EDA) visualizations for complaint datasets.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the visualizer with a cleaned complaint DataFrame.

        Args:
            df (pd.DataFrame): Cleaned complaint dataset with narratives.
        Raises:
            ValueError: If input DataFrame is missing or empty.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("❌ Input must be a pandas DataFrame.")
        if df.empty:
            raise ValueError("❌ Provided DataFrame is empty.")
        self.df = df.copy()
        self.stop_words = set(stopwords.words("english"))

    def generate_all(self) -> None:
        """
        Runs all EDA visualizations in sequence.
        """
        self.plot_product_distribution()
        self.plot_narrative_length_distribution()
        self.plot_missingness_heatmap()
        self.plot_time_trends()
        self.plot_wordcloud()

    def plot_product_distribution(self) -> None:
        """
        Plots complaint count distribution by financial product.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(
            y="Product",
            data=self.df,
            order=self.df["Product"].value_counts().index,
            palette="Blues_r",
        )
        plt.title("Complaint Volume by Product")
        plt.xlabel("Number of Complaints")
        plt.ylabel("Financial Product")
        plt.tight_layout()
        plt.show()

    def plot_narrative_length_distribution(self) -> None:
        """
        Plots the distribution of complaint narrative lengths (word counts).
        """
        self.df["narrative_length"] = (
            self.df["Consumer complaint narrative"].str.split().str.len()
        )
        plt.figure(figsize=(10, 5))
        sns.histplot(self.df["narrative_length"], bins=50, color="skyblue")
        plt.title("Distribution of Complaint Narrative Lengths")
        plt.xlabel("Number of Words")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()

    def plot_missingness_heatmap(self) -> None:
        """
        Plots a heatmap showing missing values across the dataset.
        """
        plt.figure(figsize=(10, 4))
        sns.heatmap(self.df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
        plt.title("Missing Values Heatmap")
        plt.tight_layout()
        plt.show()

    def plot_time_trends(self) -> None:
        """
        Plots monthly complaint volumes over time if date information is available.
        """
        if "Date received" not in self.df.columns:
            print("⚠️ Skipping time trends: 'Date received' column not found.")
            return

        self.df["Date received"] = pd.to_datetime(
            self.df["Date received"], errors="coerce"
        )
        time_series = self.df.groupby(self.df["Date received"].dt.to_period("M")).size()

        plt.figure(figsize=(12, 6))
        time_series.plot()
        plt.title("Monthly Complaint Volume Over Time")
        plt.xlabel("Month")
        plt.ylabel("Number of Complaints")
        plt.tight_layout()
        plt.show()

    def plot_wordcloud(self) -> None:
        """
        Generates a WordCloud from complaint narratives after basic stopword removal.
        """
        text_blob = " ".join(self.df["Consumer complaint narrative"].dropna().tolist())
        words = [
            word
            for word in text_blob.lower().split()
            if word not in self.stop_words and len(word) > 3
        ]
        word_freq = Counter(words)
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)

        plt.figure(figsize=(15, 7))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Most Frequent Words in Complaint Narratives")
        plt.tight_layout()
        plt.show()
