"""
schema_auditor.py – Complaint Data Schema Diagnostic Tool (B5W6)
------------------------------------------------------------------------------
Provides detailed structural diagnostics on consumer complaint datasets.
Summarizes missingness, uniqueness, type consistency, and schema readiness for text processing.

Core responsibilities:
  • Computes per-column metrics: dtype, n_unique, % missing, constant-value flags
  • Flags high-null fields with severity bands for risk assessment
  • Identifies non-informative or problematic columns (empty, constant, overly unique)
  • Supports styled schema summaries for EDA and documentation
  • Raises clear, actionable exceptions for invalid inputs

Used in Task 1 EDA, text cleaning audits, and downstream embedding validation.

Author: Nabil Mohamed
"""

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Standard Library Imports
# ───────────────────────────────────────────────────────────────────────────────
from typing import List  # For type hinting list inputs

# ───────────────────────────────────────────────────────────────────────────────
# 📦 Third-Party Imports
# ───────────────────────────────────────────────────────────────────────────────
import pandas as pd  # For DataFrame handling


# ───────────────────────────────────────────────────────────────────────────────
# 🧠 Class: ComplaintSchemaAuditor
# ───────────────────────────────────────────────────────────────────────────────
class ComplaintSchemaAuditor:
    """
    Class for performing schema-level structural diagnostics on complaint datasets.
    Provides missingness, uniqueness, stability, and readiness checks for text processing.
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the schema auditor with the provided DataFrame.

        Args:
            df (pd.DataFrame): Complaint-level dataset for intelligent analysis.

        Raises:
            TypeError: If input is not a pandas DataFrame.
            ValueError: If the DataFrame is empty or lacks columns.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Input must be a pandas DataFrame, received {type(df)}.")

        if df.empty or df.shape[1] == 0:
            raise ValueError("Input DataFrame is empty or contains no columns.")

        self.df = df.copy()  # Defensive copy
        self.schema_df = None  # Placeholder for schema summary

    def summarize_schema(self) -> pd.DataFrame:
        """
        Computes and stores per-column schema metrics for complaint data.

        Returns:
            pd.DataFrame: DataFrame summarizing schema characteristics.
        """
        try:
            schema = pd.DataFrame(
                {
                    "dtype": self.df.dtypes.astype(str),  # Data type of each column
                    "n_unique": self.df.nunique(
                        dropna=False
                    ),  # Number of unique values
                    "n_missing": self.df.isna().sum(),  # Number of missing entries
                }
            )

            schema["%_missing"] = (schema["n_missing"] / len(self.df) * 100).round(
                2
            )  # Missing % rounded
            schema["is_constant"] = (
                schema["n_unique"] <= 1
            )  # Flags columns with no variation
            schema["high_null_flag"] = pd.cut(  # Categorizes missingness severity
                schema["%_missing"],
                bins=[-1, 0, 20, 50, 100],
                labels=["✅ OK", "🟡 Moderate", "🟠 High", "🔴 Critical"],
            )

            self.schema_df = schema.sort_values(
                "%_missing", ascending=False
            )  # Sort by most missing
            return self.schema_df

        except Exception as e:
            raise RuntimeError(f"Error generating schema summary: {e}")

    def styled_summary(self):
        """
        Creates a visually enhanced schema summary for Jupyter display.

        Returns:
            pd.io.formats.style.Styler: Styled DataFrame with heatmap and flag highlights.
        """
        try:
            if self.schema_df is None:
                self.summarize_schema()

            styled = (
                self.schema_df.style.background_gradient(
                    subset="%_missing", cmap="OrRd"
                )
                .applymap(
                    lambda val: (
                        "background-color: gold; font-weight: bold;" if val else ""
                    ),
                    subset=["is_constant"],
                )
                .format({"%_missing": "{:.2f}"})
            )

            return styled

        except Exception as e:
            raise RuntimeError(f"Error styling schema summary: {e}")

    def print_diagnostics(self) -> None:
        """
        Prints concise diagnostics on schema health for complaint analysis.

        Raises:
            RuntimeError: If diagnostics cannot be generated.
        """
        try:
            if self.schema_df is None:
                self.summarize_schema()

            n_const = self.schema_df["is_constant"].sum()
            n_null_20 = (self.schema_df["%_missing"] > 20).sum()
            n_null_50 = (self.schema_df["%_missing"] > 50).sum()

            print("\n📝 Complaint Schema Diagnostics:")
            print(f"• Constant-value columns:    {n_const}")
            print(f"• Columns >20% missing:      {n_null_20}")
            print(f"• Columns >50% missing:      {n_null_50}")

        except Exception as e:
            raise RuntimeError(f"Error printing diagnostics: {e}")

    def check_duplicate_ids(self, id_columns: List[str]) -> None:
        """
        Checks for duplicate values in key identifier columns.

        Args:
            id_columns (List[str]): List of columns expected to be unique (e.g., Complaint IDs).

        Raises:
            ValueError: If columns do not exist or duplicates are found.
        """
        try:
            for col in id_columns:
                if col not in self.df.columns:
                    raise ValueError(f"Identifier column '{col}' not found.")

                n_duplicates = self.df[col].duplicated().sum()
                print(
                    f"• {col}: {n_duplicates:,} duplicates"
                    + (" ⚠️ Potential integrity risk." if n_duplicates > 0 else "")
                )

        except Exception as e:
            raise RuntimeError(f"Error checking duplicate IDs: {e}")
