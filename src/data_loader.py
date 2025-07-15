"""
data_loader.py – Complaint Data Loader with Product Mapping (B5W6)
------------------------------------------------------------------------------
Processes large-scale complaint datasets safely in chunks, maps products using
both Product and Sub-product fields, filters and cleans narratives, and provides
interactive diagnostics.

Key Features:
  • Hybrid Product–Sub-product mapping to 5 target categories
  • Robust filtering, text cleaning, and diagnostics
  • Memory-safe chunk processing with detailed logging
  • Cleaned file loader for downstream tasks (Task 2+)

Author: Nabil Mohamed
"""

# ---------------------------
# Standard Library Imports
# ---------------------------
import os  # For file path validation
import re  # For text cleaning
import pandas as pd  # For data handling
from tqdm import tqdm  # For progress tracking

# ---------------------------
# ComplaintChunkProcessor Class
# ---------------------------


class ComplaintChunkProcessor:
    """
    Safely loads, maps, filters, and cleans CFPB complaint datasets in chunks.
    """

    def __init__(self, filepath: str, output_path: str, chunk_size: int = 100_000):
        """
        Initializes the processor with file paths and chunk size.

        Args:
            filepath (str): Path to input complaint CSV file.
            output_path (str): Path to save cleaned file.
            chunk_size (int): Number of rows per chunk for processing.
        """
        if not isinstance(filepath, str) or not isinstance(output_path, str):
            raise TypeError(
                "❌ Both filepath and output_path must be strings."
            )  # Validate input types
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f"❌ Cannot find file at: {filepath}"
            )  # Validate file existence

        self.filepath = filepath  # Store file path
        self.output_path = output_path  # Store output path
        self.chunk_size = chunk_size  # Store chunk size

        self.allowed_products = [  # Define allowed product categories
            "Credit card",
            "Personal loan",
            "Buy Now, Pay Later",
            "Savings account",
            "Money transfer, virtual currency",
        ]

        self.stats = {  # Initialize statistics tracker
            "chunks_processed": 0,
            "rows_loaded": 0,
            "rows_kept": 0,
            "rows_dropped_total": 0,
            "rows_dropped_no_narrative": 0,
            "rows_dropped_wrong_product": 0,
            "products_found": {},
        }

        self.cleaned_df = None  # Initialize placeholder for cleaned DataFrame

    def map_product(self, product: str, sub_product: str) -> str:
        """
        Maps Product and Sub-product to target categories.
        """
        product = str(product).lower()  # Convert to lowercase
        sub_product = str(sub_product).lower()  # Convert to lowercase

        if "credit card" in product or "credit card" in sub_product:
            return "Credit card"
        elif any(term in product for term in ["payday", "personal loan"]) or any(
            term in sub_product for term in ["loan", "installment"]
        ):
            return "Personal loan"
        elif "buy now" in product or "bnpl" in sub_product:
            return "Buy Now, Pay Later"
        elif any(term in product for term in ["savings", "checking"]):
            return "Savings account"
        elif any(term in product for term in ["money transfer", "virtual currency"]):
            return "Money transfer, virtual currency"
        else:
            return "Unmapped"

    def clean_text(self, text: str) -> str:
        """
        Cleans complaint narrative text.
        """
        if pd.isna(text):
            return ""  # Return empty string for NaN
        text = re.sub(r"\s+", " ", str(text)).strip()  # Remove excess whitespace
        text = re.sub(r'[^A-Za-z0-9.,!?\'" ]+', "", text)  # Remove special characters
        return text.lower()  # Lowercase text

    def process_chunks(self, return_dataframe: bool = True) -> pd.DataFrame:
        """
        Processes complaints: maps products, filters, cleans, saves, and optionally returns DataFrame.
        """
        filtered_data = []  # List to hold processed chunks

        try:
            for chunk in tqdm(
                pd.read_csv(self.filepath, chunksize=self.chunk_size),
                desc="🚀 Processing Chunks",
            ):
                self.stats["chunks_processed"] += 1
                self.stats["rows_loaded"] += len(chunk)

                chunk["MappedProduct"] = chunk.apply(
                    lambda row: self.map_product(
                        row.get("Product", ""), row.get("Sub-product", "")
                    ),
                    axis=1,
                )

                original_size = len(chunk)
                chunk = chunk[chunk["MappedProduct"].isin(self.allowed_products)]
                self.stats["rows_dropped_wrong_product"] += original_size - len(chunk)

                pre_narrative_size = len(chunk)
                chunk = chunk.dropna(subset=["Consumer complaint narrative"])
                chunk = chunk[chunk["Consumer complaint narrative"].str.strip() != ""]
                self.stats["rows_dropped_no_narrative"] += pre_narrative_size - len(
                    chunk
                )

                chunk["Consumer complaint narrative"] = chunk[
                    "Consumer complaint narrative"
                ].apply(self.clean_text)

                product_counts = chunk["MappedProduct"].value_counts().to_dict()
                for product, count in product_counts.items():
                    self.stats["products_found"][product] = (
                        self.stats["products_found"].get(product, 0) + count
                    )

                self.stats["rows_kept"] += len(chunk)
                filtered_data.append(chunk)

        except pd.errors.ParserError as e:
            raise RuntimeError(f"❌ CSV parsing failed: {e}")
        except Exception as e:
            raise RuntimeError(f"❌ Unexpected error: {e}")

        if filtered_data:
            try:
                self.cleaned_df = pd.concat(filtered_data, ignore_index=True)
                self.cleaned_df.to_csv(self.output_path, index=False)
                self.stats["rows_dropped_total"] = (
                    self.stats["rows_loaded"] - self.stats["rows_kept"]
                )
                self._print_summary()
                if return_dataframe:
                    return self.cleaned_df
            except Exception as e:
                raise RuntimeError(f"❌ Failed to save cleaned dataset: {e}")
        else:
            print("⚠️ No matching complaints found.")
            return pd.DataFrame()

    def _print_summary(self):
        """
        Prints filtering summary and category distribution.
        """
        print("\n✅ Complaint Data Cleaning Complete")
        print(f"• Chunks processed:         {self.stats['chunks_processed']:,}")
        print(f"• Total rows loaded:        {self.stats['rows_loaded']:,}")
        print(f"• Rows retained:            {self.stats['rows_kept']:,}")
        print(f"• Rows dropped (total):     {self.stats['rows_dropped_total']:,}")
        print("\n❌ Drop Reasons:")
        print(
            f"• Wrong product:            {self.stats['rows_dropped_wrong_product']:,}"
        )
        print(
            f"• Missing narrative:        {self.stats['rows_dropped_no_narrative']:,}"
        )
        print("\n📊 Product Distribution (Filtered):")
        for product, count in self.stats["products_found"].items():
            print(f"• {product}: {count:,} complaints")
        print(f"\n📄 Cleaned dataset saved to: {self.output_path}")

    def load_cleaned_data(self) -> pd.DataFrame:
        """
        Loads an already cleaned complaint dataset from CSV.

        Returns:
            pd.DataFrame: Loaded complaint dataset.
        """
        try:
            if not os.path.isfile(self.filepath):  # Check if file exists
                raise FileNotFoundError(f"❌ Cleaned file not found: {self.filepath}")

            df = pd.read_csv(self.filepath)  # Load cleaned CSV

            if df.empty:  # Check for empty DataFrame
                raise ValueError(
                    f"⚠️ The loaded complaint dataset is empty: {self.filepath}"
                )

            self.cleaned_df = df  # Store in class for optional reuse

            print(
                f"✅ Cleaned complaint dataset loaded successfully: {df.shape[0]:,} rows × {df.shape[1]} columns"
            )
            return df  # Return loaded DataFrame

        except FileNotFoundError as fnf_error:
            print(str(fnf_error))
            return pd.DataFrame()

        except Exception as e:
            print(f"❌ Failed to load cleaned complaint data: {e}")
            return pd.DataFrame()
