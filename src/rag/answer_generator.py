# ------------------------------------------------------------------------------
# 📄 AnswerGenerator Module for B5W6 – Gemini Integration (v3 Robust with Self-Evaluation)
# ------------------------------------------------------------------------------
# Author: Nabil Mohamed
# Date: July 2025
# Description:
#   Generates answers and performs automated qualitative evaluation using
#   Google's Gemini API (free tier). Includes robust JSON parsing and fallback.
# ------------------------------------------------------------------------------

# ---------------------------
# Standard Library Imports
# ---------------------------
import requests  # For making API calls
import json  # For handling JSON safely


# ---------------------------
# AnswerGenerator Class
# ---------------------------


class AnswerGenerator:
    """
    Generates answers and performs self-evaluation using Google's Gemini API.
    """

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        """
        Initializes the AnswerGenerator instance.

        Args:
            api_key (str): Google AI Studio API key.
            model (str): Gemini model name (default: 'gemini-2.5-flash').
        """
        try:
            if not api_key:
                raise ValueError("❌ API key must be provided.")

            self.api_key = api_key  # Store API key
            self.model = model  # Store selected model
            self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

        except Exception as e:
            raise RuntimeError(f"❌ Failed to initialize AnswerGenerator: {e}") from e

    def generate_answer(self, prompt: str) -> str:
        """
        Generates an answer from the provided prompt using Gemini.

        Args:
            prompt (str): Assembled prompt text.

        Returns:
            str: Generated answer or placeholder if failed.
        """
        try:
            if not prompt or not isinstance(prompt, str):
                raise ValueError("❌ Prompt must be a non-empty string.")

            headers = {"Content-Type": "application/json"}
            payload = {"contents": [{"parts": [{"text": prompt}]}]}

            response = requests.post(
                url=f"{self.api_url}?key={self.api_key}",
                headers=headers,
                data=json.dumps(payload),
            )

            response.raise_for_status()

            answer = (
                response.json()
                .get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            ).strip()

            if not answer:
                raise ValueError("❌ No answer returned from Gemini API.")

            return answer

        except Exception as e:
            raise RuntimeError(f"❌ Failed to generate answer using Gemini: {e}") from e

    def evaluate_answer(
        self, question: str, context: str, generated_answer: str
    ) -> dict:
        """
        Evaluates the quality of an answer using Gemini with robust parsing.

        Args:
            question (str): Original business question.
            context (str): Retrieved context (top complaint chunks).
            generated_answer (str): AI-generated answer to evaluate.

        Returns:
            dict: Evaluation results including relevance, accuracy, completeness, score, and comments.
        """
        try:
            if not all([question, context, generated_answer]):
                raise ValueError(
                    "❌ Question, context, and answer must all be provided for evaluation."
                )

            evaluation_prompt = f"""
You are an impartial evaluator assessing the quality of an AI-generated answer based on the provided context.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{generated_answer}

Evaluate strictly in this JSON format:
{{
  "Relevance": "",
  "Accuracy": "",
  "Completeness": "",
  "Quality Score": "",
  "Comments": ""
}}
""".strip()

            headers = {"Content-Type": "application/json"}
            payload = {"contents": [{"parts": [{"text": evaluation_prompt}]}]}

            response = requests.post(
                url=f"{self.api_url}?key={self.api_key}",
                headers=headers,
                data=json.dumps(payload),
            )

            response.raise_for_status()

            # ✅ Extract raw text
            raw_text = (
                response.json()
                .get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            ).strip()

            if not raw_text:
                raise ValueError("❌ Gemini returned an empty evaluation response.")

            # ✅ Safely extract JSON substring
            json_start = raw_text.find("{")
            json_end = raw_text.rfind("}") + 1
            json_text = raw_text[json_start:json_end]

            # ✅ Parse the JSON safely
            evaluation_result = json.loads(json_text)

            return evaluation_result

        except Exception as e:
            print(f"❌ Gemini evaluation failed: {e}")
            print(
                "🔍 Raw response for debugging:\n",
                raw_text if "raw_text" in locals() else "No response captured.",
            )

            # ✅ Return safe fallback to prevent pipeline breakage
            return {
                "Relevance": "❌",
                "Accuracy": "❌",
                "Completeness": "❌",
                "Quality Score": "1",
                "Comments": "No valid evaluation returned. Check prompt format or API limits.",
            }
