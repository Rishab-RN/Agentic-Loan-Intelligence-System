"""
ALIS — ExplainerVoice: Ollama LLM Integration
================================================
Optionally refines template-based explanations using a local LLM
(llama3.2:3b via Ollama) for more natural, conversational tone.

Design principle: templates are the BACKBONE, LLM is the POLISH.
If Ollama is not running, the system works perfectly with templates alone.
The LLM never changes facts, numbers, or advice — it only rephrases
for more natural flow.

Why Ollama + llama3.2:3b:
  - Runs 100% locally on CPU (no API keys, no cost, no data leaks)
  - 3B params = fast inference even on a laptop (~2-3 seconds)
  - Small enough for a competition demo laptop
  - Surprisingly good at paraphrasing and tone matching

Usage:
    python llm_engine.py  # runs Ramu example with/without Ollama
"""

import json
from pathlib import Path

from templates import RAMU_EXAMPLE, build_explanation

# Ollama config
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_BASE_URL = "http://localhost:11434"

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


class ExplainerLLM:
    """
    LLM-powered explanation refiner.

    Uses Ollama for optional natural language polish on top of
    template-generated explanations.
    """

    def __init__(self, model: str = OLLAMA_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url
        self.available = self._check_ollama()

    def _check_ollama(self) -> bool:
        """Check if Ollama is running and the model is available."""
        try:
            import httpx
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                if any(self.model in m for m in models):
                    return True
                print(f"  ⚠ Ollama running but {self.model} not found.")
                print(f"    Run: ollama pull {self.model}")
                return False
        except Exception:
            pass
        print("  ⚠ Ollama not running. Using template-only mode (still works great).")
        return False

    def generate_vernacular_explanation(
        self,
        decision_data: dict,
        language: str = "english",
    ) -> str:
        """
        Generate a warm, vernacular explanation for a loan decision.

        Pipeline:
          1. Build template-based explanation (always works)
          2. If Ollama available, refine with LLM for natural flow
          3. Return final explanation as plain text

        Parameters
        ----------
        decision_data : dict
            Full decision packet from RiskMind + FraudSentinel + ComplianceGuard.
        language : str
            "english", "kannada", or "hindi".

        Returns
        -------
        str — plain text explanation (no markdown).
        """
        # Step 1: Template-based explanation (the backbone)
        template_explanation = build_explanation(decision_data, language)

        # Step 2: LLM refinement (optional polish)
        if self.available:
            try:
                refined = self._refine_with_llm(template_explanation, language)
                if refined and len(refined) > 50:
                    return refined
            except Exception as e:
                print(f"  ⚠ LLM refinement failed: {e}. Using template output.")

        return template_explanation

    def _refine_with_llm(self, text: str, language: str) -> str:
        """Send the template explanation to Ollama for natural language polish."""
        import httpx

        language_instruction = {
            "english": (
                "Rewrite this loan decision explanation to sound more warm and "
                "conversational, like a helpful older brother explaining it simply. "
                "Keep ALL numbers, scores, advice, and facts EXACTLY the same. "
                "Only improve the flow and warmth. Do NOT add any new information. "
                "Output plain text only, no markdown."
            ),
            "kannada": (
                "ಈ ಸಾಲ ನಿರ್ಧಾರ ವಿವರಣೆಯನ್ನು ಹೆಚ್ಚು ಬೆಚ್ಚಗಾಗಿ ಮತ್ತು ಸಂವಾದಾತ್ಮಕವಾಗಿ "
                "ಬರೆಯಿರಿ, ಒಬ್ಬ ಸಹಾಯಕ ಅಣ್ಣ ಸರಳವಾಗಿ ಹೇಳಿದಂತೆ. "
                "ಎಲ್ಲಾ ಸಂಖ್ಯೆಗಳು, ಸ್ಕೋರ್‌ಗಳು, ಸಲಹೆಗಳನ್ನು ಹಾಗೆಯೇ ಇಡಿ. "
                "ಕೇವಲ ಭಾಷೆಯ ಹರಿವನ್ನು ಸುಧಾರಿಸಿ. ಸರಳ ಪಠ್ಯ ಮಾತ್ರ."
            ),
            "hindi": (
                "इस लोन फैसले की व्याख्या को और गर्मजोशी से लिखें, "
                "जैसे एक मददगार बड़ा भाई सरल भाषा में समझा रहा हो। "
                "सभी नंबर, स्कोर, सलाह बिल्कुल वही रखें। "
                "केवल भाषा का प्रवाह सुधारें। सादा टेक्स्ट, कोई मार्कडाउन नहीं।"
            ),
        }

        prompt = f"""{language_instruction.get(language, language_instruction['english'])}

---
{text}
---"""

        response = httpx.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 1024,
                },
            },
            timeout=30,
        )

        if response.status_code == 200:
            return response.json().get("response", "").strip()
        return ""

    def generate_explanation_for_orchestrator(
        self,
        riskmind_result: dict,
        fraud_result: dict,
        compliance_result: dict,
        applicant_name: str = "Applicant",
        loan_amount: int = 25000,
        language: str = "english",
    ) -> str:
        """
        Main entry point when called by the LoanOrchestrator.

        Combines outputs from all three upstream agents into a single
        decision packet, then generates the vernacular explanation.
        """
        # Determine decision based on upstream results
        fraud_level = fraud_result.get("risk_level", "CLEAN")
        is_compliant = compliance_result.get("is_compliant", True)
        credit_score = riskmind_result.get("credit_score", 0)

        if fraud_level in ("HIGH_RISK", "BLOCK"):
            decision = "FRAUD_FLAGGED"
        elif not is_compliant:
            decision = "MORE_INFO_NEEDED"
        elif credit_score >= 600:
            decision = "APPROVED"
        else:
            decision = "REJECTED"

        decision_data = {
            "applicant_name": applicant_name,
            "decision": decision,
            "credit_score": credit_score,
            "loan_amount": loan_amount,
            "shap_values": riskmind_result.get("shap_values", {}),
            "fraud_risk_level": fraud_level,
            "compliance_status": is_compliant,
            "missing_items": compliance_result.get("recommended_corrections", []),
            "cooling_off_days": 3,
            "days_to_improve": 45,
            "helpline": "1800-XXX-XXXX",
        }

        return self.generate_vernacular_explanation(decision_data, language)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*60}")
    print("  ALIS ExplainerVoice — LLM Engine Demo")
    print(f"{'='*60}\n")

    engine = ExplainerLLM()

    mode = "LLM-refined" if engine.available else "Template-only (Ollama not running)"
    print(f"  Mode: {mode}\n")

    # Ramu example in Kannada
    print("  Generating Ramu's explanation in Kannada...\n")
    explanation = engine.generate_vernacular_explanation(RAMU_EXAMPLE, "kannada")
    print(explanation)

    print(f"\n{'='*60}")
    print("  ✓ Done.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
