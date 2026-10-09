"""Prove task: theorem statement -> simple reflection loop -> proof.

Entry point: `run_prove(problem, llm, retriever_mode, verifier, ...)`.
"""

from .run import ProveLLMs, ProveProblem, ProveResult, run_prove
from .integrity import (
    IntegrityResult,
    ProofHoleContract,
    split_sorry_proof_hole,
    validate_proof_integrity,
)
from .verifier import LeanInteractVerifier, Verifier, VerifyResult
from .budget import A2CallBudget, BudgetedLLMClient, BudgetExhausted

__all__ = [
    "ProveLLMs", "ProveProblem", "ProveResult", "run_prove",
    "IntegrityResult", "ProofHoleContract", "split_sorry_proof_hole",
    "validate_proof_integrity",
    "Verifier", "LeanInteractVerifier", "VerifyResult",
    "A2CallBudget", "BudgetedLLMClient", "BudgetExhausted",
]
