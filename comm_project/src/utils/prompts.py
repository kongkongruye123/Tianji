# -*- coding: utf-8 -*-
"""
This module freezes the core prompts, schemas, and templates for the project.
All subsequent steps (data generation, training, evaluation, demo) must import from this file
to ensure consistency.
"""

import json

# --- System Prompt (Frozen as per Plan 0.5) ---
SYSTEM_PROMPT = ("""
你是通信标准问答专家。你必须严格遵守以下规则：
1. 只能根据【证据】部分提供的内容回答问题，禁止使用任何外部知识。
2. 如果【证据】不足以回答问题，必须明确拒绝回答，并在 `cannot_answer_reason` 字段中说明原因。
3. 你的输出必须且只能是一个合法的 JSON 对象，不包含任何 JSON 之外的额外文本、解释或标记（如 ```json）。
""")

# --- User Prompt Template (Frozen as per Plan 0.5) ---
# This template will be used to format the input for the model.
# --- User Prompt Template (Frozen as per Plan 0.5) ---
# NOTE: evidence must be provided as a sequence of chunks with explicit headers.
# Required evidence chunk header format per chunk:
#   [chunk_id={...} doc_id={...} section={...}]
#   {text}
USER_PROMPT_TEMPLATE = (
    "问题: {question}\n"
    "证据:\n"
    "{evidence_text}\n"
)


def format_evidence_chunks(chunks):
    """Format evidence chunks into the frozen evidence text block.

    Args:
        chunks (list[dict]): Each chunk must contain:
            - chunk_id (str)
            - doc_id (str)
            - section (str)
            - text (str)

    Returns:
        str: Evidence text ready to be inserted into USER_PROMPT_TEMPLATE.
    """
    lines = []
    for c in chunks:
        lines.append(
            f"[chunk_id={c['chunk_id']} doc_id={c['doc_id']} section={c['section']}]\n{c['text']}"
        )
    return "\n\n".join(lines)


def format_user_prompt(question, chunks=None, evidence_text=None):
    """Frozen user prompt formatter.

    Provide either `chunks` (preferred) or `evidence_text` (already formatted).

    Args:
        question (str): User question.
        chunks (list[dict] | None): Evidence chunks.
        evidence_text (str | None): Pre-formatted evidence text.

    Returns:
        str: Final user prompt string.
    """
    if evidence_text is None:
        evidence_text = "" if chunks is None else format_evidence_chunks(chunks)
    return USER_PROMPT_TEMPLATE.format(question=question, evidence_text=evidence_text)

# --- Output JSON Schema (Frozen as per Plan 0.4) ---
# Represented as a Python dictionary for validation and reference.
OUTPUT_SCHEMA = {
    "answer": "string",
    "citations": [
        {
            "doc_id": "string",
            "section": "string",
            "chunk_id": "string",
            "quote": "string"
        }
    ],
    "confidence": "low|mid|high",
    "cannot_answer_reason": "string|null"
}


def validate_json_output(json_obj):
    """
    A lightweight validator to check if a JSON object conforms to the OUTPUT_SCHEMA.

    Args:
        json_obj (dict): The JSON object (as a Python dict) to validate.

    Returns:
        tuple[bool, str]: A tuple containing a boolean indicating validity
                          and a string with an error message if invalid.
    """
    if not isinstance(json_obj, dict):
        return False, "Output is not a dictionary."

    # Check for required top-level keys
    required_keys = ["answer", "citations", "confidence", "cannot_answer_reason"]
    for key in required_keys:
        if key not in json_obj:
            return False, f"Missing required key: '{key}'"

    # Check types and constraints
    if not isinstance(json_obj["answer"], str):
        return False, "'answer' must be a string."

    if not isinstance(json_obj["citations"], list):
        return False, "'citations' must be a list."

    # Check citation items
    citation_keys = ["doc_id", "section", "chunk_id", "quote"]
    for i, citation in enumerate(json_obj["citations"]):
        if not isinstance(citation, dict):
            return False, f"Citation at index {i} is not a dictionary."
        for key in citation_keys:
            if key not in citation:
                return False, f"Citation at index {i} is missing key: '{key}'"
            if not isinstance(citation[key], str):
                return False, f"Value for '{key}' in citation {i} must be a string."

    # Check confidence value
    valid_confidences = ["low", "mid", "high"]
    if json_obj["confidence"] not in valid_confidences:
        return False, f"'confidence' must be one of {valid_confidences}."

    # Check cannot_answer_reason
    if not (isinstance(json_obj["cannot_answer_reason"], str) or json_obj["cannot_answer_reason"] is None):
        return False, "'cannot_answer_reason' must be a string or null."

    # Cross-validation rules from Plan 0.4
    is_answerable = json_obj["cannot_answer_reason"] is None
    if is_answerable and not json_obj["citations"]:
        return False, "If the question is answerable (cannot_answer_reason is null), 'citations' must not be empty."

    if not is_answerable and not isinstance(json_obj["cannot_answer_reason"], str):
         return False, "If the question is unanswerable, 'cannot_answer_reason' must be a non-empty string."

    return True, "Validation successful."


if __name__ == '__main__':
    # Example usage and demonstration of the frozen components
    print("--- System Prompt ---")
    print(SYSTEM_PROMPT)
    print("\n--- User Prompt Template ---")
    print(USER_PROMPT_TEMPLATE.format(
        question="What is a PDU session?",
        evidence_text="[chunk_id=TS23501-5.6.1-0001 doc_id=TS 23.501 section=5.6.1]\nA PDU Session is a logical connection between the UE and a Data Network."
    ))
    print("\n--- Output JSON Schema ---")
    print(json.dumps(OUTPUT_SCHEMA, indent=2))

    # --- Validation Examples ---
    print("\n--- Validation Tests ---")
    valid_output = {
        "answer": "A PDU Session is a logical connection between a UE and a Data Network.",
        "citations": [
            {
                "doc_id": "TS 23.501",
                "section": "5.6.1",
                "chunk_id": "TS23501-5.6.1-0001",
                "quote": "A PDU Session is a logical connection between the UE and a Data Network."
            }
        ],
        "confidence": "high",
        "cannot_answer_reason": None
    }
    print(f"Valid case: {validate_json_output(valid_output)}")

    invalid_output_no_reason = {
        "answer": "",
        "citations": [],
        "confidence": "low",
        "cannot_answer_reason": None  # Invalid: should have a reason if unanswerable
    }
    print(f"Invalid case (unanswerable but no reason): {validate_json_output(invalid_output_no_reason)}")

    invalid_output_bad_confidence = {
        "answer": "Test",
        "citations": [],
        "confidence": "very high", # Invalid value
        "cannot_answer_reason": "Evidence does not mention this."
    }
    print(f"Invalid case (bad confidence value): {validate_json_output(invalid_output_bad_confidence)}")

