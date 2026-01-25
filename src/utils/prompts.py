ZERO_SHOT_BINARY = """Analyze this image of a human face carefully.

Question: Is this a real photograph of a person, or is it a deepfake/AI-manipulated image?

Provide your answer in this exact format:
VERDICT: [Real or Fake]
CONFIDENCE: [High, Medium, or Low]
REASONING: [Brief explanation of visual cues that led to your decision]"""


ZERO_SHOT_DETECTION = """You are an expert forensic analyst specializing in detecting manipulated images and deepfakes.

Examine this facial image for signs of digital manipulation or AI generation. Look for:
- Unnatural skin texture or lighting
- Irregular facial boundaries or blending artifacts
- Asymmetric or distorted facial features
- Inconsistent background or context
- Compression artifacts in unexpected areas

Is this image authentic or manipulated?

Answer format:
VERDICT: [Authentic/Manipulated]
REASONING: [Your analysis]"""


ZERO_SHOT_SIMPLE = """Is this image of a real human face or a deepfake? Answer 'Real' or 'Fake' with a brief explanation."""


FEW_SHOT_TEMPLATE = """You are analyzing facial images to detect deepfakes. Here are some examples:

{examples}

Now analyze this new image:
Is this a real photograph or a deepfake? Answer 'Real' or 'Fake' with your reasoning."""


FEW_SHOT_EXAMPLE_REAL = """Example {idx}: This is a REAL image.
Characteristics: Natural skin texture, consistent lighting across facial features, clear and sharp facial boundaries, symmetric facial features, no visible blending artifacts."""


FEW_SHOT_EXAMPLE_FAKE = """Example {idx}: This is a FAKE (deepfake) image.
Characteristics: Signs of manipulation include subtle blending around facial edges, slight inconsistencies in lighting or texture, possible artifacts near eyes/mouth/hairline."""


COT_PROMPT = """Analyze this facial image step by step to determine if it's real or a deepfake:

Step 1: Examine the overall face structure and symmetry
Step 2: Check skin texture and lighting consistency
Step 3: Inspect facial boundaries (hairline, jaw, ears)
Step 4: Look at the eyes and mouth for artifacts
Step 5: Assess the background for inconsistencies

Based on your analysis:
VERDICT: [Real or Fake]
CONFIDENCE: [0-100%]
KEY_EVIDENCE: [Most important indicator]"""


def get_prompt(prompt_type: str = "zero_shot_binary") -> str:
    prompts = {
        "zero_shot_binary": ZERO_SHOT_BINARY,
        "zero_shot_detection": ZERO_SHOT_DETECTION,
        "zero_shot_simple": ZERO_SHOT_SIMPLE,
        "chain_of_thought": COT_PROMPT,
    }
    return prompts.get(prompt_type, ZERO_SHOT_BINARY)


def build_few_shot_prompt(examples: list, k: int = 4) -> str:
    example_texts = []
    
    for idx, (image, label, _) in enumerate(examples[:k], 1):
        if label == 0:
            example_texts.append(FEW_SHOT_EXAMPLE_REAL.format(idx=idx))
        else:
            example_texts.append(FEW_SHOT_EXAMPLE_FAKE.format(idx=idx))
    
    examples_str = "\n\n".join(example_texts)
    return FEW_SHOT_TEMPLATE.format(examples=examples_str)


# ==================== JSON-BASED FEW-SHOT PROMPTS ====================

JSON_FEW_SHOT_SYSTEM = """You are a forensic expert in deepfake detection. Analyze facial images for signs of digital manipulation.

FORENSIC CHECKLIST:
1. Facial boundaries (hairline, jaw, ears)
2. Skin texture and pore consistency
3. Eye/mouth artifact detection
4. Lighting coherence across face
5. Compression pattern anomalies

You must output ONLY valid JSON in this exact format:
{{"label":"real|fake","p_fake":0.0-1.0,"evidence":["reason1","reason2"]}}"""


JSON_FEW_SHOT_EXAMPLE_REAL = """Image {idx}: REAL
{{"label":"real","p_fake":0.08,"evidence":["consistent facial boundaries","natural skin texture","coherent lighting"]}}"""


JSON_FEW_SHOT_EXAMPLE_FAKE = """Image {idx}: FAKE  
{{"label":"fake","p_fake":0.91,"evidence":["blending artifacts at jaw","inconsistent skin texture","unnatural eye reflections"]}}"""


JSON_FEW_SHOT_QUERY = """Now analyze the new image. Output ONLY the JSON:"""


def build_json_few_shot_prompt(num_real: int, num_fake: int) -> str:
    """Build JSON-based few-shot prompt with example count."""
    examples = []
    
    idx = 1
    for _ in range(num_real):
        examples.append(JSON_FEW_SHOT_EXAMPLE_REAL.format(idx=idx))
        idx += 1
    
    for _ in range(num_fake):
        examples.append(JSON_FEW_SHOT_EXAMPLE_FAKE.format(idx=idx))
        idx += 1
    
    prompt = f"""{JSON_FEW_SHOT_SYSTEM}

EXAMPLES (with images shown):
{chr(10).join(examples)}

{JSON_FEW_SHOT_QUERY}"""
    
    return prompt


def parse_json_response(response: str) -> dict:
    """
    Parse JSON response from model, with robust fallback handling.
    
    Returns:
        dict with keys:
        - label: "real", "fake", or "unknown"
        - p_fake: float 0.0-1.0, or None if parsing failed
        - evidence: list of strings
        - parse_success: bool indicating if JSON was parsed successfully
    """
    import re
    import json
    
    # Strip code fences first (```json ... ``` or ``` ... ```)
    cleaned_response = response
    code_fence_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    code_match = re.search(code_fence_pattern, response)
    if code_match:
        cleaned_response = code_match.group(1)
    
    # Try to extract JSON from response (handle nested braces with greedy match)
    json_patterns = [
        r'\{[^{}]*"label"[^{}]*"p_fake"[^{}]*\}',  # Simple object with required fields
        r'\{[^{}]*\}',  # Any simple JSON object
    ]
    
    for pattern in json_patterns:
        json_match = re.search(pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
        
        if json_match:
            try:
                result = json.loads(json_match.group())
                # Validate required fields
                if "label" in result and "p_fake" in result:
                    p_fake = float(result.get("p_fake", 0.5))
                    # Clamp p_fake to valid range
                    p_fake = max(0.0, min(1.0, p_fake))
                    return {
                        "label": str(result.get("label", "unknown")).lower().strip(),
                        "p_fake": p_fake,
                        "evidence": result.get("evidence", []),
                        "parse_success": True,
                    }
            except (json.JSONDecodeError, ValueError, TypeError):
                continue
    
    # Fallback: try to extract from text (less reliable)
    # Use boundary-aware regex for proper negation handling
    response_lower = response.lower()
    
    # Negation patterns (check first)
    not_fake_pattern = r'\b(not\s+(a\s+)?fake|isn\'?t\s+(a\s+)?fake)\b'
    not_real_pattern = r'\b(not\s+(a\s+)?real|isn\'?t\s+real)\b'
    
    if re.search(not_fake_pattern, response_lower):
        return {
            "label": "real", 
            "p_fake": None,
            "evidence": ["'not fake' detected in text"], 
            "parse_success": False
        }
    if re.search(not_real_pattern, response_lower):
        return {
            "label": "fake", 
            "p_fake": None,
            "evidence": ["'not real' detected in text"], 
            "parse_success": False
        }
    
    # Positive patterns with word boundaries
    if re.search(r'\b(fake|deepfake|manipulated)\b', response_lower):
        return {
            "label": "fake", 
            "p_fake": None,
            "evidence": ["extracted from text, no JSON"], 
            "parse_success": False
        }
    if re.search(r'\b(real|authentic|genuine)\b', response_lower):
        return {
            "label": "real", 
            "p_fake": None,
            "evidence": ["extracted from text, no JSON"], 
            "parse_success": False
        }
    
    # Complete parse failure - return unknown with pred=-1 equivalent
    return {
        "label": "unknown", 
        "p_fake": None, 
        "evidence": ["parsing failed completely"], 
        "parse_success": False
    }


