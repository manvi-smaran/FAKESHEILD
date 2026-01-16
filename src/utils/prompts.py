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
