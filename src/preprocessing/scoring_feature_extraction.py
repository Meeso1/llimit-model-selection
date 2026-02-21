"""Feature extraction functions for prompt embeddings in scoring models."""

from dataclasses import dataclass
import re
import numpy as np
from src.data_models.data_models import EvaluationMessage
from src.preprocessing.simple_scaler import SimpleScaler
from src.utils.timer import Timer


def extract_task_type_features(prompt: str) -> np.ndarray:  # [10], all boolean features
    """
    Detect what type of task the prompt is requesting.
    
    Args:
        prompt: Prompt text
        timer: Optional timer for profiling
        
    Returns:
        Array of 10 task type indicator features
    """
    features = []
    prompt_lower = prompt.lower()
    
    # Code-related indicators
    code_keywords = ['code', 'function', 'implement', 'debug', 'python', 'javascript', 
                     'java', 'c++', 'algorithm', 'class', 'def ', 'import ']
    has_code_request = any(kw in prompt_lower for kw in code_keywords)
    has_code_block = '```' in prompt or '`' in prompt
    features.extend([float(has_code_request), float(has_code_block)])
    
    # Math/reasoning indicators  
    math_keywords = ['calculate', 'solve', 'equation', 'proof', 'derive', 'probability']
    has_math = any(kw in prompt_lower for kw in math_keywords)
    has_numbers = bool(re.search(r'\d+', prompt))
    has_math_symbols = bool(re.search(r'[+\-*/=^√∫∑]', prompt))
    features.extend([float(has_math), float(has_numbers), float(has_math_symbols)])
    
    # Creative writing indicators
    creative_keywords = ['write', 'story', 'poem', 'creative', 'imagine', 'fiction']
    has_creative = any(kw in prompt_lower for kw in creative_keywords)
    features.append(float(has_creative))
    
    # Factual/knowledge indicators
    factual_keywords = ['what is', 'who is', 'when did', 'where is', 'explain', 'define']
    has_factual = any(kw in prompt_lower for kw in factual_keywords)
    features.append(float(has_factual))
    
    # Instruction-following indicators
    instruction_keywords = ['step by step', 'list', 'summarize', 'translate', 'convert']
    has_instruction = any(kw in prompt_lower for kw in instruction_keywords)
    features.append(float(has_instruction))
    
    # Roleplay/persona indicators
    roleplay_keywords = ['act as', 'pretend', 'roleplay', 'you are a', 'imagine you']
    has_roleplay = any(kw in prompt_lower for kw in roleplay_keywords)
    features.append(float(has_roleplay))
    
    # Analysis/reasoning
    analysis_keywords = ['compare', 'analyze', 'evaluate', 'pros and cons', 'difference']
    has_analysis = any(kw in prompt_lower for kw in analysis_keywords)
    features.append(float(has_analysis))
    
    return np.array(features, dtype=np.float32)


def extract_prompt_complexity_features(prompt: str) -> np.ndarray:  # [8], all float features
    """
    Extract features indicating prompt complexity.
    
    Args:
        prompt: Prompt text
        timer: Optional timer for profiling
        
    Returns:
        Array of 8 complexity features
    """
    features = []
    words = prompt.split()
    sentences = re.split(r'[.!?]+', prompt)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Length features
    features.append(float(len(prompt)))  # char count
    features.append(float(len(words)))   # word count
    features.append(float(len(sentences)))  # sentence count
    
    # Vocabulary complexity
    unique_words = set(w.lower() for w in words)
    type_token_ratio = len(unique_words) / max(len(words), 1)
    inverse_type_token_ratio = 1 - type_token_ratio # TODO: handle backwards compatibility with changes to features
    features.append(inverse_type_token_ratio)
    
    # Average word length (longer words = more technical)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
    features.append(avg_word_len)
    
    # Question complexity (number of question marks / total characters)
    features.append(float(prompt.count('?')) / max(len(prompt), 1))
    
    # Nested structure indicators (parentheses, quotes)
    nesting_depth = prompt.count('(') + prompt.count('[') + prompt.count('{')
    features.append(float(nesting_depth) / max(len(prompt), 1))
    
    # Multi-part request (numbered items, "and", "also")
    multi_part_indicators = len(re.findall(r'\d+\.|\band\b|\balso\b', prompt.lower()))
    features.append(float(multi_part_indicators) / max(len(prompt), 1))
    
    return np.array(features, dtype=np.float32)


def extract_domain_features(prompt: str) -> np.ndarray:  # [12], all float features
    """
    Extract domain-specific indicators.
    
    Args:
        prompt: Prompt text
        timer: Optional timer for profiling
        
    Returns:
        Array of 12 domain indicator features
    """
    prompt_lower = prompt.lower()
    
    domains = {
        'science': ['physics', 'chemistry', 'biology', 'quantum', 'molecule', 'cell'],
        'medicine': ['medical', 'health', 'symptom', 'disease', 'treatment', 'doctor'],
        'law': ['legal', 'law', 'court', 'contract', 'rights', 'liability'],
        'finance': ['money', 'invest', 'stock', 'finance', 'bank', 'economy'],
        'tech': ['software', 'hardware', 'api', 'database', 'server', 'cloud'],
        'academic': ['research', 'paper', 'study', 'hypothesis', 'methodology'],
        'casual': ['hey', 'hi', 'thanks', 'please', 'help me', 'can you'],
        'formal': ['hereby', 'pursuant', 'regarding', 'therefore', 'furthermore'],
        'philosophical': ['meaning', 'existence', 'consciousness', 'ethics', 'moral'],
        'historical': ['history', 'century', 'war', 'ancient', 'revolution'],
        'personal': ['my', 'i am', 'i have', 'i want', 'i need', 'advice'],
        'business': ['company', 'startup', 'marketing', 'revenue', 'customer'],
    }
    
    features = []
    for _, keywords in domains.items():
        score = sum(1 for kw in keywords if kw in prompt_lower) / len(keywords) / max(len(prompt), 1)
        features.append(score)
    
    return np.array(features, dtype=np.float32)


def extract_prompt_style_features(prompt: str) -> tuple[np.ndarray, np.ndarray]:  # ([4] (numeric), [2] (boolean)) 
    """
    Extract linguistic style features from prompt.
    
    Args:
        prompt: Prompt text
        timer: Optional timer for profiling
        
    Returns:
        Array of 6 style features
    """
    numeric_features = []
    boolean_features = []
    prompt_lower = prompt.lower()
    
    # Formality indicators
    informal_markers = ['lol', 'btw', 'idk', 'gonna', 'wanna', 'u ', 'ur ']
    informal_score = sum(1 for m in informal_markers if m in prompt_lower)
    numeric_features.append(float(informal_score) / max(len(prompt), 1))
    
    # Imperative vs interrogative
    starts_with_verb = bool(re.match(r'^(write|create|make|give|tell|explain|show)', prompt_lower))
    boolean_features.append(float(starts_with_verb))
    
    is_question = prompt.strip().endswith('?')
    boolean_features.append(float(is_question))
    
    # Specificity (presence of specific constraints)
    constraint_words = ['exactly', 'must', 'should', 'only', 'limit', 'maximum', 'minimum']
    specificity = sum(1 for w in constraint_words if w in prompt_lower)
    numeric_features.append(float(specificity) / max(len(prompt), 1))
    
    # Politeness markers
    polite_words = ['please', 'thank', 'could you', 'would you', 'kindly']
    politeness = sum(1 for w in polite_words if w in prompt_lower)
    numeric_features.append(float(politeness) / max(len(prompt), 1))
    
    # Urgency markers
    urgent_words = ['urgent', 'asap', 'immediately', 'quick', 'fast', 'now']
    urgency = sum(1 for w in urgent_words if w in prompt_lower)
    numeric_features.append(float(urgency) / max(len(prompt), 1))
    
    return np.array(numeric_features, dtype=np.float32), np.array(boolean_features, dtype=np.float32)


def extract_context_features(
    prompt: str, 
    conversation_history: list[EvaluationMessage]
) -> tuple[np.ndarray, np.ndarray]:  # ([3] (numeric), [1] (boolean))
    """
    Extract features from conversation context.
    
    Args:
        prompt: Current prompt text
        conversation_history: List of previous messages in the conversation
        timer: Optional timer for profiling
        
    Returns:
        Array of 4 context features
    """
    numeric_features = []
    boolean_features = []
    
    # Conversation turn count
    numeric_features.append(float(len(conversation_history)))
    
    # Is this a follow-up? (references to previous context)
    followup_markers = ['that', 'this', 'it', 'the above', 'previous', 'earlier', 'you said']
    is_followup = any(m in prompt.lower() for m in followup_markers)
    boolean_features.append(float(is_followup))
    
    # Total context length (affects model performance)
    total_context_chars = sum(len(m.content) for m in conversation_history)
    numeric_features.append(float(total_context_chars))
    
    # Assistant response count (how many turns of dialogue)
    assistant_turns = sum(1 for m in conversation_history if m.role == 'assistant')
    numeric_features.append(float(assistant_turns))
    
    return np.array(numeric_features, dtype=np.float32), np.array(boolean_features, dtype=np.float32)


def extract_output_format_features(prompt: str) -> np.ndarray:  # [5], all boolean features
    """
    Detect expected output format.
    
    Args:
        prompt: Prompt text
        timer: Optional timer for profiling
        
    Returns:
        Array of 5 output format indicator features
    """
    prompt_lower = prompt.lower()
    
    expects_list = bool(re.search(r'list|bullet|enumerate|\d+ (things|ways|reasons)', prompt_lower))
    expects_table = 'table' in prompt_lower or 'csv' in prompt_lower
    expects_json = 'json' in prompt_lower or 'format:' in prompt_lower
    expects_code = '```' in prompt or 'code' in prompt_lower or 'script' in prompt_lower
    expects_long = any(w in prompt_lower for w in ['detailed', 'comprehensive', 'thorough', 'in-depth'])
    
    return np.array([
        float(expects_list),
        float(expects_table),
        float(expects_json),
        float(expects_code),
        float(expects_long),
    ], dtype=np.float32)


def extract_all_prompt_features(
    prompt: str,
    conversation_history: list[EvaluationMessage],
    timer: Timer | None = None
) -> tuple[np.ndarray, np.ndarray]:  # ([27] (numeric), [18] (boolean))
    """
    Extract all scalar features for a prompt.
    
    This function combines all feature extraction functions:
    - Task type indicators (10 features)
    - Prompt complexity (8 features)
    - Domain indicators (12 features)
    - Linguistic style (6 features)
    - Context features (4 features)
    - Output format (5 features)
    
    Total: 45 features
    
    Args:
        prompt: Prompt text
        conversation_history: List of previous messages in the conversation
        timer: Optional timer for profiling
        
    Returns:
        Concatenated array of all prompt features [45]
    """    
    with Timer("task_type_features", parent=timer):
        task_type_boolean = extract_task_type_features(prompt)
    with Timer("complexity_features", parent=timer):
        complexity_numeric = extract_prompt_complexity_features(prompt)
    with Timer("domain_features", parent=timer):
        domain_numeric = extract_domain_features(prompt)
    with Timer("style_features", parent=timer):
        style_numeric, style_boolean = extract_prompt_style_features(prompt)
    with Timer("context_features", parent=timer):
        context_numeric, context_boolean = extract_context_features(prompt, conversation_history)
    with Timer("output_format_features", parent=timer):
        output_format_boolean = extract_output_format_features(prompt)
    
    return np.concatenate([complexity_numeric, domain_numeric, style_numeric, context_numeric]), \
        np.concatenate([task_type_boolean, style_boolean, context_boolean, output_format_boolean])


def extract_and_transform_all_prompt_features(
    prompts: list[str],
    conversation_histories: list[list[EvaluationMessage]],
    scaler: SimpleScaler | None = None,
    timer: Timer | None = None
) -> tuple[list[np.ndarray], SimpleScaler]:
    """
    Extract all scalar features for a list of prompts, transforming and scaling them.
    Output from this function can be used as input for a model.
    """
    numeric_features = []
    boolean_features = []
    for prompt, conversation_history in zip(prompts, conversation_histories):
        numeric_feature, boolean_feature = extract_all_prompt_features(prompt, conversation_history, timer)
        numeric_features.append(numeric_feature)
        boolean_features.append(boolean_feature)

    numeric_features = np.stack(numeric_features)
    boolean_features = np.stack(boolean_features)

    numeric_feature_descriptions, _ = get_feature_descriptions()
    scaling_data: list[np.ndarray] = []
    non_zero_indexes_per_feature: list[np.ndarray] = []
    for index, feature_description in enumerate(numeric_feature_descriptions):
        values = numeric_features[:, index]
        non_zero_indexes = np.nonzero(values)[0]
        non_zero_indexes_per_feature.append(non_zero_indexes)

        if feature_description.logarythmic:
            # Clip values to 1e-6 to avoid log(0)
            values = np.log(np.where(values > 1e-6, values, 1e-6))
        if feature_description.many_zeros:
            # If the feature has many zeros, we only want to scale the non-zero values. Zeros will be treated separately.
            values = values[non_zero_indexes]
        scaling_data.append(values)

    if scaler is None:
        scaler = SimpleScaler().fit_unbalanced(scaling_data)

    scaled_data = scaler.transform_unbalanced(scaling_data)
    for index, feature_description in enumerate(numeric_feature_descriptions):
        if feature_description.many_zeros:
            # Use softplus for non-zero values, setting zeros to 0
            non_zero_indexes = non_zero_indexes_per_feature[index]
            numeric_features[:, index] = 0
            numeric_features[non_zero_indexes, index] = np.log(1 + np.exp(scaled_data[index]))
        else:
            numeric_features[:, index] = scaled_data[index]

    return [
        np.concatenate([numeric_feature, boolean_feature]) \
        for numeric_feature, boolean_feature in zip(numeric_features, boolean_features) \
    ], scaler  # [n_prompts, n_features]


def unscale_prompt_features(
    prompt_features: list[np.ndarray],
    scaler: SimpleScaler
) -> list[np.ndarray]:
    """
    Unscale numeric prompt features, leaving boolean features as is.
    """
    features_array = np.stack(prompt_features)  # [n_samples, n_features]
    
    # Numeric features are the first 27, boolean are the remaining 18
    numeric_dim = 27
    numeric_part = features_array[:, :numeric_dim]  # [n_samples, 27]
    
    unscaled_numeric = scaler.inverse_transform(numeric_part)  # [n_samples, 27]
    boolean_part = features_array[:, numeric_dim:]  # [n_samples, 18]
    
    unscaled_features = np.concatenate([unscaled_numeric, boolean_part], axis=1)  # [n_samples, 45]
    
    return [unscaled_features[i] for i in range(len(prompt_features))]


def get_feature_descriptions() -> tuple[list["NumericFeatureDescription"], list["BooleanFeatureDescription"]]:
    """
    Get the names of all prompt features, in order of appearance.
    """
    return [
        # Complexity (8)
        NumericFeatureDescription(name="Character count", logarythmic=True),
        NumericFeatureDescription(name="Word count", logarythmic=True),
        NumericFeatureDescription(name="Sentence count", logarythmic=True),
        NumericFeatureDescription(name="Inverse Type-token ratio", many_zeros=True),
        NumericFeatureDescription(name="Average word length", logarythmic=True),
        NumericFeatureDescription(name="Question density", logarythmic=True, many_zeros=True),
        NumericFeatureDescription(name="Nesting depth", logarythmic=True, many_zeros=True),
        NumericFeatureDescription(name="Multi-part request density", logarythmic=True, many_zeros=True),
        # Domain (12)
        NumericFeatureDescription(name="Science domain score", many_zeros=True, logarythmic=True),
        NumericFeatureDescription(name="Medicine domain score", many_zeros=True, logarythmic=True),
        NumericFeatureDescription(name="Law domain score", many_zeros=True, logarythmic=True),
        NumericFeatureDescription(name="Finance domain score", many_zeros=True, logarythmic=True),
        NumericFeatureDescription(name="Tech domain score", many_zeros=True, logarythmic=True),
        NumericFeatureDescription(name="Academic domain score", many_zeros=True, logarythmic=True),
        NumericFeatureDescription(name="Casual domain score", many_zeros=True, logarythmic=True),
        NumericFeatureDescription(name="Formal domain score", many_zeros=True, logarythmic=True),
        NumericFeatureDescription(name="Philosophical domain score", many_zeros=True, logarythmic=True),
        NumericFeatureDescription(name="Historical domain score", many_zeros=True, logarythmic=True),
        NumericFeatureDescription(name="Personal domain score", many_zeros=True, logarythmic=True),
        NumericFeatureDescription(name="Business domain score", many_zeros=True, logarythmic=True),
        # Style numeric (4)
        NumericFeatureDescription(name="Informal marker density", logarythmic=True, many_zeros=True),
        NumericFeatureDescription(name="Specificity density", logarythmic=True, many_zeros=True),
        NumericFeatureDescription(name="Politeness density", logarythmic=True, many_zeros=True),
        NumericFeatureDescription(name="Urgency density", logarythmic=True, many_zeros=True),
        # Context numeric (3)
        NumericFeatureDescription(name="Conversation turn count", logarythmic=True, many_zeros=True),
        NumericFeatureDescription(name="Total context length", logarythmic=True, many_zeros=True),
        NumericFeatureDescription(name="Assistant turn count", logarythmic=True, many_zeros=True),
    ], \
    [
        # Task type (10)
        BooleanFeatureDescription(name="Has code request"),
        BooleanFeatureDescription(name="Has code block"),
        BooleanFeatureDescription(name="Has math"),
        BooleanFeatureDescription(name="Has numbers"),
        BooleanFeatureDescription(name="Has math symbols"),
        BooleanFeatureDescription(name="Has creative writing"),
        BooleanFeatureDescription(name="Has factual query"),
        BooleanFeatureDescription(name="Has instruction following"),
        BooleanFeatureDescription(name="Has roleplay"),
        BooleanFeatureDescription(name="Has analysis"),
        # Style boolean (2)
        BooleanFeatureDescription(name="Starts with verb"),
        BooleanFeatureDescription(name="Is question"),
        # Context boolean (1)
        BooleanFeatureDescription(name="Is follow-up"),
        # Output format (5)
        BooleanFeatureDescription(name="Expects list"),
        BooleanFeatureDescription(name="Expects table"),
        BooleanFeatureDescription(name="Expects JSON"),
        BooleanFeatureDescription(name="Expects code"),
        BooleanFeatureDescription(name="Expects long response"),
    ]


@dataclass
class NumericFeatureDescription:
    name: str
    logarythmic: bool = False
    many_zeros: bool = False

@dataclass
class BooleanFeatureDescription:
    name: str
