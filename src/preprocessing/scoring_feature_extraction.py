"""Feature extraction functions for prompt embeddings in scoring models."""

import re
import numpy as np
from src.data_models.data_models import EvaluationMessage
from src.utils.timer import Timer


def extract_task_type_features(prompt: str) -> np.ndarray:  # [10]
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


def extract_prompt_complexity_features(prompt: str) -> np.ndarray:  # [8]
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
    features.append(type_token_ratio)
    
    # Average word length (longer words = more technical)
    avg_word_len = np.mean([len(w) for w in words]) if words else 0.0
    features.append(avg_word_len)
    
    # Question complexity (number of question marks)
    features.append(float(prompt.count('?')))
    
    # Nested structure indicators (parentheses, quotes)
    nesting_depth = prompt.count('(') + prompt.count('[') + prompt.count('{')
    features.append(float(nesting_depth))
    
    # Multi-part request (numbered items, "and", "also")
    multi_part_indicators = len(re.findall(r'\d+\.|\band\b|\balso\b', prompt.lower()))
    features.append(float(multi_part_indicators))
    
    return np.array(features, dtype=np.float32)


def extract_domain_features(prompt: str) -> np.ndarray:  # [12]
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
    for domain, keywords in domains.items():
        score = sum(1 for kw in keywords if kw in prompt_lower) / len(keywords)
        features.append(score)
    
    return np.array(features, dtype=np.float32)


def extract_prompt_style_features(prompt: str, timer: Timer | None = None) -> np.ndarray:  # [6]
    """
    Extract linguistic style features from prompt.
    
    Args:
        prompt: Prompt text
        timer: Optional timer for profiling
        
    Returns:
        Array of 6 style features
    """
    features = []
    prompt_lower = prompt.lower()
    
    # Formality indicators
    informal_markers = ['lol', 'btw', 'idk', 'gonna', 'wanna', 'u ', 'ur ']
    informal_score = sum(1 for m in informal_markers if m in prompt_lower)
    features.append(float(informal_score))
    
    # Imperative vs interrogative
    starts_with_verb = bool(re.match(r'^(write|create|make|give|tell|explain|show)', prompt_lower))
    features.append(float(starts_with_verb))
    
    is_question = prompt.strip().endswith('?')
    features.append(float(is_question))
    
    # Specificity (presence of specific constraints)
    constraint_words = ['exactly', 'must', 'should', 'only', 'limit', 'maximum', 'minimum']
    specificity = sum(1 for w in constraint_words if w in prompt_lower)
    features.append(float(specificity))
    
    # Politeness markers
    polite_words = ['please', 'thank', 'could you', 'would you', 'kindly']
    politeness = sum(1 for w in polite_words if w in prompt_lower)
    features.append(float(politeness))
    
    # Urgency markers
    urgent_words = ['urgent', 'asap', 'immediately', 'quick', 'fast', 'now']
    urgency = sum(1 for w in urgent_words if w in prompt_lower)
    features.append(float(urgency))
    
    return np.array(features, dtype=np.float32)


def extract_context_features(
    prompt: str, 
    conversation_history: list[EvaluationMessage]
) -> np.ndarray:  # [4]
    """
    Extract features from conversation context.
    
    Args:
        prompt: Current prompt text
        conversation_history: List of previous messages in the conversation
        timer: Optional timer for profiling
        
    Returns:
        Array of 4 context features
    """
    features = []
    
    # Conversation turn count
    features.append(float(len(conversation_history)))
    
    # Is this a follow-up? (references to previous context)
    followup_markers = ['that', 'this', 'it', 'the above', 'previous', 'earlier', 'you said']
    is_followup = any(m in prompt.lower() for m in followup_markers)
    features.append(float(is_followup))
    
    # Total context length (affects model performance)
    total_context_chars = sum(len(m.content) for m in conversation_history)
    features.append(float(total_context_chars))
    
    # Assistant response count (how many turns of dialogue)
    assistant_turns = sum(1 for m in conversation_history if m.role == 'assistant')
    features.append(float(assistant_turns))
    
    return np.array(features, dtype=np.float32)


def extract_output_format_features(prompt: str, timer: Timer | None = None) -> np.ndarray:  # [5]
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
) -> np.ndarray:  # [45]
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
        task_type = extract_task_type_features(prompt)
    with Timer("complexity_features", parent=timer):
        complexity = extract_prompt_complexity_features(prompt)
    with Timer("domain_features", parent=timer):
        domain = extract_domain_features(prompt)
    with Timer("style_features", parent=timer):
        style = extract_prompt_style_features(prompt)
    with Timer("context_features", parent=timer):
        context = extract_context_features(prompt, conversation_history)
    with Timer("output_format_features", parent=timer):
        output_format = extract_output_format_features(prompt)
    
    return np.concatenate([task_type, complexity, domain, style, context, output_format])

