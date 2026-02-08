"""Feature extraction functions for the attention embedding model."""

import re
import numpy as np
from collections import Counter
from src.utils.timer import Timer
from src.preprocessing.simple_scaler import SimpleScaler


def extract_interaction_features(
    prompt_emb: np.ndarray,  # [d_emb]
    response_emb: np.ndarray,  # [d_emb]
    prompt: str,
    response: str,
    timer: Timer | None = None
) -> np.ndarray:  # [6]
    """
    Extract interaction features between prompt and response.
    
    Args:
        prompt_emb: Prompt embedding
        response_emb: Response embedding
        prompt: Prompt text
        response: Response text
        
    Returns:
        Array of 6 interaction features
    """
    features = []
    
    # Embedding-based interactions
    # Cosine similarity
    cos_sim = np.dot(prompt_emb, response_emb) / (
        np.linalg.norm(prompt_emb) * np.linalg.norm(response_emb) + 1e-8
    )
    features.append(cos_sim)
    
    # Euclidean distance
    euclidean_dist = np.linalg.norm(prompt_emb - response_emb)
    features.append(euclidean_dist)
    
    # Dot product
    dot_prod = np.dot(prompt_emb, response_emb)
    features.append(dot_prod)
    
    # Length-based interactions
    # Character length ratio
    length_ratio = len(response) / max(len(prompt), 1)
    features.append(length_ratio)
    
    # Token ratio (approximate by splitting on whitespace)
    prompt_tokens = len(prompt.split())
    response_tokens = len(response.split())
    token_ratio = response_tokens / max(prompt_tokens, 1)
    features.append(token_ratio)
    
    # Characters per token in response
    char_per_token = len(response) / max(response_tokens, 1)
    features.append(char_per_token)
    
    return np.array(features, dtype=np.float32)


def extract_lexical_features(text: str, timer: Timer | None = None) -> np.ndarray:  # [11]
    """
    Extract lexical features from text.
    
    Args:
        text: Input text
        timer: Optional timer for profiling
        
    Returns:
        Array of 11 lexical features
    """
    features = []
    
    # Tokenize into words (simple whitespace-based)
    words = text.split()
    chars = list(text)
    
    if len(words) == 0:
        return np.zeros(11, dtype=np.float32)
    
    # Type-token ratio
    unique_words = set(words)
    type_token_ratio = len(unique_words) / max(len(words), 1)
    features.append(type_token_ratio)
    
    # Hapax legomena ratio (words that appear only once)
    word_counts = Counter(words)
    hapax_count = sum(1 for count in word_counts.values() if count == 1)
    hapax_ratio = hapax_count / max(len(words), 1)
    features.append(hapax_ratio)
    
    # Average and std word length
    word_lengths = [len(w) for w in words]
    avg_word_length = np.mean(word_lengths) if word_lengths else 0
    std_word_length = np.std(word_lengths) if word_lengths else 0
    features.append(avg_word_length)
    features.append(std_word_length)
    
    # Word length distribution (binned)
    def count_words_in_range(words: list[str], min_len: int, max_len: int) -> int:
        return sum(1 for w in words if min_len <= len(w) <= max_len)
    
    word_len_1_3 = count_words_in_range(words, 1, 3) / max(len(words), 1)
    word_len_4_6 = count_words_in_range(words, 4, 6) / max(len(words), 1)
    word_len_7_9 = count_words_in_range(words, 7, 9) / max(len(words), 1)
    word_len_10plus = count_words_in_range(words, 10, 1000) / max(len(words), 1)
    features.extend([word_len_1_3, word_len_4_6, word_len_7_9, word_len_10plus])
    
    # Character-level features
    total_chars = max(len(chars), 1)
    uppercase_ratio = sum(1 for c in chars if c.isupper()) / total_chars
    digit_ratio = sum(1 for c in chars if c.isdigit()) / total_chars
    whitespace_ratio = sum(1 for c in chars if c.isspace()) / total_chars
    features.extend([uppercase_ratio, digit_ratio, whitespace_ratio])
    
    return np.array(features, dtype=np.float32)


def extract_structural_features(text: str, timer: Timer | None = None) -> np.ndarray:  # [15]
    """
    Extract structural features from text.
    
    Args:
        text: Input text
        timer: Optional timer for profiling
        
    Returns:
        Array of 15 structural features
    """
    features = []
    
    # Simple sentence splitting (by .!?)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Paragraph splitting (by double newline)
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Sentence-level features
    num_sentences = len(sentences)
    features.append(num_sentences)
    
    if sentences:
        sentence_lengths = [len(s.split()) for s in sentences]
        avg_sentence_length = np.mean(sentence_lengths)
        std_sentence_length = np.std(sentence_lengths)
        max_sentence_length = max(sentence_lengths)
    else:
        avg_sentence_length = std_sentence_length = max_sentence_length = 0
    
    features.extend([avg_sentence_length, std_sentence_length, max_sentence_length])
    
    # Paragraph-level features
    num_paragraphs = len(paragraphs)
    features.append(num_paragraphs)
    
    if paragraphs:
        paragraph_lengths = [len(p.split()) for p in paragraphs]
        avg_paragraph_length = np.mean(paragraph_lengths)
    else:
        avg_paragraph_length = 0
    
    features.append(avg_paragraph_length)
    
    # Punctuation patterns
    total_chars = max(len(text), 1)
    period_ratio = text.count('.') / total_chars
    comma_ratio = text.count(',') / total_chars
    question_ratio = text.count('?') / total_chars
    exclamation_ratio = text.count('!') / total_chars
    colon_ratio = text.count(':') / total_chars
    semicolon_ratio = text.count(';') / total_chars
    features.extend([
        period_ratio, comma_ratio, question_ratio, 
        exclamation_ratio, colon_ratio, semicolon_ratio
    ])
    
    # Special patterns
    code_block_count = text.count('```')
    bullet_point_count = len(re.findall(r'^\s*[-*•]', text, re.MULTILINE))
    numbered_list_count = len(re.findall(r'^\s*\d+\.', text, re.MULTILINE))
    features.extend([code_block_count, bullet_point_count, numbered_list_count])
    
    return np.array(features, dtype=np.float32)


def extract_all_scalar_features(
    prompt_emb: np.ndarray,  # [d_emb]
    response_emb: np.ndarray,  # [d_emb]
    prompt: str,
    response: str,
    timer: Timer | None = None
) -> np.ndarray:  # [32]
    """
    Extract all scalar features for a (prompt, response) pair.
    
    Args:
        prompt_emb: Prompt embedding
        response_emb: Response embedding
        prompt: Prompt text
        response: Response text
        timer: Optional timer for profiling
        
    Returns:
        Concatenated array of all scalar features (6 + 11 + 15 = 32)
    """
    if timer is not None:
        with Timer("interaction_features", parent=timer):
            interaction = extract_interaction_features(prompt_emb, response_emb, prompt, response, timer)
        with Timer("lexical_features", parent=timer):
            lexical = extract_lexical_features(response, timer)
        with Timer("structural_features", parent=timer):
            structural = extract_structural_features(response, timer)
    else:
        interaction = extract_interaction_features(prompt_emb, response_emb, prompt, response)
        lexical = extract_lexical_features(response)
        structural = extract_structural_features(response)
    
    return np.concatenate([interaction, lexical, structural])


def extract_response_features_for_many(
    prompt_embeddings: np.ndarray,  # [n, d_emb]
    response_embeddings: np.ndarray,  # [n, d_emb]
    prompts: list[str],
    responses: list[str],
    scaler: SimpleScaler | None = None,
    timer: Timer | None = None
) -> tuple[list[np.ndarray], SimpleScaler]:
    """
    Extract all scalar features for a list of (prompt, response) pairs with scaling.
    
    This function follows the same pattern as extract_prompt_features_for_many from
    scoring_feature_extraction.py, handling feature extraction and scaling in one place.
    
    Args:
        prompt_embeddings: Prompt embeddings array
        response_embeddings: Response embeddings array
        prompts: List of prompt texts
        responses: List of response texts
        scaler: Optional pre-fitted scaler (for inference); if None, fits a new one
        timer: Optional timer for profiling
        
    Returns:
        Tuple of (list of feature arrays, fitted scaler)
    """
    features_list = []
    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        features = extract_all_scalar_features(
            prompt_embeddings[i],
            response_embeddings[i],
            prompt,
            response,
            timer=timer
        )
        features_list.append(features)

    features_array = np.stack(features_list)  # [n, n_features]

    if scaler is None:
        scaler = SimpleScaler().fit(features_array)

    scaled_features = scaler.transform(features_array)

    return [scaled_features[i] for i in range(len(scaled_features))], scaler  # [n] x [n_features]

