import pandas as pd
import numpy as np
import warnings
from src.data_models.data_models import (
    TrainingData, 
    EvaluationEntry, 
    EvaluationMessage,
    CategoryTag,
    CreativeWritingTag,
    CriteriaTag,
    IfTag,
    MathTag,
)


def _parse_category_tag(category_tag_dict: dict) -> CategoryTag:
    """
    Parses a category_tag dictionary into a CategoryTag dataclass.
    Expects all required fields to be present.
    """
    creative_writing_data = category_tag_dict["creative_writing_v0.1"]
    creative_writing = CreativeWritingTag(
        creative_writing=creative_writing_data["creative_writing"],
        score=creative_writing_data["score"]
    )
    
    criteria_data = category_tag_dict["criteria_v0.1"]
    criteria = CriteriaTag(
        complexity=criteria_data["complexity"],
        creativity=criteria_data["creativity"],
        domain_knowledge=criteria_data["domain_knowledge"],
        problem_solving=criteria_data["problem_solving"],
        real_world=criteria_data["real_world"],
        specificity=criteria_data["specificity"],
        technical_accuracy=criteria_data["technical_accuracy"]
    )
    
    if_data = category_tag_dict["if_v0.1"]
    if_tag = IfTag(
        if_=if_data["if"],
        score=if_data["score"]
    )
    
    math_data = category_tag_dict["math_v0.1"]
    math_tag = MathTag(
        math=math_data["math"]
    )
    
    return CategoryTag(
        creative_writing_v0_1=creative_writing,
        criteria_v0_1=criteria,
        if_v0_1=if_tag,
        math_v0_1=math_tag
    )


def _extract_text(content: np.ndarray | str, row_id: str) -> str:
    """
    Extracts text from a message content field.
    Content is a numpy array containing a list of dicts with 'type'='text' in lmarena-human-preference-140k,
    or a string in chatbot_arena.
    """
    if isinstance(content, str):
        return content

    content_list = content.tolist()
        
    text_parts = []
    for i, item in enumerate(content_list):
        if not isinstance(item, dict):
            raise ValueError(f"Row {row_id}: Content item {i} is not a dict, got '{type(item).__name__}'")
        
        item_type = item.get("type")
        if item_type == "text":
            if "text" not in item:
                raise ValueError(f"Row {row_id}: Content item {i} has type='text' but missing 'text' field")
            text_parts.append(item["text"])
        elif item_type is None:
            raise ValueError(f"Row {row_id}: Content item {i} missing 'type' field")
                
    return "".join(text_parts)


def _parse_conversation(conversation: np.ndarray, row_id: str) -> tuple[list[EvaluationMessage], str, str]:
    """
    Parses a conversation into history, user prompt, and model response.
    Conversation is always a numpy array of message dicts.
    Returns: (history, user_prompt, model_response)
    """
    conv_list = conversation.tolist()

    if len(conv_list) < 2:
        raise ValueError(f"Row {row_id}: Conversation must have at least 2 messages, got {len(conv_list)}")
        
    last_msg = conv_list[-1]
    if last_msg.get("role") != "assistant":
        raise ValueError(f"Row {row_id}: Expected last message role to be 'assistant', got '{last_msg.get('role')}'")
        
    response = _extract_text(last_msg.get("content"), row_id)
    
    user_msg = conv_list[-2]
    if user_msg.get("role") != "user":
        raise ValueError(f"Row {row_id}: Expected second to last message role to be 'user', got '{user_msg.get('role')}'")
        
    user_prompt = _extract_text(user_msg.get("content"), row_id)
    
    history_msgs = conv_list[:-2]
    history = []
    for i, msg in enumerate(history_msgs):
        role = msg.get("role")
        if not role:
            raise ValueError(f"Row {row_id}: History message {i} missing 'role' field")
        
        if role not in ["user", "assistant"]:
            raise ValueError(f"Row {row_id}: History message {i} has invalid role '{role}'")
             
        content = _extract_text(msg.get("content"), row_id)
        history.append(EvaluationMessage(role=role, content=content))
        
    return history, user_prompt, response


def load_training_data_lmarena(df: pd.DataFrame) -> TrainingData:
    """
    Converts a Pandas DataFrame from the lmarena-human-preference-140k dataset
    into a TrainingData instance.
    
    Rows with errors are skipped with a warning.
    """
    required_columns = [
        "model_a", "model_b", "winner", 
        "evaluation_session_id", "evaluation_order", 
        "conversation_a", "conversation_b", "timestamp"
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    entries = []
    skipped_count = 0
    
    for idx, row in df.iterrows():
        row_id = f"idx={idx}, session={row.get('evaluation_session_id', 'unknown')}"
        
        try:
            history_a, prompt_a, response_a = _parse_conversation(row["conversation_a"], row_id)
            _, _, response_b = _parse_conversation(row["conversation_b"], row_id)
            
            winner = row["winner"]
            if winner not in ["model_a", "model_b", "tie", "both_bad"]:
                raise ValueError(f"Row {row_id}: Invalid winner value '{winner}'")
            
            category_tag = None
            if "category_tag" in row and pd.notna(row["category_tag"]):
                try:
                    category_tag = _parse_category_tag(row["category_tag"])
                except (KeyError, TypeError) as e:
                    warnings.warn(f"Row {row_id}: Failed to parse category_tag: {str(e)}", UserWarning)
            
            entry = EvaluationEntry(
                model_a=row["model_a"],
                model_b=row["model_b"],
                winner=winner,
                evaluation_session_id=row["evaluation_session_id"],
                evaluation_order=int(row["evaluation_order"]),
                conversation_history=history_a,
                user_prompt=prompt_a,
                model_a_response=response_a,
                model_b_response=response_b,
                timestamp=str(row["timestamp"]),
                category_tag=category_tag
            )
            entries.append(entry)
            
        except (ValueError, KeyError, TypeError) as e:
            skipped_count += 1
            warnings.warn(f"Skipping row: {str(e)}", UserWarning)
            continue
    
    if skipped_count > 0:
        warnings.warn(f"Skipped {skipped_count} rows due to errors", UserWarning)
            
    return TrainingData(entries=entries)


def load_training_data_chatbot_arena(df: pd.DataFrame) -> TrainingData:
    """
    Converts a Pandas DataFrame from the chatbot_arena dataset
    into a TrainingData instance.
    """
    required_columns = [
        "model_a", "model_b", "winner", 
        "question_id", "turn", 
        "conversation_a", "conversation_b", "tstamp"
    ]
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    entries = []
    skipped_count = 0
    
    for idx, row in df.iterrows():
        row_id = f"idx={idx}, session={row.get('question_id', 'unknown')}"
        
        try:
            history_a, prompt_a, response_a = _parse_conversation(row["conversation_a"], row_id)
            _, _, response_b = _parse_conversation(row["conversation_b"], row_id)
            
            winner = row["winner"]
            if winner not in ["model_a", "model_b", "tie", "tie (bothbad)"]:
                raise ValueError(f"Row {row_id}: Invalid winner value '{winner}'")

            if winner == "tie (bothbad)":
                winner = "both_bad"
            
            entry = EvaluationEntry(
                model_a=row["model_a"],
                model_b=row["model_b"],
                winner=winner,
                evaluation_session_id=row["question_id"],
                evaluation_order=int(row["turn"]),
                conversation_history=history_a,
                user_prompt=prompt_a,
                model_a_response=response_a,
                model_b_response=response_b,
                timestamp=str(row["tstamp"]),
                category_tag=None  # Chatbot arena dataset doesn't have category tags
            )
            entries.append(entry)
            
        except (ValueError, KeyError, TypeError) as e:
            skipped_count += 1
            warnings.warn(f"Skipping row: {str(e)}", UserWarning)
            continue
    
    if skipped_count > 0:
        warnings.warn(f"Skipped {skipped_count} rows due to errors", UserWarning)
            
    return TrainingData(entries=entries)
