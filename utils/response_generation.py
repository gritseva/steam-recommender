import re
import logging
from models.transformer_model import load_transformer_model


# The generate_response function reloads the transformer model if needed.
# In a production setup, you might store a reference to the loaded model in your application context to avoid reloading.


logger = logging.getLogger(__name__)


def generate_custom_response(raw_text: str, cleaning_patterns: list = None) -> str:
    """
    Clean up raw generated text by removing instruction tokens and normalizing whitespace.

    Args:
        raw_text (str): The raw text from the model.
        cleaning_patterns (list, optional): List of regex patterns to remove.

    Returns:
        str: The cleaned response text.
    """
    if cleaning_patterns is None:
        cleaning_patterns = [
            r'\[INST\].*?\[/INST\]',  # Remove instructional markers
            r'<s>|</s>',              # Remove start/end tokens
            r'\s+',                   # Normalize whitespace
        ]
    cleaned = raw_text
    for pattern in cleaning_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.strip()
    if len(cleaned.split()) < 5:
        return "I'm here to help with game recommendations! Let me know what you're looking for."
    return cleaned


def generate_response(user_message: str, recommendations, session, max_new_tokens: int = 200) -> str:
    """
    Generate a friendly recommendation response using the list of recommended games.

    Constructs a prompt from the user message and a truncated list of recommendations,
    then uses the transformer model to generate a dynamic response.

    Args:
        user_message (str): The original user message.
        recommendations (pd.DataFrame): DataFrame containing recommended game data.
        session: The user session (for potential context).
        max_new_tokens (int): Maximum tokens for generation.

    Returns:
        str: The generated response.
    """
    # Build a recommendation list with truncated descriptions
    recommendation_list = "\n".join([
        f"{idx+1}. {row['title']}: {row['description'][:200]}..."
        for idx, row in recommendations.head(3).iterrows()
    ])

    prompt = (
        "[INST] You are a helpful game recommendation assistant. Based on the user's interests, "
        "provide a friendly, engaging response that introduces the following recommended games in 2-3 sentences.\n\n"
        f"User message: {user_message}\n\n"
        "Recommended games:\n"
        f"{recommendation_list}\n\n"
        "Response:[/INST]"
    )

    # Load transformer model and tokenizer (using our existing loader)
    tokenizer, model = load_transformer_model()
    if tokenizer is None or model is None:
        logger.error("Transformer model unavailable in generate_response.")
        return generate_custom_response(prompt)

    try:
        inputs = tokenizer(prompt, return_tensors="pt",
                           padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.65,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id
        )
        generated_text = tokenizer.decode(
            output[0], skip_special_tokens=True).strip()
        return generate_custom_response(generated_text)
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return generate_custom_response(prompt)
