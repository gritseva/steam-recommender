import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config.config import TRANSFORMER_CHECKPOINT, DEVICE
import logging
import traceback

logger = logging.getLogger(__name__)


def load_transformer_model():
    """
    Load the pre-trained Transformer model and tokenizer with quantization settings for optimized memory usage.

    Returns:
        tuple: (tokenizer, model) if successful, or (None, None) if loading fails.
    """
    try:
        # Load the tokenizer from the specified checkpoint directory
        tokenizer = AutoTokenizer.from_pretrained(
            TRANSFORMER_CHECKPOINT, use_fast=False)

        # Configure BitsAndBytes settings for quantization (adjust based on your hardware)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=False,
            bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        # Load the causal language model with quantization configuration
        model = AutoModelForCausalLM.from_pretrained(
            TRANSFORMER_CHECKPOINT,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="cuda",
            quantization_config=bnb_config
        )

        # Ensure that token IDs for EOS and PAD are set correctly
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(
                "<|endoftext|>")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        logger.info(
            f"Transformer model and tokenizer loaded successfully from {TRANSFORMER_CHECKPOINT}.")
        return tokenizer, model
    except Exception as e:
        logger.error(
            f"Error loading Transformer model from {TRANSFORMER_CHECKPOINT}: {e}")
        print("Exception traceback while loading transformer model:")
        traceback.print_exc()
        return None, None
