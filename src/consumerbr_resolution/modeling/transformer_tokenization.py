def add_single_sequence_special_tokens(tokenizer, content_ids):
    if tokenizer.cls_token_id is None or tokenizer.sep_token_id is None:
        raise ValueError("Tokenizer must define CLS and SEP token IDs.")
    if tokenizer.num_special_tokens_to_add(pair=False) != 2:
        raise ValueError("Expected two special tokens for a single BERT sequence.")
    return [int(tokenizer.cls_token_id), *content_ids, int(tokenizer.sep_token_id)]


def validate_special_token_construction(tokenizer):
    text = "ConsumerBR tokenizer compatibility validation."
    arguments = {"truncation": False, "padding": False,
                 "return_attention_mask": False, "return_token_type_ids": False}
    content = tokenizer(text, add_special_tokens=False, **arguments)["input_ids"]
    reference = tokenizer(text, add_special_tokens=True, **arguments)["input_ids"]
    if add_single_sequence_special_tokens(tokenizer, content) != reference:
        raise ValueError("Special-token construction does not match native tokenization.")


def build_input_ids(tokenizer, content_ids, max_length, strategy="head"):
    if strategy != "head":
        raise ValueError("The reduced study uses only head truncation.")
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if budget <= 0:
        raise ValueError("Maximum length must leave room for content tokens.")
    return add_single_sequence_special_tokens(tokenizer, content_ids[:budget])
