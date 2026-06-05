"""Generate a tiny vanilla-BERT fixture for unit tests.

Vanilla BERT path: absolute position embeddings + GeLU MLP + token_type_ids.
Output: tests/fixtures/tiny_bert/{config.json, model.safetensors, tokenizer.json,
tokenizer_config.json, 1_Pooling/config.json, special_tokens_map.json, vocab.txt}.

Deterministic via seeded numpy RNG. Safe to re-run; will overwrite.
"""

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors

REPO_ROOT = Path(__file__).resolve().parent.parent
DIR = REPO_ROOT / "tests" / "fixtures" / "tiny_bert"
DIR.mkdir(parents=True, exist_ok=True)

config = {
    "architectures": ["BertModel"],
    "model_type": "bert",
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "vocab_size": 256,
    "max_position_embeddings": 128,
    "type_vocab_size": 2,
    "hidden_act": "gelu",
    "position_embedding_type": "absolute",
    "layer_norm_eps": 1e-12,
}
(DIR / "config.json").write_text(json.dumps(config, indent=2))

rng = np.random.default_rng(42)
weights = {}
H = config["hidden_size"]
V = config["vocab_size"]
P = config["max_position_embeddings"]
TT = config["type_vocab_size"]
IS = config["intermediate_size"]

weights["embeddings.word_embeddings.weight"] = rng.normal(0, 0.02, (V, H)).astype(np.float32)
weights["embeddings.position_embeddings.weight"] = rng.normal(0, 0.02, (P, H)).astype(np.float32)
weights["embeddings.token_type_embeddings.weight"] = rng.normal(0, 0.02, (TT, H)).astype(np.float32)
weights["embeddings.LayerNorm.weight"] = np.ones(H, dtype=np.float32)
weights["embeddings.LayerNorm.bias"] = np.zeros(H, dtype=np.float32)

for layer in range(config["num_hidden_layers"]):
    p = f"encoder.layer.{layer}"
    for proj in ("query", "key", "value"):
        weights[f"{p}.attention.self.{proj}.weight"] = rng.normal(0, 0.02, (H, H)).astype(np.float32)
        weights[f"{p}.attention.self.{proj}.bias"] = np.zeros(H, dtype=np.float32)
    weights[f"{p}.attention.output.dense.weight"] = rng.normal(0, 0.02, (H, H)).astype(np.float32)
    weights[f"{p}.attention.output.dense.bias"] = np.zeros(H, dtype=np.float32)
    weights[f"{p}.attention.output.LayerNorm.weight"] = np.ones(H, dtype=np.float32)
    weights[f"{p}.attention.output.LayerNorm.bias"] = np.zeros(H, dtype=np.float32)
    weights[f"{p}.intermediate.dense.weight"] = rng.normal(0, 0.02, (IS, H)).astype(np.float32)
    weights[f"{p}.intermediate.dense.bias"] = np.zeros(IS, dtype=np.float32)
    weights[f"{p}.output.dense.weight"] = rng.normal(0, 0.02, (H, IS)).astype(np.float32)
    weights[f"{p}.output.dense.bias"] = np.zeros(H, dtype=np.float32)
    weights[f"{p}.output.LayerNorm.weight"] = np.ones(H, dtype=np.float32)
    weights[f"{p}.output.LayerNorm.bias"] = np.zeros(H, dtype=np.float32)

save_file(weights, str(DIR / "model.safetensors"))

# --- Tokenizer ---
# Minimal BERT-style WordPiece. Vocab has special tokens + 256 ordinary tokens
# (we use a 256-vocab in config; tokens beyond vocab_size won't be produced
# because tokenize_special and the dummy WordPiece only consume known words).
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
vocab = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
for i in range(len(SPECIAL_TOKENS), config["vocab_size"]):
    vocab[f"t{i}"] = i

tok = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]"))
tok.pre_tokenizer = pre_tokenizers.Whitespace()
tok.decoder = decoders.WordPiece()
tok.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", vocab["[CLS]"]), ("[SEP]", vocab["[SEP]"])],
)
tok.enable_padding(pad_id=vocab["[PAD]"], pad_token="[PAD]")
tok.save(str(DIR / "tokenizer.json"))

# vocab.txt for slow-tokenizer fallback
(DIR / "vocab.txt").write_text("\n".join(vocab.keys()) + "\n")

# tokenizer_config.json — needed by transformers.AutoTokenizer to know this is BERT
tokenizer_config = {
    "do_lower_case": False,
    "model_max_length": config["max_position_embeddings"],
    "tokenizer_class": "BertTokenizerFast",
    "unk_token": "[UNK]",
    "sep_token": "[SEP]",
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "mask_token": "[MASK]",
}
(DIR / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2))

(DIR / "special_tokens_map.json").write_text(
    json.dumps(
        {
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]",
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
        },
        indent=2,
    )
)

# Sentence-Transformers-style pooling config
(DIR / "1_Pooling").mkdir(exist_ok=True)
(DIR / "1_Pooling" / "config.json").write_text(
    json.dumps(
        {
            "word_embedding_dimension": H,
            "pooling_mode_cls_token": False,
            "pooling_mode_mean_tokens": True,
            "pooling_mode_max_tokens": False,
            "pooling_mode_mean_sqrt_len_tokens": False,
        },
        indent=2,
    )
)

print(f"Fixture written to {DIR}")
