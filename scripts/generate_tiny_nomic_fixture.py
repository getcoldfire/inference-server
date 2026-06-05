"""Generate a tiny nomic-bert fixture for unit tests.

Nomic path: RoPE (rotary) position embeddings + SwiGLU MLP, NO token_type_ids,
NO absolute position embedding weight tensor.

Output: tests/fixtures/tiny_nomic/{config.json, model.safetensors, tokenizer.json,
tokenizer_config.json, 1_Pooling/config.json, special_tokens_map.json, vocab.txt}.

Deterministic via seeded numpy RNG. Safe to re-run; will overwrite.
"""

import json
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors

REPO_ROOT = Path(__file__).resolve().parent.parent
DIR = REPO_ROOT / "tests" / "fixtures" / "tiny_nomic"
DIR.mkdir(parents=True, exist_ok=True)

config = {
    "architectures": ["NomicBertModel"],
    "model_type": "nomic_bert",
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 64,
    "vocab_size": 256,
    "max_position_embeddings": 2048,
    "type_vocab_size": 0,
    "hidden_act": "swiglu",
    "position_embedding_type": "rotary",
    "rope_theta": 10000.0,
    "layer_norm_eps": 1e-12,
}
(DIR / "config.json").write_text(json.dumps(config, indent=2))

rng = np.random.default_rng(43)
weights = {}
H = config["hidden_size"]
V = config["vocab_size"]
IS = config["intermediate_size"]

weights["embeddings.word_embeddings.weight"] = rng.normal(0, 0.02, (V, H)).astype(np.float32)
# No position_embeddings, no token_type_embeddings — RoPE in attention, no segment ids
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
    # SwiGLU MLP: gate + up + down projections (no biases)
    weights[f"{p}.mlp.gate.weight"] = rng.normal(0, 0.02, (IS, H)).astype(np.float32)
    weights[f"{p}.mlp.up.weight"] = rng.normal(0, 0.02, (IS, H)).astype(np.float32)
    weights[f"{p}.mlp.down.weight"] = rng.normal(0, 0.02, (H, IS)).astype(np.float32)
    weights[f"{p}.mlp.LayerNorm.weight"] = np.ones(H, dtype=np.float32)
    weights[f"{p}.mlp.LayerNorm.bias"] = np.zeros(H, dtype=np.float32)

save_file(weights, str(DIR / "model.safetensors"))

# --- Tokenizer (same as tiny_bert; vocab=256, BERT-style WordPiece) ---
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

(DIR / "vocab.txt").write_text("\n".join(vocab.keys()) + "\n")

tokenizer_config = {
    "do_lower_case": False,
    "model_max_length": 2048,
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
