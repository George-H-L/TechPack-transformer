# Merges Claude tops data with Ollama bottoms data and rebuilds the tokenizer.
# Run from TechPackApp/: python -m techpack_generator.ml_model.prepare_combined_data

import json
import random
import shutil
from pathlib import Path

from .tokenizer import GarmentTokenizer
from .config import ModelConfig

config = ModelConfig()

CLAUDE_TRAIN = Path(config.data_dir) / 'synthetic' / 'claude' / 'techpack_training_data_train.json'
CLAUDE_VAL   = Path(config.data_dir) / 'synthetic' / 'claude' / 'techpack_training_data_val.json'
OLLAMA_FILE  = Path(config.data_dir) / 'synthetic' / 'ollama' / 'bottoms_qwen25_7b_1000.json'

OUT_TRAIN    = Path(config.data_dir) / 'combined_train.json'
OUT_VAL      = Path(config.data_dir) / 'combined_val.json'

TOKENIZER_PATH   = Path(config.tokenizer_file)
TOKENIZER_BACKUP = TOKENIZER_PATH.parent / 'tokenizer_tops_only.json'

# the tokenizer regex (\w+|[^\w\s]) keeps underscores inside words so
# "leg_opening" is a single token, but "5-pocket" splits to ["5", "-", "pocket"]
CRITICAL_TERMS = [
    'inseam', 'outseam', 'rise', 'hips', 'thigh', 'leg_opening',
    'waistband_height', 'jeans', 'trousers', 'chinos', 'shorts',
    'joggers', 'skirt', 'denim', 'twill', 'corduroy', 'fly',
    'drawstring', '5', 'pocket', 'slash', 'skinny', 'tapered',
    'straight', 'baggy', 'ankle',
]


def _load_flat(path: Path) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _flatten_ollama(path: Path) -> list:
    # converts raw Ollama conversation dicts into the flat {input, output} format
    with open(path, 'r', encoding='utf-8') as f:
        conversations = json.load(f)

    examples = []
    for conv in conversations:
        cid = conv.get('conversation_id', 'unknown')
        for turn in conv.get('turns', []):
            user_text = turn.get('user', '').strip()
            assistant = turn.get('assistant', {})
            if not user_text or not assistant:
                continue

            output_dict = {
                'action': assistant.get('action', 'create'),
                'tech_pack': assistant.get('tech_pack', {}),
            }
            if assistant.get('changes'):
                output_dict['changes'] = assistant['changes']

            examples.append({
                'conversation_id': f"{cid}_t{turn.get('turn', 1)}",
                'turn': turn.get('turn', 1),
                'history': [],
                'input': user_text,
                'output': json.dumps(output_dict),
            })

    return examples


def build_combined_dataset():
    print("Flatten and merge")

    claude_train = _load_flat(CLAUDE_TRAIN)
    claude_val   = _load_flat(CLAUDE_VAL)
    ollama       = _flatten_ollama(OLLAMA_FILE)

    print(f"  Claude train : {len(claude_train):,} examples")
    print(f"  Claude val   : {len(claude_val):,} examples")
    print(f"  Ollama flat  : {len(ollama):,} examples")

    all_examples = claude_train + claude_val + ollama

    random.seed(42)
    random.shuffle(all_examples)

    split      = int(len(all_examples) * 0.9)
    train_data = all_examples[:split]
    val_data   = all_examples[split:]

    with open(OUT_TRAIN, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    with open(OUT_VAL, 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)

    print(f"  combined_train.json  {len(train_data):,} examples")
    print(f"  combined_val.json    {len(val_data):,} examples")
    return train_data, val_data


def rebuild_tokenizer(train_data: list, val_data: list):
    print("\nRebuild tokenizer")

    texts = [ex['input'] + ' ' + ex['output'] for ex in train_data + val_data]

    tokenizer = GarmentTokenizer()
    # min_freq=1 so rare bottoms terms still get their own IDs
    tokenizer.build_vocab(texts, min_freq=1)

    missing = [t for t in CRITICAL_TERMS if t not in tokenizer.token2id]
    if missing:
        print(f"  Manually adding {len(missing)} missing terms: {missing}")
        for term in missing:
            tokenizer.token2id[term] = tokenizer.vocab_size
            tokenizer.id2token[tokenizer.vocab_size] = term
            tokenizer.vocab_size += 1

    print("\n  Critical term check:")
    for term in CRITICAL_TERMS:
        status = "ok" if term in tokenizer.token2id else "MISSING"
        print(f"    {term:<22} {status}")

    if TOKENIZER_PATH.exists():
        shutil.copy(TOKENIZER_PATH, TOKENIZER_BACKUP)
        print(f"\n  Backed up old tokenizer to {TOKENIZER_BACKUP.name}")

    tokenizer.save(str(TOKENIZER_PATH))
    print(f"  New vocab size: {tokenizer.vocab_size}")
    return tokenizer


if __name__ == '__main__':
    train_data, val_data = build_combined_dataset()
    rebuild_tokenizer(train_data, val_data)
