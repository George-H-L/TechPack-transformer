import json
from pathlib import Path

from .tokenizer import GarmentTokenizer
from .config import ModelConfig


def main():
    config = ModelConfig()

    # Load training data
    try:
        with open(config.train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
    except Exception as e:
        print(f"Error loading training data: {e}")
        return

    # Extract input/output text
    texts = []
    for example in train_data:
        texts.append(example['input'])
        texts.append(example['output'])

    # Build tokenizer vocabulary
    tokenizer = GarmentTokenizer()
    vocab_size = tokenizer.build_vocab(texts, min_freq=2)

    # Unknown token rate check , adjust min freq if too high
    unk_id = tokenizer.token2id["<UNK>"]
    unknown_count = 0
    total_tokens = 0

    for text in texts[:1000]:  # sample subset for speed
        ids = tokenizer.encode(text, max_length=config.max_seq_length)
        total_tokens += len(ids)
        unknown_count += ids.count(unk_id)

    unknown_rate = (unknown_count / total_tokens) * 100 if total_tokens else 0

    print(f"Vocab size: {vocab_size}")
    print(f"Unknown token rate: {unknown_rate:.2f}%")

    # Save tokenizer
    Path(config.tokenizer_file).parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(config.tokenizer_file)

    # Test encode/decode
    test_text = train_data[0]['input']
    encoded = tokenizer.encode(test_text, max_length=20)
    decoded = tokenizer.decode(encoded)

    print("Sample encode/decode test:")
    print(decoded)


if __name__ == "__main__":
    main()
