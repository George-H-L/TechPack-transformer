# making a tokenizer to convert text from the conversaions to token ids for training.

import json 
import re
from collections import Counter


class GarmentTokenizer:

    #tokenizer that should; build vocab, encode text to token ids, decode token ids back to text, save and load vocab.

    def __init__(self):
        #initialise with special tokens
        
        #special tokens - padding, start of seq and end of seq and uknowkn.
        self.PAD = '<PAD>' #for padding sequences to same length in batches
        self.SOS = '<SOS>' # start of sequence token to indicate beginning of input for model
        self.EOS = '<EOS>'# end of sequence token to indicate end of input for model
        self.UNK = '<UNK>' # unknown token for words not in vocab

        #token to id mapping
        self.token2id = {
            self.PAD: 0,
            self.SOS: 1,
            self.EOS: 2,
            self.UNK: 3
        }

        #id to token mapping
        self.id2token = {
            0: self.PAD,
            1: self.SOS,
            2: self.EOS,
            3: self.UNK
        }

        self.vocab_size = 4 #initial vocab size with special tokens

    def build_vocab(self, texts, min_freq=2):
        # build vocab from training text 
        #take list of text strings and the min frquency for a word to be introudced to vocab,

        # should return a total vocab size

        #count word frequencies

        word_counts = Counter()
        for text in texts:
            #tokenize text, not done as of yet 
            tokens = self._tokenize(text)
            word_counts.update(tokens)

         #add words above freq threshold 
        added = 0
        for word, count in word_counts.items():
            if count >= min_freq and word not in self.token2id:
                self.token2id[word] = self.vocab_size
                self.id2token[self.vocab_size] = word
                self.vocab_size += 1
                added += 1

        print(f'Added {added} words to vocab. Total vocab size: {self.vocab_size}')
        return self.vocab_size
    
    def _tokenize(self, text):
        #split text into tokens 

        #takes input string and returns list of tokens.

        if not isinstance(text, str):
            text = str(text)

        text = text.lower() #lowercase for case insensitivity

        #split on white space and keep punctuation as separate tokens
        tokens = re.findall(r'\w+|[^\w\s]', text,)
        
        return tokens
    
    def _split_compound(self, token):
        #try to split a compound word (e.g. woolblend -> wool blend) into known subwords
        #returns list of tokens - either the split parts or the original token if no split found

        if token in self.token2id:
            return [token]

        #try every possible split point, prefer longer left parts
        best_split = None
        for i in range(len(token) - 1, 1, -1):
            left = token[:i]
            right = token[i:]
            if left in self.token2id and right in self.token2id:
                best_split = [left, right]
                break

        if best_split:
            return best_split

        #try 3-way split for longer compounds (e.g. heavycottonblend)
        for i in range(2, len(token) - 2):
            for j in range(i + 2, len(token)):
                left = token[:i]
                mid = token[i:j]
                right = token[j:]
                if left in self.token2id and mid in self.token2id and right in self.token2id:
                    return [left, mid, right]

        return [token]

    def encode(self, text, max_length=120):
        #convert text to tokens
        #takes string to encode and maximum seq length, rturn list of token ids

        tokens = self._tokenize(text)

        #split compound words that arent in vocab into known subwords
        expanded = []
        for token in tokens:
            expanded.extend(self._split_compound(token))
        tokens = expanded

        #add start and end tokens
        tokens = [self.SOS] + tokens + [self.EOS]

        #convert tokens to ids, use UNK for unknown words
        ids = []
        for token in tokens:
            if token in self.token2id:
                ids.append(self.token2id[token])
            else:
                ids.append(self.token2id[self.UNK])

        # Pad or truncate
        if len(ids) < max_length:
            ids = ids + [self.token2id[self.PAD]] * (max_length - len(ids))
        else:
            ids = ids[:max_length-1] + [self.token2id[self.EOS]]
        
        return ids
    
    def decode(self, ids, skip_special_tokens=True):
        #convert token ids back to text 
        #takes list of token ids and returns decoded string, optionally skip special tokens

        tokens = []
        for id in ids:
            token = self.id2token.get(id, self.UNK)
            if skip_special_tokens and token in [self.PAD, self.SOS, self.EOS, self.UNK]:
                continue
            tokens.append(token)

        return ' '.join(tokens)

    def save(self, filepath):
        #save tokenizer to JSON file
        data = {
            'token2id': self.token2id,
            'id2token': {str(k): v for k, v in self.id2token.items()},
            'vocab_size': self.vocab_size
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ Tokenizer saved: {filepath}")
        print(f"  Vocabulary size: {self.vocab_size}")

    def load(self, filepath):
        #load tokenizer from JSON file
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.token2id = data['token2id']
        self.id2token = {int(k): v for k, v in data['id2token'].items()}
        self.vocab_size = data['vocab_size']
    
if __name__ == "__main__":
    print("="*60)
    print("TOKENIZER SELF TEST")
    print("="*60)
    
    tokenizer = GarmentTokenizer()
    
    sample_texts = [
        "black leather jacket",
        "black jacket with zippers",
        "navy blue blazer formal",
        "navy blue coat",
        "red cropped jacket",
        "red leather jacket fitted",
        '{"garment_type": "jacket", "colour": "black"}',
        '{"garment_type": "blazer", "colour": "navy"}',
    ]
    
    tokenizer.build_vocab(sample_texts, min_freq=2)
    
    print(f"\nVocabulary size: {tokenizer.vocab_size}")
    print(f"Sample mappings:")
    for token in ['black', 'leather', 'jacket', 'navy', '{', '"', ':']:
        if token in tokenizer.token2id:
            print(f"  '{token}' → {tokenizer.token2id[token]}")
    
    # Test encode
    print("\n" + "="*60)
    print("ENCODE TEST")
    print("="*60)
    
    test = "black leather jacket"
    encoded = tokenizer.encode(test, max_length=15)
    
    print(f"\nInput:   '{test}'")
    print(f"Encoded: {encoded}")
    
    # Test decode
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: '{decoded}'")
    
    if decoded.strip() == test.strip():
        print("\nDecode matches original text")

    


       