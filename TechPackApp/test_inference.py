#test if model can gen valid json 

import torch
import json
from techpack_generator.ml_model.model import Transformer
from techpack_generator.ml_model.config import ModelConfig
from techpack_generator.ml_model.dataset import GarmentTokenizer

def test_model():
    config = ModelConfig()

    #load tokenizer
    tokenizer = GarmentTokenizer()
    tokenizer.load(config.tokenizer_file)

    #create model
    model = Transformer(
        src_vocab_size=tokenizer.vocab_size,
        tgt_vocab_size=tokenizer.vocab_size,
        d_model=config.d_model,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout_rate
    )

    #load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        checkpoint = torch.load(f'{config.model_dir}/best_model.pth', 
                               map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
    except FileNotFoundError:
        return
    
    test_cases = [
        #basic inputs
        "black leather jacket",
        "navy blue blazer",
        "red cropped jacket with zippers",
        "oversized cream sweater",
        "fitted black dress",

        #compound words - tests the compound splitter
        "woolblend overcoat charcoal grey",
        "darkgreen silkblend evening gown",
        "lightweight cottonblend hoodie",

        #complex multi-detail inputs
        "slim fit navy pinstripe double breasted wool suit jacket with peak lapels and flap pockets",
        "high waisted wide leg cream linen trousers with pleated front and side zip",
        "oversized washed black heavyweight cotton hoodie with kangaroo pocket and ribbed cuffs",
        "fitted emerald green satin midi dress with cowl neck and adjustable spaghetti straps",

        #edge cases
        "jacket",                           #single word
        "a]b[c invalid!! chars...",         #noisy input with symbols
        "UPPERCASE RED LEATHER COAT",       #all caps (should lowercase)
    ]
    
    for test_input in test_cases:
        print(f"\nInput: {test_input}")

        # Encode
        input_ids = tokenizer.encode(test_input, config.max_seq_length)
        src = torch.tensor([input_ids]).to(device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(src, tokenizer, max_length=config.max_seq_length, device=device)

        # Decode
        output_text = tokenizer.decode(output_ids[0].tolist())

        try:
            #remove spaces that tokenizer may have added
            json_str = output_text.replace(" ", "")
            json_data = json.loads(json_str)
            print("valid JSON")
            print(f"   Action: {json_data['action']}")

            if 'tech_pack' in json_data:
                tp = json_data['tech_pack']
                print(f"   Garment type: {tp['garment_type']}")
                if 'material' in tp:
                    print(f"   Colour: {tp['material']['colour']}")
                    print(f"   Fabric: {tp['material']['fabric_type']}")

        except json.JSONDecodeError:
            print("Invalid JSON output:")
            print(output_text)

    #test compound splitting directly
    print("\n" + "="*60)
    print("COMPOUND SPLITTING TEST")
    print("="*60)

    compound_tests = [
        ("woolblend", ["wool", "blend"]),
        ("darkgreen", ["dark", "green"]),
        ("cottonblend", ["cotton", "blend"]),
        ("silkblend", ["silk", "blend"]),
        ("leather", None),     #should stay as-is if in vocab
        ("xyzabc", None),      #unknown, no valid split
    ]

    for word, expected_parts in compound_tests:
        result = tokenizer._split_compound(word.lower())
        if expected_parts:
            #check if the expected parts are in vocab first
            parts_in_vocab = all(p in tokenizer.token2id for p in expected_parts)
            if parts_in_vocab:
                if result == expected_parts:
                    print(f"   '{word}' -> {result}")
                else:
                    print(f"   '{word}' -> {result} (expected {expected_parts})")
            else:
                print(f"   '{word}' -> {result} (parts not in vocab, UNK expected)")
        else:
            print(f"   '{word}' -> {result}")

if __name__ == "__main__":
    test_model()