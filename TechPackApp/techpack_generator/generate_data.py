# data gen using claude call 
import json 
from dotenv import load_dotenv
load_dotenv()
import requests 
import anthropic 
import os
import time 
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


class DataGenerator:
    #gen multi turn training data 

    def __init__(self, api_key: None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.total_cost = 0

    def generate_single_turn_batch(self, num_examples=20):
        #gen single turn example

        prompt = f"""generate {num_examples} SINGLE-TURN garment creation examples.

Return JSON array with initial tech pack creations.

[
  {{
    "conversation_id": "conv_001",
    "turns": [
      {{
        "turn": 1,
        "user": "black leather jacket with zippers",
        "context": null,
        "assistant": {{
          "action": "create",
          "tech_pack": {{
            "garment_type": "leather jacket",
            "material": {{"fabric_type": "leather", "colour": "black", "fabric_weight": "1.2mm"}},
            "measurements": {{"chest": 42, "waist": 38, "sleeve_length": 25, "body_length": 28}},
            "construction": {{"seam_type": "topstitch", "closure_type": "zipper", "pockets": "zippered"}},
            "style": {{"fit": "fitted", "length": "regular", "details": ["silver hardware", "front zipper"]}}
          }}
        }}
      }}
    ]
  }}
]

Vary garment types, colours, fits, styles!"""

        return self._call_api(prompt)
    
    def generate_two_turn_batch(self, num_examples=10):
        #gen 2 turn conversations with mutlitple in turn 2
        
        prompt = f"""Generate {num_examples} TWO-TURN conversations.

Turn 1: User creates garment
Turn 2: User makes MULTIPLE changes at once (realistic!)

IMPORTANT: Turn 2 should include 2-4 changes in ONE request.

Examples of multi-change requests:
- "add zippers on shoulders and arms, make it cropped, and change to burgundy"
- "remove the hood, make it oversized fit, and add patch pockets"
- "change to navy blue, add quilted panels, and make sleeves longer"
- "make it distressed leather, add more zippers everywhere, and fitted instead of loose"

Return JSON array:
[
  {{
    "conversation_id": "conv_100",
    "turns": [
      {{
        "turn": 1,
        "user": "black leather jacket",
        "context": null,
        "assistant": {{
          "action": "create",
          "tech_pack": {{
            "garment_type": "leather jacket",
            "material": {{"fabric_type": "leather", "colour": "black", "fabric_weight": "1.2mm"}},
            "measurements": {{"chest": 42, "waist": 38, "sleeve_length": 25, "body_length": 28}},
            "construction": {{"seam_type": "topstitch", "closure_type": "zipper", "pockets": "zippered"}},
            "style": {{"fit": "fitted", "length": "regular", "details": ["silver hardware", "front zipper"]}}
          }}
        }}
      }},
      {{
        "turn": 2,
        "user": "add zippers on shoulders and arms, make it cropped, and change to burgundy",
        "context": {{
          "previous_tech_pack": "{{...see turn 1...}}",
          "previous_user_input": "black leather jacket"
        }},
        "assistant": {{
          "action": "modify",
          "changes": [
            {{
              "field": "style.details",
              "operation": "add",
              "value": ["shoulder zippers", "arm zippers"],
              "reason": "user requested zippers on shoulders and arms"
            }},
            {{
              "field": "measurements.body_length",
              "operation": "set",
              "value": 22,
              "reason": "user requested cropped length"
            }},
            {{
              "field": "style.length",
              "operation": "set",
              "value": "cropped",
              "reason": "user requested cropped length"
            }},
            {{
              "field": "material.colour",
              "operation": "set",
              "value": "burgundy",
              "reason": "user requested burgundy colour"
            }}
          ],
          "tech_pack": {{
            "garment_type": "leather jacket",
            "material": {{"fabric_type": "leather", "colour": "burgundy", "fabric_weight": "1.2mm"}},
            "measurements": {{"chest": 42, "waist": 38, "sleeve_length": 25, "body_length": 22}},
            "construction": {{"seam_type": "topstitch", "closure_type": "zipper", "pockets": "zippered"}},
            "style": {{"fit": "fitted", "length": "cropped", "details": ["silver hardware", "front zipper", "shoulder zippers", "arm zippers"]}}
          }}
        }}
      }}
    ]
  }}
]

Make turn 2 have 2-4 DIFFERENT CHANGES (colour + feature + fit, etc)!"""
        return self._call_api(prompt)
    
    def generate_three_turn_batch(self, num_examples=8):
        #gen 3 turn conversations
        
        prompt = f"""Generate {num_examples} THREE-TURN conversations.
Turn 1: Intial garment creation
Turn 2: Multiple changes (2-3 changes) 
Turn 3: MOre multiple changes (2-3 changes)

Example realistic flow:
Turn 1: "navy blazer, formal"
Turn 2: "add patch pockets and make it more fitted"
Turn 3: "change to charcoal grey and add elbow patches"

Each modification should have 2-3 changes

Return JSON array with proper change tracking:  
[
  {{
    "conversation_id": "conv_200",
    "turns": [
      {{turn 1: creation}},
      {{
        "turn": 2,
        "user": "add patch pockets and make it more fitted",
        "assistant": {{
          "action": "modify",
          "changes": [
            {{
              "field": "construction.pockets",
              "operation": "set",
              "value": "patch pockets"
            }},
            {{
              "field": "style.fit",
              "operation": "set",
              "value": "fitted"
            }},
            {{
              "field": "measurements.chest",
              "operation": "set",
              "value": 40
            }},
            {{
              "field": "measurements.waist",
              "operation": "set",
              "value": 34
            }}
          ],
          "tech_pack": {{...updated...}}
        }}
      }},
      {{
        "turn": 3,
        "user": "change to charcoal grey and add elbow patches",
        "assistant": {{
          "action": "modify",
          "changes": [
            {{
              "field": "material.colour",
              "operation": "set",
              "value": "charcoal grey"
            }},
            {{
              "field": "style.details",
              "operation": "add",
              "value": ["elbow patches"]
            }}
          ],
          "tech_pack": {{...updated...}}
        }}
      }}
    ]
  }}
]"""
        return self._call_api(prompt)
    
    def generate_four_turn_batch(self, num_examples=5):
        #gen 4 turn conversations
        
        prompt = f"""Generate {num_examples} FOUR-TURN conversations.

Include COMPLEX multi-change requests (3-5 changes in one turn).

example flow:
Turn 1: "black leather jacket"
Turn 2: "add zippers on shoulders, arms, and chest, plus make it cropped"
Turn 3: "change to distressed brown leather, add quilted panels, and make it oversized fit"
Turn 4: "remove chest zippers, add hood, and change sleeve zippers to silver"

Turn 2: 4 changes
Turn 3: 3 changes (including fit change that affects measurements)
Turn 4: 3 changes (mix of add/remove/modify)

Return JSON array with detailed change tracking."""

        return self._call_api(prompt)
    
    def generate_complex_modifications_batch(self, num_examples=20):
       
        #gen complex multi change requests (5-8 changes in one turn)
        prompt = f"""Generate {num_examples} conversations with COMPLEX MULTI-CHANGE modifications.

Focus on REALISTIC complex requests users actually make:
TURN 1: User creates garment (simple)
TURN 2: User makes COMPLEX multi-change request (5-8 changes) in ONE turn.

Examples of complex multi-change requests for turn 2:
1. "make it oversized fit, add a hood, change to grey, put zippers on sleeves and chest, make it cropped, and add quilted panels"
   (6 changes: fit, feature, colour, multiple features, length, texture)

2. "completely redo it - switch to denim instead of leather, add patches all over, make it fitted, remove all pockets, change to distressed finish, and add fringe on bottom"
   (6 changes: material, features, fit, remove, finish, details)

3. "transform it into oversized style, burgundy distressed leather, quilted shoulders, multiple zippers everywhere, slim fit arms, cropped length, and remove the collar"
   (7 changes: fit, colour+material, texture, features, fit variation, length, remove)

Return JSON array with TWO-TURN structure:
[
  {{
    "conversation_id": "conv_001",
    "turns": [
      {{
        "turn": 1,
        "user": "black leather jacket",
        "context": null,
        "assistant": {{
          "action": "create",
          "tech_pack": {{
            "garment_type": "leather jacket",
            "material": {{"fabric_type": "leather", "colour": "black", "fabric_weight": "1.2mm"}},
            "measurements": {{"chest": 42, "waist": 38, "sleeve_length": 25, "body_length": 28}},
            "construction": {{"seam_type": "topstitch", "closure_type": "zipper", "pockets": "zippered"}},
            "style": {{"fit": "fitted", "length": "regular", "details": ["silver hardware", "front zipper"]}}
          }}
        }}
      }},
      {{
        "turn": 2,
        "user": "make it oversized, add a hood, change to grey, put zippers everywhere, make it cropped, and add quilted shoulders",
        "context": {{
          "previous_tech_pack": "{{...see turn 1...}}",
          "previous_user_input": "black leather jacket"
        }},
        "assistant": {{
          "action": "modify",
          "changes": [
            {{
              "field": "style.fit",
              "operation": "set",
              "value": "oversized",
              "reason": "user requested oversized fit"
            }},
            {{
              "field": "measurements.chest",
              "operation": "set",
              "value": 48,
              "reason": "oversized fit requires larger chest measurement"
            }},
            {{
              "field": "measurements.waist",
              "operation": "set",
              "value": 46,
              "reason": "oversized fit requires larger waist measurement"
            }},
            {{
              "field": "style.details",
              "operation": "add",
              "value": ["hood"],
              "reason": "user requested hood"
            }},
            {{
              "field": "material.colour",
              "operation": "set",
              "value": "grey",
              "reason": "user requested grey colour"
            }},
            {{
              "field": "style.details",
              "operation": "add",
              "value": ["sleeve zippers", "chest zippers", "side zippers"],
              "reason": "user requested zippers everywhere"
            }},
            {{
              "field": "measurements.body_length",
              "operation": "set",
              "value": 22,
              "reason": "user requested cropped length"
            }},
            {{
              "field": "style.length",
              "operation": "set",
              "value": "cropped",
              "reason": "user requested cropped length"
            }},
            {{
              "field": "style.details",
              "operation": "add",
              "value": ["quilted shoulders"],
              "reason": "user requested quilted shoulders"
            }}
          ],
          "tech_pack": {{
            "garment_type": "leather jacket",
            "material": {{"fabric_type": "leather", "colour": "grey", "fabric_weight": "1.2mm"}},
            "measurements": {{"chest": 48, "waist": 46, "sleeve_length": 25, "body_length": 22}},
            "construction": {{"seam_type": "topstitch", "closure_type": "zipper", "pockets": "zippered"}},
            "style": {{"fit": "oversized", "length": "cropped", "details": ["silver hardware", "front zipper", "hood", "sleeve zippers", "chest zippers", "side zippers", "quilted shoulders"]}}
          }}
        }}
      }}
    ]
  }}
]

Return JSON with proper change arrays showing ALL modifications. Make turn 2 have 5-8 changes, covering multiple aspects!"""

        return self._call_api(prompt)
    
    def _call_api(self, prompt):
        #make the actual API call to anthropic and track cost

        try: 
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=12000,
                messages=[{"role": "user", "content": prompt}]
            )

            #track cost
            input_tokens = message.usage.input_tokens
            output_tokens = message.usage.output_tokens
            cost = (input_tokens * 3.0 / 1_000_000) + (output_tokens * 15.0 / 1_000_000)
            self.total_cost += cost

            response_text = message.content[0].text

            #extract JSON from response
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            json_text = response_text[json_start:json_end]

            conversations = json.loads(json_text)

            return conversations, cost
        
        except Exception as e:
            print(f"Error during API call: {e}")
            return [], 0
        
    def generate_complete_dataset(self):
        #gen complete multi turn dataset 

        print("Multi turn data generation")
        print(f"{'-'*40}")

        #loading existing data, and removing single turn data - as single turn data was produced then it stopped.
        try:
            checkpoint_file = 'techpack_generator/training_data/checkpoint_4730.json'
            with open(checkpoint_file, 'r') as f:
                all_conversations = json.load(f)
            print(f"Resuming from checkpoint with {len(all_conversations)} examples.")

        except Exception as e:
            print(f"No checkpoint found, starting fresh. ({e})")
            all_conversations = []

    
        
        #5 - complex modifications 
        print("5. Generating complex modification examples...")
        for i in range(25): # reduced from 100 to 50, getting too costly, reduced then down to 25 as 25 were made but of only single turn - again too costly.
            print(f" Batch {i+1}/50...", end='', flush=True)
            batch,cost = self.generate_complex_modifications_batch(5)
            all_conversations.extend(batch)
            print(f" Done. Cost: ${cost:.3f}")
            time.sleep(5) # rate limit - avoid!
            if (i + 1) %8 == 0: # every 8 batches, checkpoint save
                self.save_dataset(all_conversations, f'checkpoint_{len(all_conversations)}.json')
                print (f" Checkpoint saved at {len(all_conversations)} examples.")

        print(f"\n{'-'*40}")
        print("Data generation complete.")
        print(f"Total conversations generated: {len(all_conversations)}")
        print(f"Total cost incurred: ${self.total_cost:.2f}")

        return all_conversations
    
    def flatten_for_training(self, conversations):
        #convert multi turn conversaitons to training format

        training_examples = []

        for conv in conversations:
            if 'turns' not in conv:
                continue  # skip malformed conversation without turns
            conversation_history = []

            for turn_data in conv['turns']:
                #skip data without user or assistant output (malformed data)
                if 'user' not in turn_data or 'assistant' not in turn_data:
                    continue 
                turn_num = turn_data['turn']
                user_input = turn_data['user']
                assistant_output = turn_data['assistant']

                #create training example with context
                example = {
                    'conversation_id': conv.get('conversation_id') or conv.get('cargo_id', 'unknown'),  # handle typo from ai, and handle missing conversation_id,
                    'turn': turn_num,
                    'history': list(conversation_history),
                    'input': user_input,
                    'output': json.dumps(assistant_output)
                }

                training_examples.append(example)
                #update history
                conversation_history.append({
                    'user': user_input,
                    'assistant': assistant_output
                })

        return training_examples
    
    def save_dataset(self, data, filename):
        #save datset
        filepath = f'techpack_generator/training_data/{filename}'
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Dataset saved to {filepath}")


if __name__ == "__main__":
    print("Starting data generation...")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set in environment.")
        exit(1)

    response = input("\nProceed? (y/n): ").lower()
    if response != 'y':
        print("Exiting.")
        exit(0)

    #generate 
    generator = DataGenerator(None)

    #load conversations
    with open('C:\\Users\\georg\\OneDrive\\Desktop\\Project AI Tech\\ghl5\\TechPackApp\\techpack_generator\\training_data\\multi_turn_techpack_conversations.json', 'r') as f:
        conversations = json.load(f)
    print(f"\nLoaded {len(conversations)} conversations for training.")

    #flatten for training
    training_data = generator.flatten_for_training(conversations)

    #split 
    split = int(len(training_data) * 0.9) # 90% train, 10% val
    train_data = training_data[:split]
    val_data = training_data[split:]

    generator.save_dataset(train_data, 'techpack_training_data_train.json')
    generator.save_dataset(val_data, 'techpack_training_data_val.json')

    