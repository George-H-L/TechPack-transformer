import argparse
import json
import time
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import ollama
from pydantic import BaseModel, Field


SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / "synthetic" / "ollama"
DEFAULT_MODEL = "qwen2.5:7b"

# reassigned in main() so generate_one can read it without
# threading the model name through every function call
MODEL_NAME = DEFAULT_MODEL


GARMENTS = ["jeans", "trousers", "chinos", "shorts", "joggers",
            "midi skirt", "pencil skirt", "a-line skirt",
            "wide-leg trousers", "cargo trousers"]
COLOURS = ["black", "navy", "indigo", "charcoal", "olive", "cream",
           "stone", "burgundy", "rust", "khaki", "beige", "white"]
FITS = ["skinny", "slim", "straight", "regular", "relaxed",
        "wide", "tapered", "baggy", "very baggy"]


# Diversity injection. The user-voice and phrasing pools push variation
# through prompt-space rather than through sampling temperature, so the
# structure stays clean while the user messages vary in register.
USER_VOICE_PERSONAS = [
    "a small clothing brand owner emailing a designer briefly",
    "a designer typing quick spec notes to themselves",
    "a non-technical customer describing what they want in everyday language",
    "an experienced pattern cutter using technical vocabulary",
    "a vintage seller describing a garment they want to reproduce",
    "a streetwear designer using informal language and slang",
    "a costume designer specifying a garment for a film or play",
    "someone leaving a comment on a sewing forum",
    "a tailor writing measurements down quickly with abbreviations",
    "a fashion student describing a project garment in detail",
]

PHRASING_HINTS = [
    "use 1 to 2 sentences, casual",
    "use 2 to 3 sentences with specific measurements mentioned",
    "use a single short sentence, fragment style",
    "use 3 to 4 sentences, more descriptive",
    "use a single run-on sentence",
    "mention a use case or occasion the garment is for",
    "mention a specific reference aesthetic without naming a brand",
    
]

EDGE_CASE_TEMPLATES = [
    "Turn 2 reverses the change from turn 1 (e.g. user asks to undo the previous request).",
    "Turn 2 asks for an unusual combination (e.g. a slim fit with an extremely wide leg opening); the assistant complies but the changes array reason field notes the unusual combination.",
    "Turn 2 changes the fabric and colour together; the assistant updates both fields.",
    "The garment is at the extreme end of the size range (waist 26 or waist 44).",
    "The fit is at an extreme (very baggy or skinny) and the measurements reflect that.",
    "Turn 2 requests three or more changes simultaneously; the changes array contains an entry for each.",
    "specify an unusual fabric like corduroy, linen, or fleece",
    "mention the fabric weight explicitly in the description",
]


# Allowed enum values for the semantic validator. These mirror the
# constraints described in the prompt so any drift is caught after
# generation rather than baked into the training set.
ALLOWED_FABRIC_WEIGHTS = {"light", "medium", "heavy"}
ALLOWED_LENGTHS = {"shorts", "cropped", "regular", "long", "midi", "ankle"}
ALLOWED_FABRICS = {"cotton", "denim", "twill", "wool", "linen", "fleece",
                   "polyester", "corduroy", "ripstop", "canvas", "jersey"}
ALLOWED_COLOURS_SET = {c.lower() for c in COLOURS}

# Username patterns the model fell back to when the user field's
# purpose was ambiguous in earlier runs (john_doe, jane_smith, user1).
USERNAME_PATTERN = re.compile(r"^[a-z]+_[a-z]+$|^[a-z]+\d+$|^user_?\d*$", re.IGNORECASE)

# Field names that must hold integer values when used in a changes entry.
NUMERIC_FIELDS = {"waist", "hips", "inseam", "outseam", "rise", "thigh",
                  "leg_opening", "waistband_height", "body_length",
                  "chest", "sleeve_length"}


class Material(BaseModel):
    fabric_type: str = Field(
        description="A fabric name in lowercase natural language, e.g. cotton, denim, wool. Never a colour."
    )
    colour: str = Field(
        description="A colour name in lowercase natural language, e.g. black, navy. Never a fabric."
    )
    fabric_weight: str = Field(
        description="Exactly one of: light, medium, heavy. Lowercase. Never a measurement string."
    )

class Measurements(BaseModel):
    waist: int
    hips: int
    inseam: Optional[int] = None
    outseam: Optional[int] = None
    rise: int
    thigh: int
    leg_opening: int

class Construction(BaseModel):
    seam_type: str = Field(
        description="A seam type in natural language with spaces, e.g. 'flat felled', 'french seam'. Never snake_case."
    )
    closure_type: str = Field(
        description="A closure type in natural language, e.g. 'zip and button', 'elastic waistband'."
    )
    pockets: str = Field(
        description="A pocket description in natural language with spaces, e.g. 'side pockets and back pockets'. Never snake_case."
    )
    waistband_height: int

class Style(BaseModel):
    fit: str
    length: str = Field(
        description="Exactly one of: shorts, cropped, regular, long, midi, ankle. Never reference inseam or outseam values."
    )
    details: list[str] = Field(
        description="A list of short natural-language detail strings with spaces, e.g. 'contrast topstitching', 'turn-up cuffs'. Never snake_case."
    )

class TechPack(BaseModel):
    garment_type: str
    material: Material
    measurements: Measurements
    construction: Construction
    style: Style

class Change(BaseModel):
    field: str
    operation: str
    value: str
    reason: str

class Assistant(BaseModel):
    action: str
    tech_pack: TechPack
    changes: Optional[list[Change]] = None

class Turn(BaseModel):
    turn: int = Field(description="Turn number, starting from 1")
    user: str = Field(
        min_length=15,
        description=(
            "The user's natural language message describing what they want. "
            "Must be a full sentence or short paragraph, never a name or username. "
            "Examples: 'I want a pair of slim black jeans with a 32 inch waist' "
            "or 'change the fabric to denim and make it more relaxed fit'."
        ),
    )
    context: Optional[str] = None
    assistant: Assistant

class Conversation(BaseModel):
    conversation_id: str
    turns: list[Turn]


MAX_TURNS = 3


def validate_measurements(conv: dict) -> bool:
    """Numeric range checks. Pydantic guarantees the types,
    these checks guarantee the values are physically plausible."""
    try:
        m = conv["turns"][0]["assistant"]["tech_pack"]["measurements"]
        c = conv["turns"][0]["assistant"]["tech_pack"]["construction"]

        if not (26 <= m["waist"] <= 44):
            return False
        if not (32 <= m["hips"] <= 50):
            return False
        if m["waist"] >= m["hips"]:
            return False
        if not (8 <= m["rise"] <= 14):
            return False
        if not (18 <= m["thigh"] <= 30):
            return False
        if not (12 <= m["leg_opening"] <= 22):
            return False
        if not (1 <= c["waistband_height"] <= 3):
            return False
        if m.get("inseam") is not None and m.get("outseam") is not None:
            if m["inseam"] >= m["outseam"]:
                return False
        return True
    except (KeyError, TypeError):
        return False


def has_duplicate_turns(conv: dict) -> bool:
    users = [t["user"] for t in conv["turns"]]
    return len(users) != len(set(users))


def has_snake_case(s: str) -> bool:
    """True if the string looks like snake_case_id rather than natural prose."""
    return bool(re.match(r"^[a-z]+(_[a-z]+)+$", s.strip()))


def validate_semantic(conv: dict) -> bool:
    """Catches failures the Pydantic schema can't see because the types
    are right but the values are semantically wrong:
      - user field is a username instead of a real prompt
      - fabric_weight as a free-text measurement string
      - length leaking schema metadata like 'inseam null, outseam null'
      - fabric and colour fields swapped
      - snake_case bleeding into free-text fields
      - modify turns with empty changes arrays
    """
    try:
        for turn in conv["turns"]:
            assistant = turn["assistant"]
            tp = assistant["tech_pack"]
            material = tp["material"]
            style = tp["style"]
            construction = tp["construction"]
            length = style["length"].lower()

            # user message must be a real prompt, not a username
            user_msg = turn["user"]
            if len(user_msg) < 15:
                return False
            if USERNAME_PATTERN.match(user_msg.strip()):
                return False
            if user_msg.count(" ") < 2:
                return False

            # fabric_weight must be the enum, not a free-text measurement string
            if material["fabric_weight"].lower() not in ALLOWED_FABRIC_WEIGHTS:
                return False

            # length must not leak schema metadata
            if "null" in length or "inseam" in length or "outseam" in length:
                return False
            if length not in ALLOWED_LENGTHS:
                return False

            # fabric and colour must not be swapped
            if material["fabric_type"].lower() in ALLOWED_COLOURS_SET:
                return False
            if material["colour"].lower() in ALLOWED_FABRICS:
                return False

            # snake_case bleeding from schema keys into free-text fields
            if has_snake_case(construction["seam_type"]):
                return False
            if has_snake_case(construction["pockets"]):
                return False
            for detail in style["details"]:
                if has_snake_case(detail):
                    return False

            # modify turns must populate changes
            if assistant["action"] == "modify":
                changes = assistant.get("changes")
                if not changes:
                    return False

        return True
    except (KeyError, TypeError, AttributeError):
        return False


def validate_modify_consistency(conv: dict) -> bool:
    """For every modify turn, verify the tech_pack actually changed
    relative to the previous turn. Catches the failure mode where the
    model populates a changes array describing a modification but
    leaves the tech_pack identical to the previous turn."""
    prev_tp = None
    for turn in conv["turns"]:
        assistant = turn["assistant"]
        current_tp = assistant["tech_pack"]
        if assistant["action"] == "modify" and prev_tp is not None:
            changes = assistant.get("changes") or []
            if not changes:
                return False
            if json.dumps(current_tp, sort_keys=True) == json.dumps(prev_tp, sort_keys=True):
                return False
        prev_tp = current_tp
    return True


def validate_change_value_types(conv: dict) -> bool:
    """Reject if a numeric field appears in changes with a non-numeric value.
    The schema's value: str is necessary because changes can target both
    string and numeric fields, but a measurement change with value '28'
    needs to be coercible to int or downstream parsing breaks."""
    for turn in conv["turns"]:
        changes = turn["assistant"].get("changes") or []
        for change in changes:
            field_name = change["field"].split(".")[-1]
            if field_name in NUMERIC_FIELDS:
                try:
                    int(str(change["value"]).strip())
                except (ValueError, TypeError):
                    return False
    return True


# Schema string injected into the prompt. The Ollama docs note it is
# "ideal to also pass the JSON schema as a string in the prompt to
# ground the model's response", so gen it once at module load
# from the same Pydantic model used for constrained decoding. This
# means the prompt and the format= argument can never drift apart.
SCHEMA_STRING = json.dumps(Conversation.model_json_schema(), indent=2)


PROMPT = """Generate a {num_turns}-turn garment conversation for a tech pack training dataset.

Turn 1: User describes a bottom garment in natural language. Assistant creates a tech pack (action: "create", changes: null).
Turn 2+: User requests modifications in natural language. Assistant applies changes (action: "modify") and MUST populate the changes array with one entry per field altered, AND update the tech_pack to reflect those changes.

User voice for this example: write the user messages as if from {persona}. {phrasing}. The user field MUST be the user's actual message in natural language, never a username, never a name, never a short identifier.

Field rules:
- fabric_type is a fabric only (cotton, denim, twill, wool, linen, fleece, polyester, corduroy, ripstop, canvas, jersey). Never a colour.
- colour is a colour only (black, navy, indigo, charcoal, olive, cream, stone, burgundy, rust, khaki, beige, white). Never a fabric.
- fabric_weight is exactly one of: light, medium, heavy. Not a measurement.
- length is exactly one of: shorts, cropped, regular, long, midi, ankle. Never reference inseam or outseam values in this field.
- seam_type, closure_type, pockets, and details entries must be natural language with spaces. Never snake_case (no underscores).
- All string values lowercase.
- All measurements in inches as integers. Ranges: waist 26-44, hips 32-50, rise 8-14, thigh 18-30, leg_opening 12-22, waistband_height 1-3.
- inseam less than outseam. Both null for shorts and skirts.
- For changes entries on numeric fields, the value must be a numeric string parseable as an integer (e.g. "28", not "28 inches").

Each turn must have a different user message. Modification turns must include a changes array with field, operation (set/add/remove), value, and reason, AND the tech_pack must actually reflect the changes.

The output must conform exactly to this JSON schema:
{schema}

For this example, generate a {fit_hint} fit {colour_hint} {garment_hint} in turn 1, then make realistic modifications in subsequent turns.{edge_case_instruction}"""


def generate_one(conv_id):
    garment_hint = random.choice(GARMENTS)
    colour_hint = random.choice(COLOURS)
    fit_hint = random.choice(FITS)
    persona = random.choice(USER_VOICE_PERSONAS)
    phrasing = random.choice(PHRASING_HINTS)
    # weighted toward single-turn to roughly match the Claude generator's mix
    num_turns = random.choice([1, 1, 2, 3])

    # 15% of multi-turn generations get an edge-case twist
    edge_case_instruction = ""
    if num_turns >= 2 and random.random() < 0.15:
        edge_case_instruction = (
            "\n\nEdge case for this example: " + random.choice(EDGE_CASE_TEMPLATES)
        )

    prompt = PROMPT.format(
        num_turns=num_turns,
        fit_hint=fit_hint,
        colour_hint=colour_hint,
        garment_hint=garment_hint,
        schema=SCHEMA_STRING,
        persona=persona,
        phrasing=phrasing,
        edge_case_instruction=edge_case_instruction,
    )

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            format=Conversation.model_json_schema(),
            # docs recommend low temperature for structured outputs;
            # 0.3 keeps a small amount of variation for diversity
            # without the field-swap drift seen at 0.7
            options={"temperature": 0.3},
        )

        content = response["message"]["content"]
        if not content or not content.strip():
            print(f"  skip: empty response")
            return None

        conv = Conversation.model_validate_json(content)
        result = conv.model_dump()
        result["conversation_id"] = conv_id
        result["teacher"] = MODEL_NAME
        result["generated_at"] = datetime.now().isoformat()

        if has_duplicate_turns(result):
            print(f"  skip: duplicate turns detected")
            return None

        if len(result["turns"]) > MAX_TURNS:
            result["turns"] = result["turns"][:MAX_TURNS]

        if not validate_measurements(result):
            print(f"  skip: measurements out of range")
            return None

        if not validate_semantic(result):
            print(f"  skip: semantic validation failed")
            return None

        if not validate_modify_consistency(result):
            print(f"  skip: modify turn made no actual changes")
            return None

        if not validate_change_value_types(result):
            print(f"  skip: numeric change value was not numeric")
            return None

        return result

    except Exception as e:
        print(f"  skip: {type(e).__name__} - {e}")
        return None


def generate_batch(count, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            conversations = json.load(f)
        print(f"resuming from {len(conversations)} existing")
    else:
        conversations = []

    start = len(conversations)
    failures = 0
    t0 = time.time()

    print(f"generating {count} with {MODEL_NAME} -> {output_path.name}")

    for i in range(start, start + count):
        conv = generate_one(f"ollama_qwen25_7b_{i:05d}")

        if conv is None:
            failures += 1
            continue

        conversations.append(conv)
        done = len(conversations) - start

        if done % 10 == 0 and done > 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(conversations, f, indent=2)
            print(f"  {done}/{count} | failed: {failures} | {time.time() - t0:.0f}s")

        if done % 50 == 0 and done > 0:
            snapshot = output_path.with_name(f"checkpoint_{len(conversations)}.json")
            with open(snapshot, "w", encoding="utf-8") as f:
                json.dump(conversations, f, indent=2)
            print(f"  snapshot saved: {snapshot.name}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(conversations, f, indent=2)

    elapsed = time.time() - t0
    succeeded = len(conversations) - start
    print(f"done. {succeeded}/{count} succeeded, {failures} failed in {elapsed:.0f}s")

    return {
        "model": MODEL_NAME,
        "requested": count,
        "succeeded": succeeded,
        "failed": failures,
        "elapsed_seconds": elapsed,
        "completed_at": datetime.now().isoformat(),
    }


def main():
    global MODEL_NAME

    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--output-name", default="day1_test.json")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    MODEL_NAME = args.model

    output_path = OUTPUT_DIR / args.output_name
    summary = generate_batch(args.count, output_path)

    with open(output_path.with_suffix(".summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()