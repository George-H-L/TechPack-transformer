# loads trained model and runs inference for django views

import re
import math
import torch
import json
from pathlib import Path
from .model import Transformer
from .tokenizer import GarmentTokenizer
from .config import ModelConfig


# main generator class that django views call into
class TechPackGenerator:

    def __init__(self, model_path=None):
        self.config = ModelConfig()

        # tokenizer
        self.tokenizer = GarmentTokenizer()
        self.tokenizer.load(self.config.tokenizer_file)

        # gpu if available, otherwise cpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # build the transformer
        self.model = Transformer(
            src_vocab_size=self.tokenizer.vocab_size,
            tgt_vocab_size=self.tokenizer.vocab_size,
            d_model=self.config.d_model,
            num_encoder_layers=self.config.num_encoder_layers,
            num_decoder_layers=self.config.num_decoder_layers,
            num_heads=self.config.num_heads,
            d_ff=self.config.d_ff,
            max_seq_length=self.config.max_seq_length,
            dropout=self.config.dropout_rate
        )

        # explicit path overrides the default lookup (used by eval_models.py for v2/v3)
        if model_path is not None:
            resolved = Path(model_path)
        else:
            v3_path      = Path(self.config.model_dir) / 'best_model_v3.pth'
            combined_path = Path(self.config.model_dir) / 'best_model_combined.pth'
            if v3_path.exists():
                resolved = v3_path
            elif combined_path.exists():
                resolved = combined_path
            else:
                resolved = Path(self.config.model_dir) / 'best_model.pth'
        if not resolved.exists():
            raise FileNotFoundError(
                f"Trained model not found at {resolved}. "
                f"Run training first: python -m techpack_generator.ml_model.train"
            )

        checkpoint = torch.load(resolved, map_location=self.device)
        old_state  = checkpoint['model_state_dict']
        ckpt_cfg   = checkpoint.get('config', {})

        # infer architecture directly from weight shapes - more reliable than the saved
        # config dict, which is empty when config attributes are class-level (not instance)
        ckpt_d_model    = old_state['encoder.embedding.weight'].shape[1]
        ckpt_enc_layers = sum(1 for k in old_state if k.startswith('encoder.layers.') and k.endswith('.norm1.weight'))
        ckpt_dec_layers = sum(1 for k in old_state if k.startswith('decoder.layers.') and k.endswith('.norm1.weight'))
        ckpt_d_ff       = old_state['encoder.layers.0.feed_forward.linear1.weight'].shape[0]
        ckpt_max_seq    = old_state['encoder.positional_encoding.pe'].shape[1]
        ckpt_vocab_size = old_state['encoder.embedding.weight'].shape[0]
        ckpt_num_heads  = ckpt_cfg.get('num_heads', self.config.num_heads)

        if ckpt_d_model != self.config.d_model or ckpt_enc_layers != self.config.num_encoder_layers:
            self.model = Transformer(
                src_vocab_size=ckpt_vocab_size,
                tgt_vocab_size=ckpt_vocab_size,
                d_model=ckpt_d_model,
                num_encoder_layers=ckpt_enc_layers,
                num_decoder_layers=ckpt_dec_layers,
                num_heads=ckpt_num_heads,
                d_ff=ckpt_d_ff,
                max_seq_length=ckpt_max_seq,
                dropout=ckpt_cfg.get('dropout_rate', self.config.dropout_rate),
            )

        # extend embedding rows if the tokenizer vocab grew since the checkpoint was saved
        new_state = self.model.state_dict()
        merged    = {}
        for key, old_w in old_state.items():
            if key not in new_state:
                continue
            if new_state[key].shape == old_w.shape:
                merged[key] = old_w
            elif new_state[key].dim() >= 1 and new_state[key].shape[0] > old_w.shape[0]:
                new_state[key][:old_w.shape[0]] = old_w
                merged[key] = new_state[key]
            else:
                merged[key] = old_w
        self.model.load_state_dict(merged, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

    # takes a text description, returns a dict with success, tech_pack, confidences, and error info
    def generate(self, user_input, return_raw=False):
        try:
            from .validation import validate_tech_pack
            from .followup import get_follow_up_questions

            input_ids = self.tokenizer.encode(user_input, self.config.max_seq_length)
            src = torch.tensor([input_ids]).to(self.device)

            with torch.no_grad():
                output_ids, token_probs = self.model.generate(
                    src,
                    self.tokenizer,
                    max_length=self.config.max_seq_length,
                    device=self.device,
                    return_probs=True,
                )

            output_text = self.tokenizer.decode(output_ids[0].tolist())
            json_str = output_text.replace(' ', '')
            tech_pack_data = json.loads(json_str)

            confidences = _map_token_confidences(
                output_ids[0].tolist(), token_probs, self.tokenizer
            )

            tech_pack_data, validation_issues = validate_tech_pack(tech_pack_data, confidences)
            follow_ups = get_follow_up_questions(
                tech_pack_data.get('tech_pack', tech_pack_data), confidences,
                user_input=user_input,
            )
            if validation_issues:
                follow_ups = follow_ups + validation_issues

            result = {
                'success': True,
                'tech_pack': tech_pack_data,
                'confidences': confidences,
                'follow_up_questions': follow_ups,
                'input': user_input,
            }
            if return_raw:
                result['raw_output'] = output_text
            return result

        except json.JSONDecodeError as e:
            return {
                'success': False,
                'tech_pack': None,
                'error': f'JSON parsing error: {str(e)}',
                'raw_output': output_text if 'output_text' in locals() else None,
                'input': user_input,
            }
        except Exception as e:
            return {
                'success': False,
                'tech_pack': None,
                'error': f'Generation error: {str(e)}',
                'input': user_input,
            }


# JSON field names we want to track confidence for
_TRACKED_FIELDS = {
    'garment_type', 'fabric_type', 'colour', 'fabric_weight',
    'seam_type', 'closure_type', 'pockets', 'waistband_height',
    'fit', 'length', 'details',
    'chest', 'waist', 'sleeve_length', 'body_length', 'shoulder',
    'inseam', 'outseam', 'rise', 'hips', 'thigh', 'leg_opening',
}

def _map_token_confidences(token_ids: list, token_probs: list, tokenizer) -> dict:
    # for each tracked JSON field, finds the value tokens that follow the key token
    # and takes their geometric mean softmax prob as the field confidence
    tokens        = tokenizer.decode(token_ids, skip_special_tokens=True).split()
    aligned_probs = token_probs[:len(tokens)]

    confidences = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i].lower().strip('"').strip("'")
        if tok in _TRACKED_FIELDS:
            value_probs = []
            j = i + 1
            while j < len(tokens) and tokens[j] not in ('{', '}', ',', ':'):
                value_probs.append(aligned_probs[j] if j < len(aligned_probs) else 1.0)
                j += 1
            if value_probs:
                log_sum = sum(math.log(max(p, 1e-9)) for p in value_probs)
                confidences[tok] = math.exp(log_sum / len(value_probs))
        i += 1

    return confidences


# per-field realistic defaults used when the model doesn't output a value
_MEASUREMENT_DEFAULTS = {
    # tops are full circumference (validators + SVG generator both expect this)
    'chest': 40, 'waist': 36, 'sleeve_length': 25, 'body_length': 28, 'shoulder': 17,
    # bottoms
    'hips': 40, 'inseam': 30, 'outseam': 40, 'rise': 10, 'thigh': 22, 'leg_opening': 14,
}

# garment-type overrides that supersede the generic defaults above
_GARMENT_MEASUREMENT_OVERRIDES = {
    # chest/waist values are full circumference, not half
    't-shirt':    {'chest': 40, 'sleeve_length': 8,  'body_length': 27},
    'tee':        {'chest': 40, 'sleeve_length': 8,  'body_length': 27},
    'polo':       {'chest': 40, 'sleeve_length': 9,  'body_length': 28},
    'tank':       {'chest': 36, 'sleeve_length': 0,  'body_length': 26},
    'singlet':    {'chest': 36, 'sleeve_length': 0,  'body_length': 26},
    'crop top':   {'chest': 36, 'sleeve_length': 8,  'body_length': 20},
    'hoodie':     {'chest': 44, 'sleeve_length': 25, 'body_length': 28},
    'sweatshirt': {'chest': 44, 'sleeve_length': 25, 'body_length': 27},
    'jacket':     {'chest': 44, 'sleeve_length': 25, 'body_length': 27},
    'coat':       {'chest': 44, 'sleeve_length': 26, 'body_length': 40},
    'blazer':     {'chest': 42, 'sleeve_length': 25, 'body_length': 30},
    'shirt':      {'chest': 42, 'sleeve_length': 25, 'body_length': 30},
    'dress':      {'chest': 36, 'sleeve_length': 0,  'body_length': 44},
    # bottoms
    'jeans':     {'waist': 32, 'hips': 40, 'inseam': 30, 'outseam': 40, 'rise': 10, 'thigh': 22, 'leg_opening': 14},
    'trousers':  {'waist': 32, 'hips': 40, 'inseam': 30, 'outseam': 41, 'rise': 10, 'thigh': 22, 'leg_opening': 16},
    'chinos':    {'waist': 32, 'hips': 40, 'inseam': 30, 'outseam': 40, 'rise': 10, 'thigh': 22, 'leg_opening': 15},
    'shorts':    {'waist': 30, 'hips': 38, 'inseam': 8,  'outseam': 18, 'rise': 10, 'thigh': 24, 'leg_opening': 22},
    'joggers':   {'waist': 30, 'hips': 40, 'inseam': 28, 'outseam': 38, 'rise': 11, 'thigh': 24, 'leg_opening': 10},
    'leggings':  {'waist': 28, 'hips': 36, 'inseam': 26, 'outseam': 36, 'rise': 9,  'thigh': 20, 'leg_opening': 8},
    'skirt':     {'waist': 28, 'hips': 38, 'inseam': 22, 'rise': 8},
    'culottes':  {'waist': 30, 'hips': 40, 'inseam': 14, 'outseam': 24, 'rise': 10, 'thigh': 26, 'leg_opening': 24},
    'sweatpants':{'waist': 30, 'hips': 40, 'inseam': 28, 'outseam': 38, 'rise': 11, 'thigh': 24, 'leg_opening': 10},
}

def _garment_meas_defaults(garment_type: str) -> dict:
    g = garment_type.lower()
    for key, overrides in _GARMENT_MEASUREMENT_OVERRIDES.items():
        if key in g:
            return overrides
    return {}

def safe_int(value, default=38):
    try:
        v = int(value)
        return v if v > 0 else default
    except (ValueError, TypeError):
        return default


# extract an explicit measurement from user input, e.g. "38 inch chest" → {'chest': 38}
_EXPLICIT_MEAS_RE = re.compile(
    r'(\d{1,3})\s*(?:inch(?:es?)?|in\b|")\s*(chest|waist|inseam|sleeve|shoulder|body[\s_]?length)',
    re.IGNORECASE,
)
_INPUT_FIELD_MAP = {
    'chest': 'chest', 'waist': 'waist', 'inseam': 'inseam',
    'sleeve': 'sleeve_length', 'shoulder': 'shoulder', 'body': 'body_length',
}

def _parse_explicit_measurements(user_input: str) -> dict:
    if not user_input:
        return {}
    found = {}
    for m in _EXPLICIT_MEAS_RE.finditer(user_input):
        val, label = int(m.group(1)), m.group(2).lower().split()[0]
        field = _INPUT_FIELD_MAP.get(label)
        if field:
            found[field] = val
    return found


# model sometimes puts words together, this fixes the common ones
COMPOUND_REPLACEMENTS = {
    'leatherjacket': 'leather jacket',
    'denimjacket': 'denim jacket',
    'woolblend': 'wool blend',
    'cottonblend': 'cotton blend',
    'navyblue': 'navy blue',
    'charcoalgrey': 'charcoal grey',
    'darkgreen': 'dark green',
    'forestgreen': 'forest green',
    'olivegreen': 'olive green',
    'sagegreen': 'sage green',
    'lightblue': 'light blue',
    'skyblue': 'sky blue',
    'royalblue': 'royal blue',
    'lightgrey': 'light grey',
    'darkgrey': 'dark grey',
    'ponteknit': 'ponte knit',
    'ribknit': 'rib knit',
    'woolcoat': 'wool coat',
    'silkblend': 'silk blend',
    'cottonfleece': 'cotton fleece',
    'woolmelton': 'wool melton',
    'silksatin': 'silk satin',
    'eveninggown': 'evening gown',
    'mididress': 'midi dress',
    'darkwash' : 'dark wash',
}


# splits joined compound words back apart
def clean_compound(text):
    if not text:
        return text
    text_lower = text.lower()
    for compound, spaced in COMPOUND_REPLACEMENTS.items():
        if compound in text_lower:
            return spaced
    # handle camelCase too
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', text)


# detect keywords from user input that the model tends to ignore
FIT_KEYWORDS = ['slim', 'fitted', 'regular', 'relaxed', 'loose', 'baggy', 'boxy', 'oversized']
LENGTH_KEYWORDS = ['cropped', 'short', 'long', 'longline']

# compound colours checked first (longest match wins), then single-word
COLOUR_KEYWORDS = [
    'forest green', 'olive green', 'sage green', 'dark green',
    'navy blue', 'light blue', 'sky blue', 'royal blue',
    'charcoal grey', 'light grey', 'dark grey',
    'black', 'white', 'navy', 'red', 'blue', 'green', 'grey', 'gray',
    'burgundy', 'cream', 'charcoal', 'emerald', 'beige', 'tan', 'brown',
    'pink', 'coral', 'mustard', 'yellow', 'orange', 'purple', 'lavender',
    'maroon', 'teal', 'khaki', 'ivory', 'camel', 'olive',
]

_LONG_SLEEVE_RE = re.compile(r'long[\s-]sleeve', re.IGNORECASE)
_FABRIC_WEIGHT_RE = re.compile(r'(\d+)\s*(gsm|g/m2|g\/m2|oz)\b', re.IGNORECASE)

# garments that are naturally short-sleeved but can be ordered long-sleeve
_SHORT_SLEEVE_GARMENTS = {'t-shirt', 'tee', 'polo', 'tank top', 'tank', 'crop top'}


def _parse_explicit_fabric_weight(user_input: str):
    if not user_input:
        return None
    m = _FABRIC_WEIGHT_RE.search(user_input)
    return f"{m.group(1)}{m.group(2).lower()}" if m else None


def _detect_from_input(user_input, keywords):
    if not user_input:
        return None
    lower = user_input.lower()
    for kw in keywords:
        if kw in lower:
            return kw
    return None


def _detect_length_from_input(user_input, keywords):
    if not user_input:
        return None
    # strip "long sleeve" first so it doesn't falsely match body length = 'long'
    clean = _LONG_SLEEVE_RE.sub('', user_input)
    lower = clean.lower()
    for kw in keywords:
        if kw in lower:
            return kw
    return None


# how fit affects width measurements (chest, waist, shoulder)
# relative to regular (1.0), applied on top of whatever the model outputs
FIT_MEASUREMENT_SCALE = {
    'slim': 0.88,
    'fitted': 0.93,
    'regular': 1.0,
    'relaxed': 1.05,
    'loose': 1.1,
    'baggy': 1.15,
    'boxy': 1.12,
    'oversized': 1.18,
}


# pulls out fields from model output, uses user input as ground truth where the model is weak
def extract_tech_pack_fields(tech_pack_data, user_input=None):
    tp = tech_pack_data.get('tech_pack', tech_pack_data)

    style = tp['style']
    material = tp['material']
    measurements = tp['measurements']
    construction = tp['construction']

    # model often ignores fit/length/colour, override from user input if present
    detected_fit    = _detect_from_input(user_input, FIT_KEYWORDS)
    detected_length = _detect_length_from_input(user_input, LENGTH_KEYWORDS)
    detected_colour = _detect_from_input(user_input, COLOUR_KEYWORDS)
    explicit_fw     = _parse_explicit_fabric_weight(user_input)

    model_colour = clean_compound(material['colour'])
    fit = detected_fit or style['fit']

    garment_type = clean_compound(tp['garment_type'])
    garment_overrides = _garment_meas_defaults(garment_type)

    def _m(field):
        override = garment_overrides.get(field)
        default  = override if override is not None else _MEASUREMENT_DEFAULTS.get(field, 38)
        return safe_int(measurements.get(field), default)

    chest         = _m('chest')
    waist         = _m('waist')
    shoulder      = _m('shoulder')
    sleeve_length = _m('sleeve_length')
    body_length   = _m('body_length')
    hips          = _m('hips')
    inseam        = _m('inseam')
    outseam       = _m('outseam')
    rise          = _m('rise')
    thigh         = _m('thigh')
    leg_opening   = _m('leg_opening')

    # apply fit-based width scaling to measurements so stored numbers reflect the fit
    fit_scale = FIT_MEASUREMENT_SCALE.get(fit, 1.0)
    if fit_scale != 1.0:
        chest    = round(chest    * fit_scale)
        waist    = round(waist    * fit_scale)
        shoulder = round(shoulder * fit_scale)
        thigh    = round(thigh    * fit_scale)
        hips     = round(hips     * fit_scale)

    # explicit measurements in user input are ground-truth, override model output
    explicit = _parse_explicit_measurements(user_input)
    chest         = explicit.get('chest',         chest)
    waist         = explicit.get('waist',         waist)
    shoulder      = explicit.get('shoulder',      shoulder)
    sleeve_length = explicit.get('sleeve_length', sleeve_length)
    body_length   = explicit.get('body_length',   body_length)
    inseam        = explicit.get('inseam',        inseam)

    # "long sleeve t-shirt" etc: bump sleeve to full shirt length
    if _LONG_SLEEVE_RE.search(user_input or ''):
        if any(g in garment_type.lower() for g in _SHORT_SLEEVE_GARMENTS):
            sleeve_length = 25

    return {
        'garment_type':  garment_type,
        'fabric_type':   clean_compound(material['fabric_type']),
        'colour':        detected_colour or model_colour or 'black',
        'fabric_weight': explicit_fw or material['fabric_weight'],
        'chest':         chest,
        'waist':         waist,
        'sleeve_length': sleeve_length,
        'body_length':   body_length,
        'shoulder':      shoulder,
        'hips':          hips,
        'inseam':        inseam,
        'outseam':       outseam,
        'rise':          rise,
        'thigh':         thigh,
        'leg_opening':   leg_opening,
        'seam_type':     construction['seam_type'],
        'closure_type':  construction['closure_type'],
        'pockets':       construction['pockets'],
        'fit':           fit,
        'length':        detected_length or style['length'],
        'details':       style['details'],
    }
