# Examines a generated tech pack and its field-level confidences to decide
# whether to ask the user any follow-up questions before saving.

CONFIDENCE_THRESHOLD = 0.5  

# fields that have an obvious correct value for a given garment type
# if a field appears here for the garment, we don't ask about it even if
# the model was uncertain - just use the default silently
GARMENT_DEFAULTS = {
    'jeans':    {'fabric_type': 'denim', 'closure_type': 'zip fly'},
    'chinos':   {'fabric_type': 'cotton twill'},
    'trousers': {'seam_type': 'flat fell'},
    'hoodie':   {'fabric_type': 'cotton fleece'},   # no closure_type, hoodie can be pullover or zip so we ask
    'sweatshirt': {'fabric_type': 'cotton fleece', 'closure_type': 'none'},
    't-shirt':  {'closure_type': 'none', 'seam_type': 'overlock'},
    'shorts':   {'fabric_type': 'cotton'},
    'joggers':  {'fabric_type': 'cotton fleece', 'closure_type': 'drawstring'},
    'leggings': {'fabric_type': 'cotton lycra', 'closure_type': 'none'},
    'blazer':   {'seam_type': 'welt'},
    'coat':     {'seam_type': 'welt'},
}

# Garment-specific question overrides (take priority over QUESTION_TEMPLATES)
GARMENT_QUESTIONS = {
    'hoodie': {
        'closure_type': 'Is this a zip-up or pullover hoodie?',
    },
}

# only these fields are ambiguous enough to warrant asking the user
AMBIGUOUS_FIELDS = [
    'colour', 'fit', 'fabric_weight', 'fabric_type',
    'closure_type', 'pockets', 'length',
]

QUESTION_TEMPLATES = {
    'colour':       'What colour would you like?',
    'fit':          'Should the fit be slim, regular, or relaxed?',
    'fabric_weight': 'What fabric weight do you have in mind (e.g. 180gsm, 12oz)?',
    'fabric_type':  'What fabric would you like (e.g. cotton, linen, denim)?',
    'closure_type': 'What closure type would you like (e.g. zip, button, none)?',
    'pockets':      'What pocket style would you like (e.g. patch, welt, none)?',
    'length':       'What length would you prefer (e.g. cropped, regular, long)?',
}

_COLOUR_KEYWORDS = {
    'forest green', 'olive green', 'sage green', 'dark green',
    'navy blue', 'light blue', 'sky blue', 'royal blue',
    'charcoal grey', 'light grey', 'dark grey',
    'black', 'white', 'navy', 'red', 'blue', 'green', 'grey', 'gray',
    'burgundy', 'cream', 'charcoal', 'emerald', 'beige', 'tan', 'brown',
    'pink', 'coral', 'mustard', 'yellow', 'orange', 'purple', 'lavender',
    'maroon', 'teal', 'khaki', 'ivory', 'camel', 'olive',
}


def _user_mentioned_colour(user_input: str) -> bool:
    if not user_input:
        return False
    lower = user_input.lower()
    for kw in sorted(_COLOUR_KEYWORDS, key=len, reverse=True):
        if kw in lower:
            return True
    return False


# Fields that must always be asked for specific garments unless the user already
# mentioned one of the listed keywords (works like the colour check above)
GARMENT_ALWAYS_ASK = {
    'hoodie': {
        'closure_type': ('zip', 'full zip', 'half zip', 'pullover', 'pull over'),
    },
}


def _user_mentioned_any(user_input: str, keywords: tuple) -> bool:
    if not user_input:
        return False
    lower = user_input.lower()
    return any(kw in lower for kw in keywords)


def get_follow_up_questions(tech_pack: dict, confidences: dict,
                            threshold: float = CONFIDENCE_THRESHOLD,
                            user_input: str = None) -> list:
    garment = tech_pack['garment_type'].lower()
    defaults  = GARMENT_DEFAULTS.get(garment, {})
    garment_q = GARMENT_QUESTIONS.get(garment, {})
    always_ask = GARMENT_ALWAYS_ASK.get(garment, {})

    questions = []
    for field in AMBIGUOUS_FIELDS:
        if field in defaults:
            continue

        # colour: always ask if the user didn't mention one
        if field == 'colour' and not _user_mentioned_colour(user_input):
            questions.append(garment_q.get(field) or QUESTION_TEMPLATES['colour'])
            continue

        # garment-specific always-ask fields (e.g. hoodie closure_type)
        if field in always_ask and not _user_mentioned_any(user_input, always_ask[field]):
            questions.append(garment_q.get(field) or QUESTION_TEMPLATES[field])
            continue

        conf = confidences.get(field, 1.0)
        if conf < threshold and field in QUESTION_TEMPLATES:
            questions.append(garment_q.get(field) or QUESTION_TEMPLATES[field])

    return questions


# reverse lookup used by views.py to build the follow-up form and map answers back
QUESTION_TO_FIELD = {v: k for k, v in QUESTION_TEMPLATES.items()}
# include garment-specific question overrides in the reverse lookup
for _gq in GARMENT_QUESTIONS.values():
    for _field, _q in _gq.items():
        QUESTION_TO_FIELD[_q] = _field


def build_enriched_description(original: str, answered_fields: dict) -> str:
    # combines the original user description with their follow-up answers into
    # a single string that can be re-run through generate() so the model handles
    # any colour/fabric/fit the user types, not just ones in a keyword list
    if not answered_fields:
        return original
    details = ', '.join(f"{field} {value}" for field, value in answered_fields.items() if value)
    return f"{original}, {details}"


# maps follow-up field names to their location inside the tech pack dict
_FIELD_PATHS = {
    'colour':        ('material',     'colour'),
    'fabric_type':   ('material',     'fabric_type'),
    'fabric_weight': ('material',     'fabric_weight'),
    'fit':           ('style',        'fit'),
    'length':        ('style',        'length'),
    'closure_type':  ('construction', 'closure_type'),
    'pockets':       ('construction', 'pockets'),
}


def apply_followup_answers(tech_pack_data: dict, answered_fields: dict) -> dict:
    #Write follow-up answers directly into the stored tech pack instead of regenerating.
    tp = tech_pack_data.get('tech_pack', tech_pack_data)
    for field, value in answered_fields.items():
        if not value:
            continue
        path = _FIELD_PATHS.get(field)
        if path:
            section, key = path
            if section not in tp:
                tp[section] = {}
            tp[section][key] = value
    if 'tech_pack' in tech_pack_data:
        tech_pack_data['tech_pack'] = tp
        return tech_pack_data
    return {'tech_pack': tp}


def get_remaining_questions(tech_pack: dict, user_answered_fields: set = None) -> list:
    #return questions for ambigious questions

    garment = tech_pack['garment_type'].lower()
    defaults = GARMENT_DEFAULTS.get(garment, {})
    mat = tech_pack['material']
    sty = tech_pack['style']
    con = tech_pack['construction']
    current = {
        'colour':        mat['colour'],
        'fit':           sty['fit'],
        'fabric_weight': mat['fabric_weight'],
        'fabric_type':   mat['fabric_type'],
        'closure_type':  con['closure_type'],
        'pockets':       con['pockets'],
        'length':        sty['length'],
    }
    questions = []
    for field in AMBIGUOUS_FIELDS:
        if field in defaults:
            continue
        if user_answered_fields and field in user_answered_fields:
            continue
        val = current.get(field, '')
        if not val or str(val).lower() in ('', 'null', 'unknown', 'n/a'):
            questions.append({'field': field, 'question': QUESTION_TEMPLATES[field]})
    return questions


def apply_garment_defaults(tech_pack: dict) -> dict:
    # fills in obvious field values from garment type when the model left them blank
    garment = tech_pack['garment_type'].lower()
    defaults = GARMENT_DEFAULTS.get(garment, {})
    material = tech_pack['material']
    construction = tech_pack['construction']

    for field, value in defaults.items():
        if field in ('fabric_type',) and not material.get(field):
            material[field] = value
        elif field in ('closure_type', 'seam_type') and not construction.get(field):
            construction[field] = value

    if material:
        tech_pack['material'] = material
    if construction:
        tech_pack['construction'] = construction

    return tech_pack
