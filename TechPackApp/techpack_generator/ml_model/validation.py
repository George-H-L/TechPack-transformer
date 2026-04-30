# Validates and fixes generated tech packs before they reach the user.

import re

TOPS_GARMENTS = {
    't-shirt', 'shirt', 'jacket', 'hoodie', 'sweatshirt', 'blazer',
    'coat', 'cardigan', 'jumper', 'sweater', 'vest', 'crop top',
    'blouse', 'polo', 'tank top', 'leather jacket', 'denim jacket',
}
BOTTOMS_GARMENTS = {
    'jeans', 'trousers', 'chinos', 'shorts', 'joggers', 'skirt',
    'leggings', 'cargo trousers', 'cargo shorts', 'sweatpants', 'culottes',
}

MEASUREMENT_RANGES = {
    'tops': {
        'chest':         (28, 60),
        'waist':         (24, 56),
        'sleeve_length': (4,  38),
        'body_length':   (16, 42),
        'shoulder':      (12, 26),
    },
    'bottoms': {
        'waist':       (22, 50),
        'hips':        (28, 56),
        'inseam':      (5,  38),
        'outseam':     (10, 50),
        'rise':        (6,  16),
        'thigh':       (16, 34),
        'leg_opening': (6,  28),
    },
}

KNOWN_FABRICS = {
    'cotton', 'denim', 'linen', 'wool', 'polyester', 'silk', 'leather',
    'twill', 'corduroy', 'fleece', 'jersey', 'canvas', 'velvet', 'satin',
    'chiffon', 'spandex', 'lycra', 'nylon', 'rayon', 'viscose', 'suede',
    'faux leather', 'cotton fleece', 'wool blend', 'cotton twill',
    'cotton lycra', 'ponte knit', 'rib knit',
}
KNOWN_COLOURS = {
    'black', 'white', 'navy', 'red', 'blue', 'green', 'grey', 'gray',
    'burgundy', 'cream', 'charcoal', 'emerald', 'beige', 'tan', 'brown',
    'pink', 'coral', 'mustard', 'yellow', 'orange', 'purple', 'lavender',
    'maroon', 'teal', 'khaki', 'ivory', 'camel', 'olive', 'sage',
    'forest green', 'olive green', 'sage green', 'dark green', 'navy blue',
    'light blue', 'sky blue', 'royal blue', 'charcoal grey', 'light grey',
    'dark grey',
}

FABRIC_WEIGHT_PATTERN = re.compile(r'\d+\s*(gsm|oz|g\/m|g/m2|mm)', re.IGNORECASE)


def _garment_schema(garment_type: str) -> str:
    gt = garment_type.lower()
    if any(g in gt for g in BOTTOMS_GARMENTS):
        return 'bottoms'
    if any(g in gt for g in TOPS_GARMENTS):
        return 'tops'
    return 'unknown'


def validate_tech_pack(tech_pack_data: dict, confidences: dict = None) -> tuple:
    tp = tech_pack_data.get('tech_pack', tech_pack_data)
    material     = tp['material']
    measurements = tp['measurements']
    construction = tp['construction']
    style        = tp['style']

    ambiguous_questions = []

    # fabric/colour swap check
    fabric = (material['fabric_type'] or '').lower().strip()
    colour = (material['colour'] or '').lower().strip()

    if fabric and colour:
        fabric_looks_like_colour = fabric in KNOWN_COLOURS
        colour_looks_like_fabric = colour in KNOWN_FABRICS
        if fabric_looks_like_colour and colour_looks_like_fabric:
            material['fabric_type'], material['colour'] = material['colour'], material['fabric_type']
        elif fabric_looks_like_colour:
            ambiguous_questions.append('What fabric would you like? (the model may have mixed up fabric and colour)')
        elif colour_looks_like_fabric:
            ambiguous_questions.append('What colour would you like? (the model may have mixed up fabric and colour)')

    # clamp measurements to plausible ranges for the garment type
    schema = _garment_schema(tp['garment_type'])
    if schema != 'unknown':
        ranges = MEASUREMENT_RANGES[schema]
        for field, (lo, hi) in ranges.items():
            val = measurements.get(field)
            if val is None:
                continue
            try:
                v = int(val)
            except (TypeError, ValueError):
                continue
            if not (lo <= v <= hi):
                measurements[field] = max(lo, min(hi, v))

    # bottoms consistency: inseam < outseam, and rise + inseam ≈ outseam
    inseam  = measurements.get('inseam')
    outseam = measurements.get('outseam')
    rise    = measurements.get('rise')
    if inseam is not None and outseam is not None:
        try:
            i, o = int(inseam), int(outseam)
            if i >= o:
                measurements['inseam'] = max(5, o - 4)
        except (TypeError, ValueError):
            pass
    if inseam is not None and outseam is not None and rise is not None:
        try:
            i, o, r = int(inseam), int(outseam), int(rise)
            if abs((r + i) - o) > 4:
                measurements['outseam'] = r + i
        except (TypeError, ValueError):
            pass

    # fabric weight needs a unit - if it's just a bare number assume gsm
    fw = material['fabric_weight']
    if fw and not FABRIC_WEIGHT_PATTERN.search(str(fw)):
        if str(fw).strip().isdigit():
            material['fabric_weight'] = f"{fw}gsm"
        else:
            ambiguous_questions.append('What fabric weight would you like? Please include a unit (e.g. 180gsm, 12oz).')

    details = style['details']
    if not isinstance(details, list) or len(details) == 0:
        style['details'] = ['standard finish']
    elif len(details) > 4:
        style['details'] = details[:4]

    tp['material']     = material
    tp['measurements'] = measurements
    tp['construction'] = construction
    tp['style']        = style

    if 'tech_pack' in tech_pack_data:
        tech_pack_data['tech_pack'] = tp
    else:
        tech_pack_data = tp

    return tech_pack_data, ambiguous_questions
