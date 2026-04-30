import sys
import json
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from techpack_generator.ml_model.inference  import TechPackGenerator, extract_tech_pack_fields
from techpack_generator.ml_model.config     import ModelConfig
from techpack_generator.ml_model.validation import MEASUREMENT_RANGES

config = ModelConfig()

TEST_CASES = [
    {'prompt': 'black cotton t-shirt slim fit',
     'exp_garment': 't-shirt',   'schema': 'tops',
     'exp': {'colour': 'black',    'fabric_type': 'cotton',  'fit': 'slim'}},

    {'prompt': 'white linen shirt regular fit',
     'exp_garment': 'shirt',     'schema': 'tops',
     'exp': {'colour': 'white',    'fabric_type': 'linen',   'fit': 'regular'}},

    {'prompt': 'navy wool blazer slim fit',
     'exp_garment': 'blazer',    'schema': 'tops',
     'exp': {'colour': 'navy',     'fabric_type': 'wool',    'fit': 'slim'}},

    {'prompt': 'grey cotton hoodie oversized fit',
     'exp_garment': 'hoodie',    'schema': 'tops',
     'exp': {'colour': 'grey',     'fabric_type': 'cotton',  'fit': 'oversized'}},

    {'prompt': 'red leather jacket fitted',
     'exp_garment': 'jacket',    'schema': 'tops',
     'exp': {'colour': 'red',      'fabric_type': 'leather', 'fit': 'fitted'}},

    {'prompt': 'cream wool coat long',
     'exp_garment': 'coat',      'schema': 'tops',
     'exp': {'colour': 'cream',    'fabric_type': 'wool',    'length': 'long'}},

    {'prompt': 'burgundy silk midi dress fitted',
     'exp_garment': 'dress',     'schema': 'tops',
     'exp': {'colour': 'burgundy', 'fabric_type': 'silk',    'fit': 'fitted'}},

    {'prompt': 'charcoal cotton sweatshirt relaxed fit',
     'exp_garment': 'sweatshirt','schema': 'tops',
     'exp': {'colour': 'charcoal', 'fabric_type': 'cotton',  'fit': 'relaxed'}},

    {'prompt': 'olive green linen shirt loose fit',
     'exp_garment': 'shirt',     'schema': 'tops',
     'exp': {'colour': 'olive',    'fabric_type': 'linen',   'fit': 'loose'}},

    {'prompt': 'black polyester puffer jacket',
     'exp_garment': 'jacket',    'schema': 'tops',
     'exp': {'colour': 'black',    'fabric_type': 'polyester'}},

    {'prompt': 'black cotton cropped hoodie',
     'exp_garment': 'hoodie',    'schema': 'tops',
     'exp': {'colour': 'black',    'fabric_type': 'cotton',  'length': 'cropped'}},

    {'prompt': 'navy wool longline coat',
     'exp_garment': 'coat',      'schema': 'tops',
     'exp': {'colour': 'navy',     'fabric_type': 'wool',    'length': 'long'}},

    {'prompt': 'black zip-up cotton hoodie',
     'exp_garment': 'hoodie',    'schema': 'tops',
     'exp': {'colour': 'black',    'fabric_type': 'cotton',  'closure_type': 'zip'}},

    {'prompt': 'white button-up cotton shirt',
     'exp_garment': 'shirt',     'schema': 'tops',
     'exp': {'colour': 'white',    'fabric_type': 'cotton',  'closure_type': 'button'}},

    {'prompt': 'heavyweight black cotton hoodie 400gsm',
     'exp_garment': 'hoodie',    'schema': 'tops',
     'exp': {'colour': 'black',    'fabric_type': 'cotton',  'fabric_weight': '400'}},

    {'prompt': 'lightweight white linen shirt 120gsm',
     'exp_garment': 'shirt',     'schema': 'tops',
     'exp': {'colour': 'white',    'fabric_type': 'linen',   'fabric_weight': '120'}},

    {'prompt': 'dark blue denim jeans slim fit',
     'exp_garment': 'jeans',     'schema': 'bottoms',
     'exp': {'fabric_type': 'denim', 'fit': 'slim'}},

    {'prompt': 'beige cotton chinos regular fit',
     'exp_garment': 'chinos',    'schema': 'bottoms',
     'exp': {'colour': 'beige',    'fabric_type': 'cotton',  'fit': 'regular'}},

    {'prompt': 'black cotton joggers relaxed fit with drawstring',
     'exp_garment': 'joggers',   'schema': 'bottoms',
     'exp': {'colour': 'black',    'fabric_type': 'cotton',  'fit': 'relaxed',
             'closure_type': 'drawstring'}},

    {'prompt': 'olive green cargo trousers regular fit with patch pockets',
     'exp_garment': 'trousers',  'schema': 'bottoms',
     'exp': {'fabric_type': 'cotton', 'fit': 'regular', 'pockets': 'patch'}},

    {'prompt': 'navy blue cotton shorts relaxed fit',
     'exp_garment': 'shorts',    'schema': 'bottoms',
     'exp': {'colour': 'navy',     'fabric_type': 'cotton',  'fit': 'relaxed'}},

    {'prompt': 'black denim jeans with zip fly and welt pockets',
     'exp_garment': 'jeans',     'schema': 'bottoms',
     'exp': {'colour': 'black',    'fabric_type': 'denim',   'closure_type': 'zip',
             'pockets': 'welt'}},

    {'prompt': 'grey wool trousers slim fit with flat fell seam',
     'exp_garment': 'trousers',  'schema': 'bottoms',
     'exp': {'colour': 'grey',     'fabric_type': 'wool',    'fit': 'slim',
             'seam_type': 'flat'}},

    {'prompt': 'burgundy cotton leggings fitted',
     'exp_garment': 'leggings',  'schema': 'bottoms',
     'exp': {'colour': 'burgundy', 'fabric_type': 'cotton',  'fit': 'fitted'}},
]

FIELD_LOCATION = {
    'garment_type':  ('root',         'garment_type'),
    'colour':        ('material',     'colour'),
    'fabric_type':   ('material',     'fabric_type'),
    'fabric_weight': ('material',     'fabric_weight'),
    'fit':           ('style',        'fit'),
    'length':        ('style',        'length'),
    'seam_type':     ('construction', 'seam_type'),
    'closure_type':  ('construction', 'closure_type'),
    'pockets':       ('construction', 'pockets'),
}

FLAT_FIELD_MAP = {
    'garment_type': 'garment_type',
    'colour':       'colour',
    'fabric_type':  'fabric_type',
    'fabric_weight':'fabric_weight',
    'fit':          'fit',
    'length':       'length',
    'seam_type':    'seam_type',
    'closure_type': 'closure_type',
    'pockets':      'pockets',
}

MEAS_FLAT = {
    'tops':    ['chest', 'waist', 'sleeve_length', 'body_length', 'shoulder'],
    'bottoms': ['waist', 'hips', 'inseam', 'outseam', 'rise', 'thigh', 'leg_opening'],
}

VARIANTS = [
    ('v1', Path(config.model_dir) / 'best_model_combined.pth'),
    ('v2', Path(config.model_dir) / 'best_model_v2.pth'),
    ('v3', Path(config.model_dir) / 'best_model_v3.pth'),
    ('v4', Path(config.model_dir) / 'best_model_v4.pth'),
]


def _unwrap(result):
    data = result['tech_pack']
    return data.get('tech_pack', data)


def _raw_val(tp, field):
    loc, key = FIELD_LOCATION[field]
    if loc == 'root':
        return (tp.get(key) or '').lower().strip()
    section = tp.get(loc) or {}
    return (section.get(key) or '').lower().strip()


def _hit(actual, kw):
    return bool(actual) and kw.lower() in actual


def _score_meas_raw(tp, schema):
    ranges = MEASUREMENT_RANGES.get(schema, {})
    meas   = tp.get('measurements') or {}
    out = {}
    for field, (lo, hi) in ranges.items():
        try:
            v = int(meas.get(field))
            out[field] = (v > 0) and (lo <= v <= hi)
        except (TypeError, ValueError):
            out[field] = False
    return out


def _score_meas_post(flat, schema):
    ranges = MEASUREMENT_RANGES.get(schema, {})
    out = {}
    for field, (lo, hi) in ranges.items():
        try:
            v = int(flat.get(field, 0))
            out[field] = (v > 0) and (lo <= v <= hi)
        except (TypeError, ValueError):
            out[field] = False
    return out


def run():
    generators = {}
    for name, path in VARIANTS:
        if path.exists():
            try:
                generators[name] = TechPackGenerator(model_path=str(path))
                print(f"Loaded {name}")
            except Exception as e:
                print(f"  {name}: {e}")
        else:
            print(f"  {name}: not found")

    if not generators:
        sys.exit("No models loaded.")

    names = list(generators.keys())

    pre_text  = {n: defaultdict(lambda: [0, 0]) for n in names}
    post_text = {n: defaultdict(lambda: [0, 0]) for n in names}
    pre_meas  = {n: {'tops': defaultdict(lambda: [0, 0]),
                     'bottoms': defaultdict(lambda: [0, 0])} for n in names}
    post_meas = {n: {'tops': defaultdict(lambda: [0, 0]),
                     'bottoms': defaultdict(lambda: [0, 0])} for n in names}

    print(f"\nRunning {len(TEST_CASES)} cases x {len(generators)} models...\n")

    for tc in TEST_CASES:
        prompt      = tc['prompt']
        exp_garment = tc['exp_garment']
        schema      = tc['schema']
        expected    = tc['exp']

        for name, gen in generators.items():
            result = gen.generate(prompt)
            if not result['success']:
                for d in (pre_text[name], post_text[name]):
                    d['garment_type'][1] += 1
                    for f in expected:
                        d[f][1] += 1
                continue

            tp   = _unwrap(result)
            flat = extract_tech_pack_fields(result['tech_pack'], user_input=prompt)

            # garment type
            for d, val in ((pre_text[name],  _raw_val(tp, 'garment_type')),
                           (post_text[name], (flat.get('garment_type') or '').lower())):
                d['garment_type'][1] += 1
                if _hit(val, exp_garment):
                    d['garment_type'][0] += 1

            # expected fields
            for field, kw in expected.items():
                pre_val  = _raw_val(tp, field)
                post_val = (flat.get(FLAT_FIELD_MAP.get(field, field)) or '').lower()
                for d, val in ((pre_text[name], pre_val), (post_text[name], post_val)):
                    d[field][1] += 1
                    if _hit(val, kw):
                        d[field][0] += 1

            # measurements
            for field, ok in _score_meas_raw(tp, schema).items():
                pre_meas[name][schema][field][1] += 1
                if ok:
                    pre_meas[name][schema][field][0] += 1

            for field, ok in _score_meas_post(flat, schema).items():
                post_meas[name][schema][field][1] += 1
                if ok:
                    post_meas[name][schema][field][0] += 1

    save_image(pre_text,  pre_meas,  names, 'PRE-PROCESSING  (raw model output)',
               Path(config.model_dir) / 'eval_table_pre.png')
    save_image(post_text, post_meas, names, 'POST-PROCESSING  (after defaults + keyword rules)',
               Path(config.model_dir) / 'eval_table_post.png')

    print_table(pre_text,  pre_meas,  names, 'PRE')
    print_table(post_text, post_meas, names, 'POST')

    out = {
        'pre':  serialise(pre_text,  pre_meas,  names),
        'post': serialise(post_text, post_meas, names),
    }
    save_path = Path(config.model_dir) / 'eval_compare.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved JSON -> {save_path}")
    print(f"Saved PRE  image -> {Path(config.model_dir) / 'eval_table_pre.png'}")
    print(f"Saved POST image -> {Path(config.model_dir) / 'eval_table_post.png'}")


def _pct(c, t):
    return round(100 * c / t) if t else 0


def _cell(c, t):
    if t == 0:
        return 'n/a'
    return f"{c}/{t}  ({_pct(c,t)}%)"


def _colour(pct):
    if pct >= 80:
        return '#c8f0c8'
    if pct >= 60:
        return '#fff3b0'
    if pct >= 40:
        return '#ffd9a0'
    return '#ffb3b3'


TEXT_FIELDS = [
    'garment_type', 'colour', 'fabric_type', 'fabric_weight',
    'fit', 'length', 'seam_type', 'closure_type', 'pockets',
]


def _build_rows(text_scores, meas_scores, names):
    rows = []

    rows.append(('--- Text fields ---', [''] * len(names), [None] * len(names)))
    for field in TEXT_FIELDS:
        cells, colours = [], []
        for n in names:
            c, t = text_scores[n][field]
            pct  = _pct(c, t)
            cells.append(_cell(c, t))
            colours.append(_colour(pct) if t > 0 else '#eeeeee')
        rows.append((field, cells, colours))

    for schema in ('tops', 'bottoms'):
        ranges = MEASUREMENT_RANGES.get(schema, {})
        if not ranges:
            continue
        rows.append((f'--- Measurements [{schema}] ---', [''] * len(names), [None] * len(names)))
        for field in ranges:
            cells, colours = [], []
            for n in names:
                c, t = meas_scores[n][schema][field]
                pct  = _pct(c, t)
                cells.append(_cell(c, t))
                colours.append(_colour(pct) if t > 0 else '#eeeeee')
            rows.append((field, cells, colours))

    return rows


def save_image(text_scores, meas_scores, names, title, path):
    rows = _build_rows(text_scores, meas_scores, names)

    n_rows = len(rows) + 1
    n_cols = len(names) + 1

    fig_h = max(6, n_rows * 0.42)
    fig, ax = plt.subplots(figsize=(4 + len(names) * 2.2, fig_h))
    ax.axis('off')

    fig.patch.set_facecolor('#f4f4f4')
    ax.set_facecolor('#f4f4f4')

    col_labels = ['Field'] + [n.upper() for n in names]
    cell_text  = []
    cell_colour = []

    header_bg = '#2c3e50'

    for label, cells, colours in rows:
        is_section = label.startswith('---')
        if is_section:
            cell_text.append([label.replace('---', '').strip()] + [''] * len(names))
            cell_colour.append(['#dde3ea'] * n_cols)
        else:
            cell_text.append([label] + cells)
            row_colours = ['#f0f0f0'] + colours
            cell_colour.append(row_colours)

    col_widths = [0.26] + [0.18] * len(names)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        cellColours=cell_colour,
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.35)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#cccccc')
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor(header_bg)
            cell.set_text_props(color='white', fontweight='bold')
        elif c == 0 and cell_text[r - 1][0].startswith('---') is False:
            cell.set_text_props(ha='left')
            cell.PAD = 0.08

    ax.set_title(title, fontsize=12, fontweight='bold', pad=14, color='#2c3e50')

    legend_items = [
        mpatches.Patch(color='#c8f0c8', label='>= 80%'),
        mpatches.Patch(color='#fff3b0', label='60-79%'),
        mpatches.Patch(color='#ffd9a0', label='40-59%'),
        mpatches.Patch(color='#ffb3b3', label='< 40%'),
    ]
    ax.legend(handles=legend_items, loc='lower right', fontsize=8,
              framealpha=0.85, bbox_to_anchor=(1.0, 0.0))

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def print_table(text_scores, meas_scores, names, label):
    cw = 14
    print(f"\n{label}")
    print(f"{'Field':<22}" + "".join(f"{n:>{cw}}" for n in names))
    print("-" * (22 + cw * len(names)))
    for field in TEXT_FIELDS:
        row = f"{field:<22}"
        for n in names:
            c, t = text_scores[n][field]
            row += f"{_cell(c,t):>{cw}}"
        print(row)
    for schema in ('tops', 'bottoms'):
        print(f"\n  [{schema} measurements]")
        for field in MEASUREMENT_RANGES.get(schema, {}):
            row = f"  {field:<20}"
            for n in names:
                c, t = meas_scores[n][schema][field]
                row += f"{_cell(c,t):>{cw}}"
            print(row)


def serialise(text_scores, meas_scores, names):
    return {
        'text':  {n: {f: {'c': v[0], 't': v[1]} for f, v in text_scores[n].items()} for n in names},
        'meas':  {n: {s: {f: {'c': v[0], 't': v[1]} for f, v in meas_scores[n][s].items()}
                      for s in ('tops', 'bottoms')} for n in names},
    }


if __name__ == '__main__':
    run()
