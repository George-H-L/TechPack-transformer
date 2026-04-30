import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from techpack_generator.ml_model.inference import TechPackGenerator
from techpack_generator.ml_model.config    import ModelConfig
from techpack_generator.ml_model.validation import MEASUREMENT_RANGES, _garment_schema

config = ModelConfig()

# tests the raw model output before extract_tech_pack_fields applies any defaults
# so you can see what the model actually learned vs what post-processing covers up

TEST_CASES = [
    # tops - garment + colour + fabric + fit
    {'prompt': 'black cotton t-shirt slim fit',
     'exp_garment': 't-shirt',   'schema': 'tops',
     'exp': {'colour': 'black',     'fabric_type': 'cotton',  'fit': 'slim'}},

    {'prompt': 'white linen shirt regular fit',
     'exp_garment': 'shirt',     'schema': 'tops',
     'exp': {'colour': 'white',     'fabric_type': 'linen',   'fit': 'regular'}},

    {'prompt': 'navy wool blazer slim fit',
     'exp_garment': 'blazer',    'schema': 'tops',
     'exp': {'colour': 'navy',      'fabric_type': 'wool',    'fit': 'slim'}},

    {'prompt': 'grey cotton hoodie oversized fit',
     'exp_garment': 'hoodie',    'schema': 'tops',
     'exp': {'colour': 'grey',      'fabric_type': 'cotton',  'fit': 'oversized'}},

    {'prompt': 'red leather jacket fitted',
     'exp_garment': 'jacket',    'schema': 'tops',
     'exp': {'colour': 'red',       'fabric_type': 'leather', 'fit': 'fitted'}},

    {'prompt': 'cream wool coat long',
     'exp_garment': 'coat',      'schema': 'tops',
     'exp': {'colour': 'cream',     'fabric_type': 'wool',    'length': 'long'}},

    {'prompt': 'burgundy silk midi dress fitted',
     'exp_garment': 'dress',     'schema': 'tops',
     'exp': {'colour': 'burgundy',  'fabric_type': 'silk',    'fit': 'fitted'}},

    {'prompt': 'charcoal cotton sweatshirt relaxed fit',
     'exp_garment': 'sweatshirt','schema': 'tops',
     'exp': {'colour': 'charcoal',  'fabric_type': 'cotton',  'fit': 'relaxed'}},

    {'prompt': 'olive green linen shirt loose fit',
     'exp_garment': 'shirt',     'schema': 'tops',
     'exp': {'colour': 'olive',     'fabric_type': 'linen',   'fit': 'loose'}},

    {'prompt': 'black polyester puffer jacket',
     'exp_garment': 'jacket',    'schema': 'tops',
     'exp': {'colour': 'black',     'fabric_type': 'polyester'}},

    # tops - length
    {'prompt': 'black cotton cropped hoodie',
     'exp_garment': 'hoodie',    'schema': 'tops',
     'exp': {'colour': 'black',     'fabric_type': 'cotton',  'length': 'cropped'}},

    {'prompt': 'navy wool longline coat',
     'exp_garment': 'coat',      'schema': 'tops',
     'exp': {'colour': 'navy',      'fabric_type': 'wool',    'length': 'long'}},

    # tops - closure
    {'prompt': 'black zip-up cotton hoodie',
     'exp_garment': 'hoodie',    'schema': 'tops',
     'exp': {'colour': 'black',     'fabric_type': 'cotton',  'closure_type': 'zip'}},

    {'prompt': 'white button-up cotton shirt',
     'exp_garment': 'shirt',     'schema': 'tops',
     'exp': {'colour': 'white',     'fabric_type': 'cotton',  'closure_type': 'button'}},

    # tops - fabric weight
    {'prompt': 'heavyweight black cotton hoodie 400gsm',
     'exp_garment': 'hoodie',    'schema': 'tops',
     'exp': {'colour': 'black',     'fabric_type': 'cotton',  'fabric_weight': '400'}},

    {'prompt': 'lightweight white linen shirt 120gsm',
     'exp_garment': 'shirt',     'schema': 'tops',
     'exp': {'colour': 'white',     'fabric_type': 'linen',   'fabric_weight': '120'}},

    # bottoms
    {'prompt': 'dark blue denim jeans slim fit',
     'exp_garment': 'jeans',     'schema': 'bottoms',
     'exp': {'fabric_type': 'denim',  'fit': 'slim'}},

    {'prompt': 'beige cotton chinos regular fit',
     'exp_garment': 'chinos',    'schema': 'bottoms',
     'exp': {'colour': 'beige',     'fabric_type': 'cotton',  'fit': 'regular'}},

    {'prompt': 'black cotton joggers relaxed fit with drawstring',
     'exp_garment': 'joggers',   'schema': 'bottoms',
     'exp': {'colour': 'black',     'fabric_type': 'cotton',  'fit': 'relaxed',
             'closure_type': 'drawstring'}},

    {'prompt': 'olive green cargo trousers regular fit with patch pockets',
     'exp_garment': 'trousers',  'schema': 'bottoms',
     'exp': {'fabric_type': 'cotton',  'fit': 'regular', 'pockets': 'patch'}},

    {'prompt': 'navy blue cotton shorts relaxed fit',
     'exp_garment': 'shorts',    'schema': 'bottoms',
     'exp': {'colour': 'navy',      'fabric_type': 'cotton',  'fit': 'relaxed'}},

    {'prompt': 'black denim jeans with zip fly and welt pockets',
     'exp_garment': 'jeans',     'schema': 'bottoms',
     'exp': {'colour': 'black',     'fabric_type': 'denim',   'closure_type': 'zip',
             'pockets': 'welt'}},

    {'prompt': 'grey wool trousers slim fit with flat fell seam',
     'exp_garment': 'trousers',  'schema': 'bottoms',
     'exp': {'colour': 'grey',      'fabric_type': 'wool',    'fit': 'slim',
             'seam_type': 'flat'}},

    {'prompt': 'burgundy cotton leggings fitted',
     'exp_garment': 'leggings',  'schema': 'bottoms',
     'exp': {'colour': 'burgundy',  'fabric_type': 'cotton',  'fit': 'fitted'}},
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

VARIANTS = [
    ('v1', Path(config.model_dir) / 'best_model_combined.pth'),
    ('v2', Path(config.model_dir) / 'best_model_v2.pth'),
    ('v3', Path(config.model_dir) / 'best_model_v3.pth'),
    ('v4', Path(config.model_dir) / 'best_model_v4.pth'),
]


def _get_raw_tp(result):
    data = result['tech_pack']
    return data.get('tech_pack', data)


def _field_val(tp, field):
    loc, key = FIELD_LOCATION[field]
    if loc == 'root':
        return (tp.get(key) or '').lower().strip()
    section = tp.get(loc) or {}
    return (section.get(key) or '').lower().strip()


def _score_text(actual, expected_keyword):
    return bool(actual) and expected_keyword.lower() in actual


def _score_measurements(tp, schema):
    if schema not in MEASUREMENT_RANGES:
        return {}, {}
    ranges = MEASUREMENT_RANGES[schema]
    meas = tp.get('measurements') or {}
    present = {}
    in_range = {}
    for field, (lo, hi) in ranges.items():
        raw = meas.get(field)
        try:
            v = int(raw)
            present[field]  = v > 0
            in_range[field] = lo <= v <= hi
        except (TypeError, ValueError):
            present[field]  = False
            in_range[field] = False
    return present, in_range


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
            print(f"  {name}: model file not found")

    if not generators:
        print("No models available.")
        sys.exit(1)

    names = list(generators.keys())

    text_scores  = {n: defaultdict(lambda: [0, 0]) for n in names}
    meas_present = {n: {'tops':    defaultdict(lambda: [0, 0]),
                        'bottoms': defaultdict(lambda: [0, 0])} for n in names}
    failures = {n: [] for n in names}

    print(f"\nRunning {len(TEST_CASES)} test cases across {len(generators)} models...\n")

    for tc in TEST_CASES:
        prompt      = tc['prompt']
        exp_garment = tc['exp_garment']
        schema      = tc['schema']
        expected    = tc['exp']

        for name, gen in generators.items():
            result = gen.generate(prompt)
            if not result['success']:
                failures[name].append(f"JSON fail | {prompt}")
                text_scores[name]['garment_type'][1] += 1
                for field in expected:
                    text_scores[name][field][1] += 1
                continue

            tp = _get_raw_tp(result)

            actual_garment = _field_val(tp, 'garment_type')
            text_scores[name]['garment_type'][1] += 1
            if _score_text(actual_garment, exp_garment):
                text_scores[name]['garment_type'][0] += 1
            else:
                failures[name].append(
                    f"garment_type | \"{prompt}\" | want: {exp_garment} | got: \"{actual_garment}\""
                )

            for field, expected_kw in expected.items():
                actual = _field_val(tp, field)
                text_scores[name][field][1] += 1
                if _score_text(actual, expected_kw):
                    text_scores[name][field][0] += 1
                else:
                    failures[name].append(
                        f"{field} | \"{prompt}\" | want: {expected_kw} | got: \"{actual}\""
                    )

            present_d, inrange_d = _score_measurements(tp, schema)
            for field, ok in present_d.items():
                meas_present[name][schema][field][1] += 1
                if ok and inrange_d[field]:
                    meas_present[name][schema][field][0] += 1

    all_text_fields = [
        'garment_type', 'colour', 'fabric_type', 'fabric_weight',
        'fit', 'length', 'seam_type', 'closure_type', 'pockets',
    ]

    col_w = 12
    head  = f"{'Field':<22}" + "".join(f"{n:>{col_w}}" for n in names)
    sep   = "-" * len(head)

    print("=" * len(head))
    print("TEXT FIELD ACCURACY   (raw model output, before any defaults)")
    print("=" * len(head))
    print(head)
    print(sep)

    for field in all_text_fields:
        row = f"{field:<22}"
        for name in names:
            c, t = text_scores[name][field]
            cell = "n/a" if t == 0 else f"{c}/{t}  ({100*c//t}%)"
            row += f"{cell:>{col_w}}"
        print(row)

    for schema_name in ('tops', 'bottoms'):
        ranges = MEASUREMENT_RANGES.get(schema_name, {})
        if not ranges:
            continue
        print()
        print(f"MEASUREMENTS  non-zero and in valid range  [{schema_name}]")
        print(sep)
        for field in ranges:
            row = f"{field:<22}"
            for name in names:
                c, t = meas_present[name][schema_name][field]
                cell = "n/a" if t == 0 else f"{c}/{t}  ({100*c//t}%)"
                row += f"{cell:>{col_w}}"
            print(row)

    print()
    print(sep)
    print(f"{'OVERALL (text fields)':<22}", end="")
    for name in names:
        total_c = sum(v[0] for v in text_scores[name].values())
        total_t = sum(v[1] for v in text_scores[name].values())
        pct = 100 * total_c // total_t if total_t else 0
        print(f"{f'{total_c}/{total_t}  ({pct}%)':>{col_w}}", end="")
    print()

    print(f"{'JSON failures':<22}", end="")
    for name in names:
        n_fail = sum(1 for f in failures[name] if f.startswith('JSON'))
        print(f"{n_fail:>{col_w}}", end="")
    print()

    print()
    for name in names:
        if failures[name]:
            print(f"\n{name.upper()} failures ({len(failures[name])}):")
            for f in failures[name]:
                print(f"  {f}")

    out = {
        'text_scores': {
            n: {f: {'correct': v[0], 'total': v[1]}
                for f, v in text_scores[n].items()}
            for n in names
        },
        'meas_scores': {
            n: {s: {f: {'correct': v[0], 'total': v[1]}
                    for f, v in meas_present[n][s].items()}
                for s in ('tops', 'bottoms')}
            for n in names
        },
    }
    save_path = Path(config.model_dir) / 'eval_fields.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {save_path}")


if __name__ == '__main__':
    run()
