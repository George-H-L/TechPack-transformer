import sys
import json
from datetime import datetime
from pathlib import Path

from techpack_generator.ml_model.inference import TechPackGenerator
from techpack_generator.ml_model.config    import ModelConfig
from techpack_generator.ml_model.validation import MEASUREMENT_RANGES, _garment_schema

config = ModelConfig()

PROMPT_SETS = {
    "General": [
        ("slim fit navy wool blazer",                                            "blazer"),
        ("straight leg dark wash denim jeans",                                   "jeans"),
        ("oversized charcoal grey cotton hoodie with kangaroo pocket",           "hoodie"),
        ("high waisted olive green cargo trousers with six pockets",             "trousers"),
        ("fitted burgundy ribbed cotton t-shirt with crew neck",                 "t-shirt"),
        ("relaxed fit cream linen shirt with chest pocket",                      "shirt"),
        ("cropped black puffer jacket with zip front",                           "jacket"),
        ("wide leg navy blue wool trousers with pleated front",                  "trousers"),
        ("fitted emerald green satin midi dress with cowl neck",                 "dress"),
        ("heavyweight charcoal grey sweatshirt with raglan sleeves",             "sweatshirt"),
    ],
    "Single Word": [
        ("jeans",                                                                "jeans"),
        ("hoodie",                                                               "hoodie"),
        ("trousers",                                                             "trousers"),
        ("coat",                                                                 "coat"),
    ],
    "Misspellings": [
        ("blakc leather jacet with zippers",                                     "jacket"),
        ("grey woll blazer slim fit",                                            "blazer"),
        ("cottn t shirt white oversized",                                        "t-shirt"),
        ("skiny jens dark blue ankle lenght",                                    "jeans"),
        ("beige chinos regulr fit with belt loops",                              "chinos"),
    ],
    "Schema Conflicts": [
        ("leather jacket with 30 inch inseam",                                   "jacket"),
        ("slim fit jeans with 40 inch chest measurement",                        "jeans"),
        ("cotton hoodie with waistband height and rise measurements",            "hoodie"),
    ],
    "With Measurements": [
        ("black t-shirt 38 inch chest medium weight cotton",                     "t-shirt"),
        ("navy chinos 32 inch waist 30 inch inseam straight leg",                "chinos"),
        ("wool coat 180gsm knee length double breasted",                         "coat"),
        ("cotton joggers 280gsm relaxed fit with elastic waist",                 "joggers"),
        ("white linen shirt 42 chest slim fit 180gsm",                           "shirt"),
    ],
    "Conversational": [
        ("something casual for a barbecue",                                      None),
        ("id like a smart casual shirt for work",                                "shirt"),
        ("make me a cosy winter jumper in dark green",                           "jumper"),
        ("need some comfortable summer shorts for the beach",                    "shorts"),
        ("want a going out top in something silky and fitted",                   None),
        ("can you make a simple everyday black t-shirt",                         "t-shirt"),
    ],
    "Edge Cases": [
        ("jacket",                                                               "jacket"),
        ("something warm for winter in dark colours",                            None),
        ("SLIM FIT BLACK DENIM JEANS WITH DISTRESSING AND SILVER ZIP FLY 32 INCH WAIST", "jeans"),
        ("a]b[c invalid!! chars...",                                             None),
        ("OVERSIZED RED HOODIE HEAVYWEIGHT COTTON DROP SHOULDER",                "hoodie"),
    ],
}

ALL_PROMPTS = [p for prompts in PROMPT_SETS.values() for p in prompts]

TRACKED_FIELDS = ['garment_type', 'fabric_type', 'colour', 'fabric_weight', 'seam_type', 'fit']

VARIANTS = [
    ('v1', Path(config.model_dir) / 'best_model_combined.pth'),
    ('v2', Path(config.model_dir) / 'best_model_v2.pth'),
    ('v3', Path(config.model_dir) / 'best_model_v3.pth'),
    ('v4', Path(config.model_dir) / 'best_model_v4.pth'),
]


def _fields_complete(tech_pack):
    tp = tech_pack.get('tech_pack', tech_pack)
    mat  = tp['material']
    meas = tp['measurements']
    con  = tp['construction']
    sty  = tp['style']
    flat = {
        'garment_type': tp['garment_type'],
        'fabric_type':  mat['fabric_type'],
        'colour':       mat['colour'],
        'fabric_weight':mat['fabric_weight'],
        'seam_type':    con['seam_type'],
        'fit':          sty['fit'],
    }
    present = sum(1 for v in flat.values() if v not in (None, '', 'null'))
    has_measurement = any(v not in (None, '', 'null') for v in meas.values())
    total = len(flat) + 1
    return (present + int(has_measurement)), total


def _garment_match(tech_pack, expected_keyword):
    if expected_keyword is None:
        return None
    tp = tech_pack.get('tech_pack', tech_pack)
    gt = (tp['garment_type'] or '').lower()
    return int(expected_keyword.lower() in gt)


def _measurements_valid(tech_pack):
    tp     = tech_pack.get('tech_pack', tech_pack)
    gt     = (tp['garment_type'] or '').lower()
    meas   = tp['measurements']
    schema = _garment_schema(gt)
    if schema == 'unknown':
        schema = 'tops'
    ranges = MEASUREMENT_RANGES[schema]
    for field, (lo, hi) in ranges.items():
        val = meas.get(field)
        if val is None:
            continue
        try:
            if not (lo <= int(val) <= hi):
                return 0
        except (TypeError, ValueError):
            return 0
    return 1


def score(result):
    if not result['success']:
        return None
    tp   = result['tech_pack']
    confs = result['confidences']
    present, total   = _fields_complete(tp)
    mean_conf        = sum(confs.values()) / len(confs) if confs else 0.0
    return {
        'json_valid':          1,
        'fields_present':      present,
        'fields_total':        total,
        'measurements_valid':  _measurements_valid(tp),
        'mean_confidence':     mean_conf,
    }


def run():
    generators = {}
    for name, path in VARIANTS:
        if path.exists():
            try:
                generators[name] = TechPackGenerator(model_path=str(path))
                print(f"Loaded {name}")
            except Exception as e:
                print(f"Could not load {name}: {e}")
        else:
            print(f"{name}: not trained")

    if not generators:
        print("No models available.")
        sys.exit(1)

    print()
    results = {name: [] for name in generators}

    sections = PROMPT_SETS.items()
    for section_name, prompts in sections:
        print(f"[{section_name}]")
        for i, (prompt, expected_garment) in enumerate(prompts, 1):
            global_idx = ALL_PROMPTS.index((prompt, expected_garment)) + 1
            print(f"\n{global_idx}. \"{prompt}\"")
            for name, gen in generators.items():
                result = gen.generate(prompt)
                s = score(result)
                if s is None:
                    print(f"   {name}  FAILED: {result['error']}")
                    results[name].append(None)
                    continue

                gm = _garment_match(result['tech_pack'], expected_garment)
                json_sym     = 'Y' if s['json_valid']         else 'N'
                meas_sym     = 'Y' if s['measurements_valid'] else 'N'
                garment_str  = ('Y' if gm else 'N') if gm is not None else 'n/a'
                print(
                    f"   {name}  json={json_sym}  "
                    f"fields={s['fields_present']}/{s['fields_total']}  "
                    f"garment={garment_str}  "
                    f"measures={meas_sym}  "
                    f"conf={s['mean_confidence']:.2f}"
                )
                results[name].append({'score': s, 'garment_match': gm, 'prompt': prompt})
        print()

    # summary
    print("=" * 56)
    print(f"{'Summary':<10} {'json':>6}  {'fields':>8}  {'garment':>8}  {'meas':>6}  {'conf':>6}")
    print("-" * 56)
    for name in generators:
        rows = [r for r in results[name] if r is not None]
        if not rows:
            print(f"{name:<10} no results")
            continue
        json_rate   = sum(r['score']['json_valid'] for r in rows)
        total       = len(rows)
        avg_fields  = sum(r['score']['fields_present'] / r['score']['fields_total'] for r in rows) / total
        gm_rows     = [r for r in rows if r['garment_match'] is not None]
        garment_str = f"{sum(r['garment_match'] for r in gm_rows)}/{len(gm_rows)}" if gm_rows else 'n/a'
        meas_rate   = sum(r['score']['measurements_valid'] for r in rows)
        avg_conf    = sum(r['score']['mean_confidence'] for r in rows) / total
        print(
            f"{name:<10} {json_rate}/{total}  "
            f"{avg_fields:>8.2f}  "
            f"{garment_str:>8}  "
            f"{meas_rate}/{total}  "
            f"{avg_conf:>6.2f}"
        )

    out = {
        'run_at': datetime.now().isoformat(timespec='seconds'),
        'results': {
            name: [
                {
                    'prompt':           r['prompt'],
                    'garment_match':    r['garment_match'],
                    **r['score'],
                } if r is not None else None
                for r in results[name]
            ]
            for name in generators
        },
        'summary': {},
    }
    for name in generators:
        rows = [r for r in results[name] if r is not None]
        if not rows:
            continue
        total      = len(rows)
        gm_rows    = [r for r in rows if r['garment_match'] is not None]
        out['summary'][name] = {
            'json_rate':       f"{sum(r['score']['json_valid'] for r in rows)}/{total}",
            'avg_fields':      round(sum(r['score']['fields_present'] / r['score']['fields_total'] for r in rows) / total, 3),
            'garment_rate':    f"{sum(r['garment_match'] for r in gm_rows)}/{len(gm_rows)}" if gm_rows else 'n/a',
            'measures_rate':   f"{sum(r['score']['measurements_valid'] for r in rows)}/{total}",
            'avg_confidence':  round(sum(r['score']['mean_confidence'] for r in rows) / total, 3),
        }

    save_path = Path(config.model_dir) / 'eval_results.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {save_path}")


if __name__ == '__main__':
    run()
