import math

# line weights for the three visual layers of the drawing
STROKE = {'OUTLINE': 1.5, 'INTERNAL': 0.8, 'STITCH': 0.5}

FIT_MULTIPLIERS = {
    'slim': 0.82, 'fitted': 0.84, 'regular': 0.92,
    'relaxed': 1.1, 'loose': 1.2, 'baggy': 1.2,
    'boxy': 1.2, 'oversized': 1.25,
}

LENGTH_MULTIPLIERS = {
    'cropped': 0.85, 'short': 0.9, 'regular': 1.0,
    'long': 1.1, 'longline': 1.2,
}

SLEEVE_SCALE = {
    't-shirt': 1.5, 'shirt': 1.25, 'polo': 1.5,
    'blazer': 1.1, 'jacket': 1.1, 'coat': 1.2,
    'hoodie': 1.25, 'sweater': 1.25, 'jumper': 1.25,
}

COLOUR_MAP = {
    'black': '#1a1a1a', 'white': '#FFFFFF', 'navy': '#010A56',
    'navy blue': '#010A56', 'red': '#DC143C', 'blue': '#0074D9',
    'light blue': '#87CEEB', 'sky blue': '#87CEEB', 'royal blue': '#4169E1',
    'green': '#2ECC40', 'forest green': '#228B22', 'olive green': '#6B8E23',
    'dark green': '#006400', 'sage green': '#9CAF88', 'grey': '#808080',
    'gray': '#808080', 'light grey': '#C0C0C0', 'dark grey': '#444444',
    'charcoal': '#36454F', 'burgundy': '#800020', 'maroon': '#5C0020',
    'cream': '#FFFDD0', 'ivory': '#FFFFF0', 'beige': '#D4C5A9',
    'tan': '#D2B48C', 'camel': '#C4A55A', 'brown': '#5C4033',
    'emerald': '#50C878', 'teal': '#008080', 'khaki': '#BDB76B',
    'pink': '#E8909C', 'coral': '#FF6F61', 'mustard': '#D4A017',
    'yellow': '#F0C040', 'orange': '#E87040', 'purple': '#7B2D8E',
    'lavender': '#B8A9C9', 'olive': '#6B7135',
}

_BOTTOMS_KEYWORDS = {
    'jean', 'trouser', 'chino', 'shorts', 'jogger', 'legging', 'skirt',
    'pant', 'cargo', 'culotte', 'sweatpant', 'tracksuit bottom'
}

# match user colour name against COLOUR_MAP, fall back to longest substring hit
def _resolve_colour(name):
    c = name.lower().strip()
    if c in COLOUR_MAP: return COLOUR_MAP[c]
    for key in sorted(COLOUR_MAP, key=len, reverse=True):
        if key in c: return COLOUR_MAP[key]
    return '#CCCCCC'

# nudge the collar away from the body fill so it actually reads as a separate
# panel. Dark bodies get lighter, light bodies get darker, mid-tones get a bump.
def _collar_colour(fill):
    try:
        h = fill.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        if lum < 0.25: r, g, b = min(255, r + 70), min(255, g + 70), min(255, b + 70)
        elif lum > 0.75: r, g, b = max(0, r - 45), max(0, g - 45), max(0, b - 45)
        else: r, g, b = min(255, r + 30), min(255, g + 30), min(255, b + 30)
        return f'#{r:02X}{g:02X}{b:02X}'
    except Exception: return fill

# clamp-safe RGB shift down by `amount` per channel
def _darken(hex_color, amount):
    try:
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f'#{max(0,r-amount):02X}{max(0,g-amount):02X}{max(0,b-amount):02X}'
    except Exception: return hex_color

# clamp-safe RGB shift up by `amount` per channel
def _lighten(hex_color, amount):
    try:
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f'#{min(255,r+amount):02X}{min(255,g+amount):02X}{min(255,b+amount):02X}'
    except Exception: return hex_color

# pick a sleeve length multiplier from SLEEVE_SCALE, falling back to 0.85
def _sleeve_scale(garment_type):
    key = garment_type.lower().strip()
    if key in SLEEVE_SCALE: return SLEEVE_SCALE[key]
    for name, scale in SLEEVE_SCALE.items():
        if name in key or key in name: return scale
    return 0.85

# keep tops out of the bottoms branch even when the keyword overlaps
# (e.g. "shirt dress" contains "shirt")
def _is_bottoms(garment_type: str) -> bool:
    gt = garment_type.lower()
    if any(k in gt for k in ['top', 'shirt', 'jacket', 'hoodie', 'sweater', 'jumper', 'tee']):
        return False
    return any(k in gt for k in _BOTTOMS_KEYWORDS)

# collapse free-text garment input into one of polo / tank / shirt / passthrough
# so the renderer doesn't have to deal with every spelling variant
def _normalise_garment(garment_type):
    g = garment_type.lower().strip()
    if 'polo' in g: return 'polo'
    if 'tank' in g or 'singlet' in g or 'vest' in g: return 'tank'
    if 'shirt' in g and 't-shirt' not in g and 'tee' not in g: return 'shirt'
    return g

# entry point. Builds the full SVG (front + back views + spec boxes) from a
# tech_pack object. Branches to the bottoms renderer for pants/skirts and the
# tops renderer for everything else.
def generate_garment_svg(tech_pack):
    scale = 8  # inches -> canvas px

    if _is_bottoms(tech_pack.garment_type):
        waist       = getattr(tech_pack, 'waist', 30) * scale / 2
        hips        = getattr(tech_pack, 'hips', 38) * scale / 2
        rise        = getattr(tech_pack, 'rise', 10) * scale
        inseam      = min(getattr(tech_pack, 'inseam', 30) * scale, 300)
        thigh       = getattr(tech_pack, 'thigh', 22) * scale / 2
        leg_opening = getattr(tech_pack, 'leg_opening', 14) * scale / 2
        garment     = tech_pack.garment_type.lower()
        fit         = getattr(tech_pack, 'fit', 'regular').lower()
        fill        = _resolve_colour(getattr(tech_pack, 'colour', 'grey'))
        front_cx, back_cx, top_y = 350, 850, 100

        closure_type = getattr(tech_pack, 'closure_type', '') or ''
        pockets      = getattr(tech_pack, 'pockets', '') or ''
        front = _draw_bottoms_garment(front_cx, top_y, waist, hips, rise, inseam, thigh, leg_opening, fill, fit, garment, is_front=True,  closure_type=closure_type, pockets=pockets)
        back  = _draw_bottoms_garment(back_cx,  top_y, waist, hips, rise, inseam, thigh, leg_opening, fill, fit, garment, is_front=False, closure_type=closure_type, pockets=pockets)
        box_y = top_y + rise + inseam + 35
        boxes = _draw_bottoms_info_boxes(tech_pack, box_y)
        svg_height = int(box_y + 140)

        return f'''<svg width="100%" height="auto" viewBox="0 0 1200 {svg_height}" xmlns="http://www.w3.org/2000/svg" style="max-width:1200px;display:block;margin:0 auto;">
    <rect width="1200" height="{svg_height}" fill="#F9F9F9"/>
    <text x="600" y="30" font-family="Arial" font-size="22" font-weight="bold" text-anchor="middle" fill="#000">{tech_pack.garment_type.upper()}</text>
    <text x="600" y="48" font-family="Arial" font-size="12" text-anchor="middle" fill="#666">{getattr(tech_pack, 'colour', '').title()} | {getattr(tech_pack, 'fabric_type', '').title()} | {getattr(tech_pack, 'fit', '').title()} Fit</text>
    <text x="{front_cx}" y="80" font-family="Arial" font-size="11" font-weight="bold" text-anchor="middle" fill="#333">FRONT</text>
    <text x="{back_cx}" y="80" font-family="Arial" font-size="11" font-weight="bold" text-anchor="middle" fill="#333">BACK</text>
    {front}
    {back}
    {boxes}
</svg>'''

    # ---- tops branch ----
    chest         = getattr(tech_pack, 'chest', 40) * scale
    waist         = getattr(tech_pack, 'waist', 38) * scale
    sleeve_length = getattr(tech_pack, 'sleeve_length', 10) * scale
    body_length   = getattr(tech_pack, 'body_length', 28) * scale
    _sh_raw       = getattr(tech_pack, 'shoulder', 18)
    # shoulder slider is amplified 3x around the 17" centre so small ticks
    # produce a visible silhouette change
    shoulder      = max(10, 17 + (_sh_raw - 17) * 3) * scale

    garment  = _normalise_garment(tech_pack.garment_type)
    fit      = getattr(tech_pack, 'fit', 'regular').lower()

    # t-shirts run narrower than other tops, big outerwear gets squashed in
    # so it doesn't blow past the canvas. shirts/polos sit between the two.
    tshirt_w = 0.7 if garment == 't-shirt' else 1.0
    large_w  = 0.8 if garment in ('hoodie', 'jumper', 'sweater', 'jacket', 'coat', 'blazer') else 1.0
    shirt_w  = 0.8 if garment in ('shirt', 'polo') else 1.0
    width_sc = tshirt_w * large_w * shirt_w
    half_chest = (chest / 2) * 0.65 * width_sc
    half_waist = (waist / 2) * 0.65 * width_sc
    shoulder   *= 0.65 * width_sc

    length = getattr(tech_pack, 'length', 'regular')
    if isinstance(length, str): length = length.lower()
    body_length *= LENGTH_MULTIPLIERS.get(length, 1.0)

    sleeve_length *= _sleeve_scale(tech_pack.garment_type)
    # boxy/baggy fits get exaggerated dropped sleeves
    if fit in ('boxy', 'baggy'):
        sleeve_length = min(sleeve_length * 2.2, sleeve_length + 20)

    fill         = _resolve_colour(getattr(tech_pack, 'colour', 'grey'))
    sleeve_width = getattr(tech_pack, 'sleeve_width', 10)
    tank_divet   = getattr(tech_pack, 'tank_divet', 10)
    closure_type = getattr(tech_pack, 'closure_type', '')
    pockets      = getattr(tech_pack, 'pockets', '')

    front_cx, back_cx, top_y = 350, 850, 140

    is_tank = garment == 'tank'
    if not is_tank:
        # clamp sleeve x-reach so it doesn't run off the canvas at width 1200
        half_sh   = shoulder / 2
        sl_dx     = math.cos(math.radians(20)) * sleeve_length
        max_sl_dx = max(0, 330 - half_sh)
        if sl_dx > max_sl_dx:
            sleeve_length *= max_sl_dx / sl_dx

    front = _draw_garment(front_cx, top_y, half_chest, half_waist, body_length,
                          sleeve_length, shoulder, fill, fit, garment, is_front=True,
                          sleeve_width=sleeve_width, tank_divet=tank_divet,
                          closure_type=closure_type, pockets=pockets)
    back  = _draw_garment(back_cx,  top_y, half_chest, half_waist, body_length,
                          sleeve_length, shoulder, fill, fit, garment, is_front=False,
                          sleeve_width=sleeve_width, tank_divet=tank_divet,
                          closure_type=closure_type, pockets=pockets)

    box_y = top_y + body_length + 35
    boxes = _draw_info_boxes(tech_pack, box_y)
    svg_height = int(box_y + 160)

    return f'''<svg width="100%" height="auto" viewBox="0 0 1200 {svg_height}" xmlns="http://www.w3.org/2000/svg" style="max-width:1200px;display:block;margin:0 auto;">
    <rect width="1200" height="{svg_height}" fill="#F9F9F9"/>
    <text x="600" y="30" font-family="Arial" font-size="22" font-weight="bold" text-anchor="middle" fill="#000">{tech_pack.garment_type.upper()}</text>
    <text x="600" y="48" font-family="Arial" font-size="12" text-anchor="middle" fill="#666">{getattr(tech_pack, 'colour', '').title()} | {getattr(tech_pack, 'fabric_type', '').title()} | {getattr(tech_pack, 'fit', '').title()} Fit</text>
    <text x="{front_cx}" y="100" font-family="Arial" font-size="11" font-weight="bold" text-anchor="middle" fill="#333">FRONT</text>
    <text x="{back_cx}"  y="100" font-family="Arial" font-size="11" font-weight="bold" text-anchor="middle" fill="#333">BACK</text>
    {front}
    {back}
    {boxes}
</svg>'''


# builds the tops silhouette: body + sleeves + collar + buttons + extras.
# centred at (cx, top_y), all measurements already scaled to canvas px.
def _draw_garment(cx, top_y, half_chest, half_waist, body_length, sleeve_length,
                  shoulder, fill, fit, garment, is_front, sleeve_width=10, tank_divet=10,
                  closure_type='', pockets=''):

    hem_y         = top_y + body_length
    half_shoulder = shoulder / 2
    is_shirt      = garment == 'shirt'
    is_tank       = garment == 'tank'
    has_collar    = garment in ('shirt', 'polo')
    is_hoodie     = 'hoodie' in garment
    is_jumper     = is_hoodie or 'jumper' in garment or 'sweater' in garment

    if not is_tank:
        half_shoulder = max(half_shoulder, half_chest * 0.85)

    # Tightened basic t-shirt collar width visually
    neck_rx = half_shoulder * (0.28 if garment == 't-shirt' else 0.40)
    if is_front:
        neck_ry = 28 if is_shirt else (22 if garment == 'polo' else (42 if is_tank else (20 if is_jumper else 18)))
    else:
        neck_ry = 8 if has_collar else (22 if is_tank else 8)

    sh_drop      = 14 if is_shirt else 12
    sleeve_angle = math.radians(20)

    l_sh_x = cx - half_shoulder
    r_sh_x = cx + half_shoulder
    sh_y   = top_y + sh_drop

    armhole_y = top_y + max(60, half_chest * 0.45)
    if fit in ('boxy', 'baggy'): armhole_y += 10
    if is_tank: armhole_y += 75

    l_body_x  = cx - half_chest
    r_body_x  = cx + half_chest

    sl_dx = math.cos(sleeve_angle) * sleeve_length
    sl_dy = math.sin(sleeve_angle) * sleeve_length

    open_drop = sleeve_length * 0.375
    taper     = 16

    if is_shirt:
        open_drop *= 0.5
        taper      = 6
        sl_dx     *= 0.9
    elif garment == 't-shirt':
        taper = 30
        open_drop *= 1.4
    elif garment == 'polo':
        taper = 4         # near-parallel sleeve edges, very subtle taper
        open_drop *= 1.15  # slightly wider cuff opening than base

    if fit == 'slim':
        taper = max(4, int(taper * 0.65))
        if is_shirt: open_drop *= 0.5

    if fit in ('boxy', 'baggy', 'oversized'):
        taper     = int(taper * 2.5)
        open_drop *= 1.2

    if sleeve_width is not None:
        factor    = float(sleeve_width) / 10.0
        open_drop *= factor
        taper     *= math.sqrt(factor)

    l_sl_out_x = l_sh_x - sl_dx
    l_sl_out_y = sh_y + sl_dy
    l_sl_in_x  = l_sl_out_x + taper
    l_sl_in_y  = l_sl_out_y + open_drop

    r_sl_out_x = r_sh_x + sl_dx
    r_sl_out_y = sh_y + sl_dy
    r_sl_in_x  = r_sl_out_x - taper
    r_sl_in_y  = r_sl_out_y + open_drop

    arm_inset = 16
    if is_tank:
        td = tank_divet if tank_divet is not None else 10
        f  = min(1.0, max(0.0, td / 10.0))
        narrow_l   = cx - neck_rx - 8
        narrow_r   = cx + neck_rx + 8
        l_sh_x_eff = narrow_l + (l_sh_x - narrow_l) * f
        r_sh_x_eff = narrow_r + (r_sh_x - narrow_r) * f
        arm_cp_y = sh_y + (armhole_y - sh_y) * 0.45
        inward = 1 - f * 0.6
        l_arm_cp = (l_sh_x_eff + (cx - l_sh_x_eff) * inward, arm_cp_y)
        r_arm_cp = (r_sh_x_eff - (r_sh_x_eff - cx) * inward, arm_cp_y)
        l_arm_path = f"Q {l_arm_cp[0]} {l_arm_cp[1]}, {l_body_x} {armhole_y}"
        r_arm_path = f"Q {r_arm_cp[0]} {r_arm_cp[1]}, {r_body_x} {armhole_y}"
    else:
        l_sh_x_eff = l_sh_x
        r_sh_x_eff = r_sh_x
        # Solid armhole curve directly integrated into body
        l_arm_path = f"Q {l_body_x+6} {sh_y+10}, {l_body_x} {armhole_y}"
        r_arm_path = f"Q {r_body_x-6} {sh_y+10}, {r_body_x} {armhole_y}"

    mid_y       = (armhole_y + hem_y) / 2
    neck_base_y = top_y + neck_ry * 0.6
    l_neck_x    = cx - neck_rx
    r_neck_x    = cx + neck_rx

    # Side-seam waist suppression: shirts get a real tailored taper rather than
    # the boxy near-straight curve used for tees/jumpers.
    waist_pull = 16 if is_shirt else 4
    if fit in ('boxy', 'baggy', 'oversized', 'loose'):
        waist_pull = 4

    bow = 6 if is_jumper else 0
    l_sl_top = f"Q {l_sh_x - sl_dx/2 - bow} {sh_y + sl_dy/2 - bow}, {l_sl_out_x} {l_sl_out_y}" if bow else f"L {l_sl_out_x} {l_sl_out_y}"
    r_sl_top = f"Q {r_sh_x + sl_dx/2 + bow} {sh_y + sl_dy/2 - bow}, {r_sl_out_x} {r_sl_out_y}" if bow else f"L {r_sl_out_x} {r_sl_out_y}"

    # Separated Sleeves drawn behind the body to eliminate geometry cutouts
    sleeves_svg = ''
    if not is_tank:
        l_sleeve_path = f"M {l_sh_x} {sh_y} {l_sl_top} L {l_sl_in_x} {l_sl_in_y} Q {l_body_x} {armhole_y+20}, {cx} {armhole_y} L {cx} {sh_y} Z"
        r_sleeve_path = f"M {r_sh_x} {sh_y} {r_sl_top} L {r_sl_in_x} {r_sl_in_y} Q {r_body_x} {armhole_y+20}, {cx} {armhole_y} L {cx} {sh_y} Z"
        sleeves_svg = f'<path d="{l_sleeve_path}" fill="{fill}" stroke="#000" stroke-width="{STROKE["OUTLINE"]}" stroke-linejoin="round"/><path d="{r_sleeve_path}" fill="{fill}" stroke="#000" stroke-width="{STROKE["OUTLINE"]}" stroke-linejoin="round"/>'

    if has_collar and is_front:
        path = f'''M {l_neck_x} {top_y} Q {cx} {top_y + 8}, {r_neck_x} {top_y} L {r_sh_x} {sh_y} {r_arm_path} Q {r_body_x - waist_pull} {mid_y}, {cx + half_waist} {hem_y - 10} Q {cx + half_waist} {hem_y}, {cx + half_waist - 8} {hem_y} L {cx - half_waist + 8} {hem_y} Q {cx - half_waist} {hem_y}, {cx - half_waist} {hem_y - 10} Q {l_body_x + waist_pull} {mid_y}, {l_body_x} {armhole_y} {l_arm_path} L {l_sh_x} {sh_y} Z'''
    elif has_collar and not is_front:
        path = f'''M {l_neck_x} {top_y} L {l_sh_x} {sh_y} {l_arm_path} Q {l_body_x + waist_pull} {mid_y}, {cx - half_waist} {hem_y - 10} Q {cx - half_waist} {hem_y}, {cx - half_waist + 8} {hem_y} L {cx + half_waist - 8} {hem_y} Q {cx + half_waist} {hem_y}, {cx + half_waist} {hem_y - 10} Q {r_body_x - waist_pull} {mid_y}, {r_body_x} {armhole_y} {r_arm_path} L {r_sh_x} {sh_y} L {r_neck_x} {top_y} Q {cx} {top_y + neck_ry}, {l_neck_x} {top_y} Z'''
    elif is_tank:
        path = f'''M {l_neck_x} {top_y} L {l_sh_x_eff} {sh_y} {l_arm_path} Q {l_body_x + 4} {mid_y}, {cx - half_waist} {hem_y - 10} Q {cx - half_waist} {hem_y}, {cx - half_waist + 8} {hem_y} L {cx + half_waist - 8} {hem_y} Q {cx + half_waist} {hem_y}, {cx + half_waist} {hem_y - 10} Q {r_body_x - 4} {mid_y}, {r_body_x} {armhole_y} {r_arm_path} L {r_sh_x_eff} {sh_y} L {r_neck_x} {top_y} Q {cx} {top_y + neck_ry}, {l_neck_x} {top_y} Z'''
    else:
        path = f'''M {l_neck_x} {top_y} L {l_sh_x} {sh_y} {l_arm_path} Q {l_body_x + 4} {mid_y}, {cx - half_waist} {hem_y - 10} Q {cx - half_waist} {hem_y}, {cx - half_waist + 8} {hem_y} L {cx + half_waist - 8} {hem_y} Q {cx + half_waist} {hem_y}, {cx + half_waist} {hem_y - 10} Q {r_body_x - 4} {mid_y}, {r_body_x} {armhole_y} {r_arm_path} L {r_sh_x} {sh_y} L {r_neck_x} {top_y} Q {cx} {top_y + neck_ry}, {l_neck_x} {top_y} Z'''

    neck_inner = ''
    if is_hoodie:
        collar = _draw_hood(cx, top_y, neck_rx, neck_ry, fill, is_front)
    elif garment == 'polo':
        collar = _draw_polo_collar(cx, top_y, neck_rx, fill, is_front, with_placket=True,  size='polo')
    elif is_shirt:
        # shirt reuses the polo collar geometry but skips the placket -
        # _draw_shirt_buttons puts the full-length button placket on instead
        collar = _draw_polo_collar(cx, top_y, neck_rx, fill, is_front, with_placket=False, size='shirt')
    else:
        collar = _draw_crew_neck(cx, top_y, neck_rx, fill, is_front, is_jumper, scoop_override=neck_ry if is_tank else neck_ry)

    buttons = ''
    if is_front and is_shirt:
        buttons = _draw_shirt_buttons(cx, neck_base_y, hem_y)

    seams = ''
    if not is_tank:
        # Armhole dashed stitches overlaid on the solid outline seam
        seams = f'''
    <path d="M {l_sh_x-3} {sh_y} Q {l_body_x+3} {sh_y+10}, {l_body_x-3} {armhole_y}" fill="none" stroke="#555" stroke-width="{STROKE['STITCH']}" stroke-dasharray="2,2"/>
    <path d="M {r_sh_x+3} {sh_y} Q {r_body_x-3} {sh_y+10}, {r_body_x+3} {armhole_y}" fill="none" stroke="#555" stroke-width="{STROKE['STITCH']}" stroke-dasharray="2,2"/>'''

    tension = f'''
    <path d="M {l_body_x} {armhole_y} Q {l_body_x+3} {armhole_y+4}, {l_body_x+1} {armhole_y+6}" fill="none" stroke="#888" stroke-width="{STROKE['STITCH']}"/>
    <path d="M {r_body_x} {armhole_y} Q {r_body_x-3} {armhole_y+4}, {r_body_x-1} {armhole_y+6}" fill="none" stroke="#888" stroke-width="{STROKE['STITCH']}"/>'''

    hem_fill = _lighten(fill, 22)
    hem = ''
    if is_jumper:
        hem_seam_y = hem_y - 30
        bw = half_waist - 5
        hem = (
            f'<rect x="{cx - bw:.1f}" y="{hem_seam_y:.1f}" width="{bw * 2:.1f}" height="{hem_y - hem_seam_y:.1f}" fill="{hem_fill}" stroke="none"/>'
            f'<path d="M {cx - bw:.1f} {hem_seam_y:.1f} Q {cx} {hem_seam_y:.1f}, {cx + bw:.1f} {hem_seam_y:.1f}" fill="none" stroke="#000" stroke-width="{STROKE["INTERNAL"]}"/>'
        )
        for rx in range(int(cx - bw + 3), int(cx + bw - 1), 6):
            hem += f'<line x1="{rx}" y1="{hem_seam_y}" x2="{rx}" y2="{hem_y-1}" stroke="#555" stroke-width="0.4" opacity="0.6"/>'
    else:
        strip_h = 8
        bw = half_waist - 8
        hem = (
            f'<rect x="{cx - bw:.1f}" y="{hem_y - strip_h:.1f}" width="{bw * 2:.1f}" height="{strip_h:.1f}" fill="{hem_fill}" stroke="none"/>'
            f'<line x1="{cx-bw+4}" y1="{hem_y-3}" x2="{cx+bw-4}" y2="{hem_y-3}" stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,1"/>'
            f'<line x1="{cx-bw+4}" y1="{hem_y-5}" x2="{cx+bw-4}" y2="{hem_y-5}" stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,1"/>'
        )

    sleeve_hem = ''
    if not is_shirt and not is_tank:
        if is_jumper:
            tc = 0.20 
            lo_x = l_sl_out_x + tc * (l_sh_x - l_sl_out_x)
            lo_y = l_sl_out_y + tc * (sh_y   - l_sl_out_y)
            li_x = l_sl_in_x  + tc * (l_sh_x - l_sl_in_x)
            li_y = l_sl_in_y  + tc * (sh_y   - l_sl_in_y)
            ro_x = r_sl_out_x + tc * (r_sh_x - r_sl_out_x)
            ro_y = r_sl_out_y + tc * (sh_y   - r_sl_out_y)
            ri_x = r_sl_in_x  + tc * (r_sh_x - r_sl_in_x)
            ri_y = r_sl_in_y  + tc * (sh_y   - r_sl_in_y)
            sleeve_hem += (
                f'<path d="M {lo_x:.1f} {lo_y:.1f} L {li_x:.1f} {li_y:.1f} L {l_sl_in_x:.1f} {l_sl_in_y:.1f} L {l_sl_out_x:.1f} {l_sl_out_y:.1f} Z" fill="{hem_fill}" stroke="none"/>'
                f'<path d="M {ro_x:.1f} {ro_y:.1f} L {ri_x:.1f} {ri_y:.1f} L {r_sl_in_x:.1f} {r_sl_in_y:.1f} L {r_sl_out_x:.1f} {r_sl_out_y:.1f} Z" fill="{hem_fill}" stroke="none"/>'
                f'<line x1="{lo_x:.1f}" y1="{lo_y:.1f}" x2="{li_x:.1f}" y2="{li_y:.1f}" stroke="#000" stroke-width="{STROKE["INTERNAL"]}"/>'
                f'<line x1="{ro_x:.1f}" y1="{ro_y:.1f}" x2="{ri_x:.1f}" y2="{ri_y:.1f}" stroke="#000" stroke-width="{STROKE["INTERNAL"]}"/>'
            )
            for step in range(1, 5):
                fx = lo_x + (l_sl_out_x - lo_x) * (step/5)
                fy = lo_y + (l_sl_out_y - lo_y) * (step/5)
                ix = li_x + (l_sl_in_x - li_x) * (step/5)
                iy = li_y + (l_sl_in_y - li_y) * (step/5)
                sleeve_hem += f'<line x1="{fx}" y1="{fy}" x2="{ix}" y2="{iy}" stroke="#555" stroke-width="0.4" opacity="0.6"/>'
            for step in range(1, 5):
                fx = ro_x + (r_sl_out_x - ro_x) * (step/5)
                fy = ro_y + (r_sl_out_y - ro_y) * (step/5)
                ix = ri_x + (r_sl_in_x - ri_x) * (step/5)
                iy = ri_y + (r_sl_in_y - ri_y) * (step/5)
                sleeve_hem += f'<line x1="{fx}" y1="{fy}" x2="{ix}" y2="{iy}" stroke="#555" stroke-width="0.4" opacity="0.6"/>'
        else:
            for t in (0.06, 0.09): 
                lo_x = l_sl_out_x + t * (l_sh_x - l_sl_out_x)
                lo_y = l_sl_out_y + t * (sh_y   - l_sl_out_y)
                li_x = l_sl_in_x  + t * (l_sh_x - l_sl_in_x)
                li_y = l_sl_in_y  + t * (sh_y   - l_sl_in_y)
                ro_x = r_sl_out_x + t * (r_sh_x - r_sl_out_x)
                ro_y = r_sl_out_y + t * (sh_y   - r_sl_out_y)
                ri_x = r_sl_in_x  + t * (r_sh_x - r_sl_in_x)
                ri_y = r_sl_in_y  + t * (sh_y   - r_sl_in_y)
                sleeve_hem += f'''
        <line x1="{lo_x}" y1="{lo_y}" x2="{li_x}" y2="{li_y}" stroke="#555" stroke-width="{STROKE['STITCH']}" stroke-dasharray="2,1"/>
        <line x1="{ro_x}" y1="{ro_y}" x2="{ri_x}" y2="{ri_y}" stroke="#555" stroke-width="{STROKE['STITCH']}" stroke-dasharray="2,1"/>'''

    cuffs = ''
    if is_shirt:
        tc = 0.10
        lco_x = l_sl_out_x + tc * (l_sh_x - l_sl_out_x)
        lco_y = l_sl_out_y + tc * (sh_y   - l_sl_out_y)
        lci_x = l_sl_in_x  + tc * (l_sh_x  - l_sl_in_x)
        lci_y = l_sl_in_y  + tc * (sh_y    - l_sl_in_y)
        rco_x = r_sl_out_x + tc * (r_sh_x  - r_sl_out_x)
        rco_y = r_sl_out_y + tc * (sh_y    - r_sl_out_y)
        rci_x = r_sl_in_x  + tc * (r_sh_x  - r_sl_in_x)
        rci_y = r_sl_in_y  + tc * (sh_y    - r_sl_in_y)
        l_tip_mid_x = (l_sl_out_x + l_sl_in_x) / 2
        l_tip_mid_y = (l_sl_out_y + l_sl_in_y) / 2
        r_tip_mid_x = (r_sl_out_x + r_sl_in_x) / 2
        r_tip_mid_y = (r_sl_out_y + r_sl_in_y) / 2
        sw_int = STROKE['INTERNAL']
        sw_stitch = STROKE['STITCH']
        cuffs = (
            f'<line x1="{lco_x}" y1="{lco_y}" x2="{lci_x}" y2="{lci_y}" stroke="#000" stroke-width="{sw_int}"/>'
            f'<line x1="{rco_x}" y1="{rco_y}" x2="{rci_x}" y2="{rci_y}" stroke="#000" stroke-width="{sw_int}"/>'
            f'<path d="M {l_sl_out_x} {l_sl_out_y} Q {l_tip_mid_x-5} {l_tip_mid_y+4}, {l_sl_in_x} {l_sl_in_y}" fill="none" stroke="#555" stroke-width="{sw_stitch}"/>'
            f'<path d="M {r_sl_out_x} {r_sl_out_y} Q {r_tip_mid_x+5} {r_tip_mid_y+4}, {r_sl_in_x} {r_sl_in_y}" fill="none" stroke="#555" stroke-width="{sw_stitch}"/>'
        )

    extras = ''
    if is_hoodie and is_front:
        has_zip         = any(z in closure_type.lower() for z in ('zip', 'zipper'))
        has_side_pockets = bool(pockets) and pockets.lower() not in ('', 'none', 'no')
        zip_top_y       = top_y + neck_ry
        if has_zip:
            extras += _draw_hoodie_zip(cx, zip_top_y, hem_y)
            if has_side_pockets:
                extras += _draw_hoodie_side_pockets(cx, armhole_y, hem_y, half_waist)
        else:
            extras += _draw_kangaroo_pocket(cx, armhole_y, hem_y, half_waist, fill)

    # left-chest patch pocket for shirts when the construction calls for one
    if is_shirt and is_front and pockets and pockets.lower() not in ('', 'none', 'no'):
        pkt_w  = max(20, half_chest * 0.32)
        pkt_h  = pkt_w * 1.05
        pkt_cx = cx - half_chest * 0.45
        pkt_top = armhole_y + 14
        ax     = pkt_cx - pkt_w / 2
        extras += (
            f'<rect x="{ax:.1f}" y="{pkt_top:.1f}" width="{pkt_w:.1f}" height="{pkt_h:.1f}" '
            f'fill="{fill}" stroke="#000" stroke-width="{STROKE["INTERNAL"]}"/>'
            f'<line x1="{ax + 2:.1f}" y1="{pkt_top + 3:.1f}" x2="{ax + pkt_w - 2:.1f}" y2="{pkt_top + 3:.1f}" '
            f'stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,2"/>'
        )

    view_id = 'front' if is_front else 'back'
    return f'''<g id="{view_id}">
    {sleeves_svg}
    {neck_inner}
    <path d="{path}" fill="{fill}" stroke="#000" stroke-width="{STROKE['OUTLINE']}" stroke-linejoin="round"/>
    {seams}{tension}{hem}{sleeve_hem}
    {collar}{buttons}{cuffs}{extras}
</g>'''


# crew/scoop neck rib band. Jumpers get a thicker band with vertical rib lines,
# everything else gets a thin band with a single dashed stitch.
def _draw_crew_neck(cx, top_y, neck_rx, fill, is_front, is_jumper=False, scoop_override=None):
    collar_fill = _collar_colour(fill)

    band_w = 26 if is_jumper else 14
    rx = neck_rx + (12 if is_jumper else 4)
    
    scoop = scoop_override if scoop_override is not None else (20 if is_front else 10)
    scoop += 2
    
    collar = f'''
    <path d="M {cx-rx} {top_y} Q {cx} {top_y+scoop}, {cx+rx} {top_y}
             Q {cx} {top_y+scoop-band_w}, {cx-rx} {top_y} Z" fill="{collar_fill}" stroke="#000" stroke-width="{STROKE['OUTLINE']}"/>'''
    
    if is_jumper:
        for offset in range(-int(rx)+4, int(rx)-3, 6):
            cy = top_y + scoop - band_w/2
            collar += f'<line x1="{cx+offset}" y1="{cy-band_w/2.5}" x2="{cx+offset}" y2="{cy+band_w/2.5}" stroke="#555" stroke-width="0.4" opacity="0.6"/>'
        collar += f'<path d="M {cx-rx+3} {top_y+band_w/2} Q {cx} {top_y+scoop-band_w/2}, {cx+rx-3} {top_y+band_w/2}" fill="none" stroke="#555" stroke-width="{STROKE['STITCH']}" stroke-dasharray="2,2"/>'
    else:
        collar += f'<path d="M {cx-rx+5} {top_y+3} Q {cx} {top_y+scoop-3}, {cx+rx-5} {top_y+3}" fill="none" stroke="#555" stroke-width="{STROKE['STITCH']}" stroke-dasharray="2,2"/>'
        
    return collar


# hoodie hood. Front view = inside lining + outer dome + face opening +
# drawstrings. Back view = outer dome + centre seam.
def _draw_hood(cx, top_y, neck_rx, neck_ry, fill, is_front):
    hood_fill = fill
    dark_fill = _lighten(fill, 18)
    hw = neck_rx + 15
    hh = 110
    apex_y = top_y - hh
    
    if is_front:
        inside = (f'<path d="M {cx-neck_rx} {top_y} Q {cx} {top_y - 15}, {cx+neck_rx} {top_y} Q {cx} {top_y + neck_ry}, {cx-neck_rx} {top_y} Z" fill="{dark_fill}" stroke="#000" stroke-width="{STROKE["OUTLINE"]}"/>')
        dome = (f'<path d="M {cx-hw} {top_y+10} C {cx-hw-10} {top_y-40}, {cx-hw*0.7} {apex_y}, {cx} {apex_y} C {cx+hw*0.7} {apex_y}, {cx+hw+10} {top_y-40}, {cx+hw} {top_y+10} Q {cx+15} {top_y + neck_ry + 10}, {cx} {top_y + neck_ry + 15} Q {cx-15} {top_y + neck_ry + 10}, {cx-hw} {top_y+10} Z" fill="{hood_fill}" stroke="#000" stroke-width="{STROKE["OUTLINE"]}"/>')
        face_w = hw * 0.65
        face_open = (f'<path d="M {cx-hw+10} {top_y+8} C {cx-face_w} {top_y-20}, {cx-face_w*0.8} {apex_y+25}, {cx} {apex_y+25} C {cx+face_w*0.8} {apex_y+25}, {cx+face_w} {top_y-20}, {cx+hw-10} {top_y+8} Q {cx} {top_y + neck_ry + 5}, {cx-hw+10} {top_y+8} Z" fill="{dark_fill}" stroke="#000" stroke-width="{STROKE["OUTLINE"]}"/>')
        drawstrings = (f'<path d="M {cx-12} {top_y + neck_ry} Q {cx-15} {top_y + neck_ry + 30}, {cx-12} {top_y + neck_ry + 60}" fill="none" stroke="#333" stroke-width="2.5"/><path d="M {cx+12} {top_y + neck_ry} Q {cx+15} {top_y + neck_ry + 30}, {cx+12} {top_y + neck_ry + 60}" fill="none" stroke="#333" stroke-width="2.5"/>')
        return inside + dome + face_open + drawstrings
    else:
        dome = (f'<path d="M {cx-hw} {top_y+10} C {cx-hw-10} {top_y-40}, {cx-hw*0.7} {apex_y}, {cx} {apex_y} C {cx+hw*0.7} {apex_y}, {cx+hw+10} {top_y-40}, {cx+hw} {top_y+10} Q {cx} {top_y + 15}, {cx-hw} {top_y+10} Z" fill="{hood_fill}" stroke="#000" stroke-width="{STROKE["OUTLINE"]}"/>')
        seam = (f'<path d="M {cx} {apex_y} Q {cx+5} {top_y - 20}, {cx} {top_y + 10}" fill="none" stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="3,3"/>')
        return dome + seam


# centre-front full-length zip running from zip_top_y down to the hem
def _draw_hoodie_zip(cx, zip_top_y, hem_y):
    zip_end_y = hem_y - 2
    out = (
        f'<line x1="{cx-2}" y1="{zip_top_y:.1f}" x2="{cx-2}" y2="{zip_end_y:.1f}" stroke="#666" stroke-width="1.2"/>'
        f'<line x1="{cx+2}" y1="{zip_top_y:.1f}" x2="{cx+2}" y2="{zip_end_y:.1f}" stroke="#666" stroke-width="1.2"/>'
    )
    for y in range(int(zip_top_y + 10), int(zip_end_y), 9):
        out += f'<line x1="{cx-5}" y1="{y}" x2="{cx+5}" y2="{y}" stroke="#999" stroke-width="0.6"/>'
    out += f'<rect x="{cx-4:.1f}" y="{zip_top_y - 3:.1f}" width="8" height="7" rx="1.5" fill="none" stroke="#555" stroke-width="1.2"/>'
    return out


# small vertical side-seam slip pockets, used on zip-up hoodies (no kangaroo)
def _draw_hoodie_side_pockets(cx, armhole_y, hem_y, half_waist):
    pocket_y     = armhole_y + (hem_y - armhole_y) * 0.28
    pocket_h     = (hem_y - armhole_y) * 0.26
    pocket_end_y = min(pocket_y + pocket_h, hem_y - 34)
    px_l = cx - half_waist * 0.72
    px_r = cx + half_waist * 0.72
    out = ''
    for px, dx in ((px_l, -2), (px_r, 2)):
        out += (
            f'<line x1="{px:.1f}" y1="{pocket_y:.1f}" x2="{px:.1f}" y2="{pocket_end_y:.1f}" stroke="#000" stroke-width="{STROKE["INTERNAL"]}"/>'
            f'<line x1="{px+dx:.1f}" y1="{pocket_y+2:.1f}" x2="{px+dx:.1f}" y2="{pocket_end_y-2:.1f}" stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,2"/>'
        )
    return out


# pullover hoodie kangaroo pocket. Hex/trapezoid shape with the bottom anchored
# above the rib hem so it doesn't overlap the bottom band.
def _draw_kangaroo_pocket(cx, armpit_y, hem_y, half_waist, fill):
    pw_top = half_waist * 0.40
    pw_bot = half_waist * 0.65

    pocket_bottom = hem_y - 30
    pocket_top    = max(armpit_y + 15, pocket_bottom - 60)
    
    ph = pocket_bottom - pocket_top
    v_drop = min(20, ph * 0.4) 
    
    path = (f'M {cx - pw_top} {pocket_top} L {cx + pw_top} {pocket_top} L {cx + pw_bot} {pocket_bottom - v_drop} L {cx + pw_bot} {pocket_bottom} L {cx - pw_bot} {pocket_bottom} L {cx - pw_bot} {pocket_bottom - v_drop} Z')
    return (
        f'<path d="{path}" fill="{fill}" stroke="#000" stroke-width="{STROKE["INTERNAL"]}"/>'
        f'<path d="M {cx - pw_top + 3} {pocket_top + 3} L {cx + pw_top - 3} {pocket_top + 3}" fill="none" stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,2"/>'
        f'<path d="M {cx - pw_top + 3} {pocket_top + 3} L {cx - pw_bot + 3} {pocket_bottom - v_drop + 3}" fill="none" stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,2"/>'
        f'<path d="M {cx + pw_top - 3} {pocket_top + 3} L {cx + pw_bot - 3} {pocket_bottom - v_drop + 3}" fill="none" stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,2"/>'
        f'<path d="M {cx - pw_bot + 3} {pocket_bottom - v_drop + 3} L {cx - pw_bot + 3} {pocket_bottom - 3}" fill="none" stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,2"/>'
        f'<path d="M {cx + pw_bot - 3} {pocket_bottom - v_drop + 3} L {cx + pw_bot - 3} {pocket_bottom - 3}" fill="none" stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,2"/>'
    )


def _draw_polo_collar(cx, top_y, neck_rx, fill, is_front, with_placket=True, size='polo'):
    # Polo/shirt collar. Back band + 2 leaves + placket. Quadratic Bezier
    # control y is set to 2x the visual offset because the curve only reaches
    # halfway from endpoint to control.
    collar_fill = _collar_colour(fill)
    band_under  = collar_fill                  # back band matches the leaves
    crease_col  = _darken(collar_fill, 30)
    fall_seam   = _darken(collar_fill, 35)

    # ~15% smaller across the board than the previous pass
    if size == 'shirt':
        band_w     = neck_rx + 10
        arch_h     = 11
        v_depth    = 22
        leaf_drop  = 48
        tip_outset = 10
        outer_bow  = 4
        roll_drop  = 5
    else:  # polo
        band_w     = neck_rx + 7
        arch_h     = 9
        v_depth    = 19
        leaf_drop  = 38
        tip_outset = 6
        outer_bow  = 3
        roll_drop  = 4

    band_inner = top_y + 4
    gorge_y    = top_y + v_depth
    tip_y      = top_y + leaf_drop
    tip_x      = band_w + tip_outset
    top_cp_y   = top_y - 2 * arch_h          # 2x so arch reaches band_top
    bot_cp_y   = 2 * gorge_y - band_inner    # 2x so dip reaches gorge_y

    # back view: closed band shape, top arches up, bottom flush at top_y
    # so it sits clean against the body neckline (no visible canvas gap)
    if not is_front:
        return (
            f'<path d="M {cx-band_w:.1f} {top_y:.1f} '
            f'Q {cx:.1f} {top_cp_y:.1f}, {cx+band_w:.1f} {top_y:.1f} '
            f'Q {cx:.1f} {top_y + 6:.1f}, {cx-band_w:.1f} {top_y:.1f} Z" '
            f'fill="{collar_fill}" stroke="#000" '
            f'stroke-width="{STROKE["OUTLINE"]}" stroke-linejoin="round"/>'
            # top edge stitch
            f'<path d="M {cx-band_w+5:.1f} {top_y - 2:.1f} '
            f'Q {cx:.1f} {top_y - arch_h*1.5:.1f}, {cx+band_w-5:.1f} {top_y - 2:.1f}" '
            f'fill="none" stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,2"/>'
            # stand/fall fold seam at top_y
            f'<path d="M {cx-band_w+4:.1f} {top_y + 1:.1f} '
            f'Q {cx:.1f} {top_y + 4:.1f}, {cx+band_w-4:.1f} {top_y + 1:.1f}" '
            f'fill="none" stroke="{fall_seam}" stroke-width="{STROKE["INTERNAL"]}"/>'
            # second dashed stitch below the fold for stand thickness
            f'<path d="M {cx-band_w+5:.1f} {top_y + 3:.1f} '
            f'Q {cx:.1f} {top_y + 6:.1f}, {cx+band_w-5:.1f} {top_y + 3:.1f}" '
            f'fill="none" stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,2"/>'
        )

    # front view layers: back band -> stitches/fold lines -> leaves -> creases
    back_band = (
        f'<path d="M {cx-band_w:.1f} {top_y:.1f} '
        f'Q {cx:.1f} {top_cp_y:.1f}, {cx+band_w:.1f} {top_y:.1f} '
        f'L {cx+band_w-2:.1f} {band_inner:.1f} '
        f'Q {cx:.1f} {bot_cp_y:.1f}, {cx-band_w+2:.1f} {band_inner:.1f} '
        f'Z" '
        f'fill="{band_under}" stroke="#000" '
        f'stroke-width="{STROKE["OUTLINE"]}" stroke-linejoin="round"/>'
    )
    # top-edge stitch on the band's arch
    band_top_stitch = (
        f'<path d="M {cx-band_w+5:.1f} {top_y - 2:.1f} '
        f'Q {cx:.1f} {top_y - arch_h*1.5:.1f}, {cx+band_w-5:.1f} {top_y - 2:.1f}" '
        f'fill="none" stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,2"/>'
    )
    # stand/fall fold seam, drawn at top_y so leaves cover the outer ends
    # and only the centre shows through the V opening
    fold_seam = (
        f'<path d="M {cx-band_w+3:.1f} {top_y:.1f} '
        f'Q {cx:.1f} {top_y + 3:.1f}, {cx+band_w-3:.1f} {top_y:.1f}" '
        f'fill="none" stroke="{fall_seam}" stroke-width="{STROKE["INTERNAL"]}"/>'
    )
    # inner roll line inside the V, hints at the band's interior curl
    inner_roll = (
        f'<path d="M {cx-band_w*0.55:.1f} {top_y + 5:.1f} '
        f'Q {cx:.1f} {gorge_y - 3:.1f}, {cx+band_w*0.55:.1f} {top_y + 5:.1f}" '
        f'fill="none" stroke="{crease_col}" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,1.5"/>'
    )

    # left leaf: straight L for top-inner and V edges (sharp gorge + tip),
    # subtle Q on the outer edge for a soft outward bow
    left_leaf = (
        f'<path d="M {cx-band_w:.1f} {top_y:.1f} '
        f'L {cx:.1f} {gorge_y:.1f} '
        f'L {cx-tip_x:.1f} {tip_y:.1f} '
        f'Q {cx-tip_x-outer_bow:.1f} {top_y + leaf_drop*0.45:.1f}, '
        f'{cx-band_w:.1f} {top_y:.1f} Z" '
        f'fill="{collar_fill}" stroke="#000" '
        f'stroke-width="{STROKE["OUTLINE"]}" stroke-linejoin="miter" stroke-miterlimit="6"/>'
    )
    right_leaf = (
        f'<path d="M {cx+band_w:.1f} {top_y:.1f} '
        f'L {cx:.1f} {gorge_y:.1f} '
        f'L {cx+tip_x:.1f} {tip_y:.1f} '
        f'Q {cx+tip_x+outer_bow:.1f} {top_y + leaf_drop*0.45:.1f}, '
        f'{cx+band_w:.1f} {top_y:.1f} Z" '
        f'fill="{collar_fill}" stroke="#000" '
        f'stroke-width="{STROKE["OUTLINE"]}" stroke-linejoin="miter" stroke-miterlimit="6"/>'
    )

    # subtle roll-line creases on each leaf, parallel to the band edge
    crease_y = top_y + roll_drop
    creases = (
        f'<path d="M {cx-band_w+5:.1f} {crease_y:.1f} '
        f'L {cx-band_w*0.18:.1f} {gorge_y - 5:.1f}" '
        f'fill="none" stroke="{crease_col}" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,1.5"/>'
        f'<path d="M {cx+band_w-5:.1f} {crease_y:.1f} '
        f'L {cx+band_w*0.18:.1f} {gorge_y - 5:.1f}" '
        f'fill="none" stroke="{crease_col}" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,1.5"/>'
    )

    # band layers go down first, then fold/roll lines, then leaves on top.
    # the leaves naturally clip the fold seam to just the V opening area.
    out = back_band + band_top_stitch + fold_seam + inner_roll + left_leaf + right_leaf + creases

    # 6. placket + 3 buttons (polo only). Placket sized so button diameter sits
    # inside with ~1.5px of margin each side, buttons spaced evenly down.
    if with_placket:
        placket_w   = 11
        btn_r       = 2.2
        placket_top = gorge_y - 2
        placket_h   = 40
        px          = cx - placket_w / 2
        out += (
            f'<rect x="{px:.1f}" y="{placket_top:.1f}" '
            f'width="{placket_w:.1f}" height="{placket_h:.1f}" '
            f'fill="{fill}" stroke="#000" stroke-width="{STROKE["INTERNAL"]}"/>'
            f'<line x1="{px+1.5:.1f}" y1="{placket_top+3:.1f}" '
            f'x2="{px+1.5:.1f}" y2="{placket_top+placket_h-3:.1f}" '
            f'stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,1.5"/>'
            f'<line x1="{px+placket_w-1.5:.1f}" y1="{placket_top+3:.1f}" '
            f'x2="{px+placket_w-1.5:.1f}" y2="{placket_top+placket_h-3:.1f}" '
            f'stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,1.5"/>'
        )
        btn_fill = _lighten(collar_fill, 28)
        # space 3 buttons across the placket inset from the top/bottom by btn_r*2
        btn_top  = placket_top + btn_r * 2 + 2
        btn_bot  = placket_top + placket_h - btn_r * 2 - 2
        btn_step = (btn_bot - btn_top) / 2
        for i in range(3):
            by = btn_top + i * btn_step
            out += (
                f'<circle cx="{cx:.1f}" cy="{by:.1f}" r="{btn_r}" '
                f'fill="{btn_fill}" stroke="#222" stroke-width="0.7"/>'
                # 4 thread holes
                f'<circle cx="{cx-0.7:.1f}" cy="{by-0.7:.1f}" r="0.3" fill="#222"/>'
                f'<circle cx="{cx+0.7:.1f}" cy="{by-0.7:.1f}" r="0.3" fill="#222"/>'
                f'<circle cx="{cx-0.7:.1f}" cy="{by+0.7:.1f}" r="0.3" fill="#222"/>'
                f'<circle cx="{cx+0.7:.1f}" cy="{by+0.7:.1f}" r="0.3" fill="#222"/>'
            )

    return out


# full-length button placket for shirts. Two parallel placket stitch lines
# plus 6 buttons evenly spaced from the neckline to ~14px above the hem.
def _draw_shirt_buttons(cx, top_y, hem_y):
    pw = 5
    placket = (f'<line x1="{cx-pw}" y1="{top_y+2}" x2="{cx-pw}" y2="{hem_y-14}" stroke="#444" stroke-width="0.8"/><line x1="{cx+pw}" y1="{top_y+2}" x2="{cx+pw}" y2="{hem_y-14}" stroke="#444" stroke-width="0.8"/>')
    buttons = ''
    n = 6
    for i in range(n):
        by = top_y + 8 + i * ((hem_y - 25 - (top_y + 8)) / (n - 1))
        buttons += (f'<circle cx="{cx}" cy="{by}" r="3" fill="none" stroke="#444" stroke-width="0.8"/><line x1="{cx-1}" y1="{by}" x2="{cx+1}" y2="{by}" stroke="#444" stroke-width="0.5"/><line x1="{cx}" y1="{by-1}" x2="{cx}" y2="{by+1}" stroke="#444" stroke-width="0.5"/>')
    return placket + buttons


# =====================  bottoms  =====================

# big one. Builds the silhouette + waistband + belt loops + fly + scoop pockets
# + back yoke + cargo patches + drawstrings depending on garment subtype.
# Jeans get gold contrast topstitch, everything else gets grey.
def _draw_bottoms_garment(cx, top_y, waist, hips, rise, inseam, thigh, leg_opening, fill, fit, garment, is_front=True, closure_type='', pockets=''):
    g = garment.lower()
    is_skirt = 'skirt' in g
    is_jeans = 'jean' in g or 'denim' in g
    closure = (closure_type or '').lower()
    pockets_l = (pockets or '').lower()
    is_cargo = 'cargo' in g or 'cargo' in pockets_l
    stitch_c = "#D4A017" if is_jeans else "#666"
    has_5pocket = is_jeans or 'chino' in g or 'trouser' in g or is_cargo
    is_soft_bottom = 'jogger' in g or 'sweatpant' in g or 'track' in g or 'legging' in g

    # safety floors - if the model returned tiny numbers fall back to defaults
    # so we don't render a garment the size of a postage stamp
    if waist < 40: waist = 128
    if hips < 40: hips = 160
    if rise < 40: rise = 80
    if inseam < 40: inseam = 240
    if thigh < 20: thigh = 88
    if leg_opening < 10: leg_opening = 56

    half_waist = waist / 2
    half_hip   = hips  / 2
    crotch_y = top_y + rise
    hem_y    = crotch_y + inseam
    hip_ctrl_y = top_y + rise * 0.25
    inseam_ctrl_y = crotch_y + max(12, inseam * 0.04)

    r_waist = cx + half_waist
    r_hip   = cx + half_hip
    l_waist = cx - half_waist
    l_hip   = cx - half_hip
    # outer seam flares outward at the hem proportional to leg_opening
    leg_flare   = leg_opening * 0.2
    r_outer_hem = r_hip + leg_flare
    l_outer_hem = l_hip - leg_flare
    r_inner_hem = r_outer_hem - leg_opening
    l_inner_hem = l_outer_hem + leg_opening
    waistband_h = 14
    dark = _darken(fill, 25)

    if is_skirt:
        hem_flare = half_hip * 1.2
        sk_ctrl_y = top_y + rise * 0.4
        path = (f'M {l_waist:.1f} {top_y} L {r_waist:.1f} {top_y} Q {cx + hem_flare:.1f} {sk_ctrl_y:.1f}, {cx + hem_flare:.1f} {hem_y:.1f} Q {cx:.1f} {hem_y + 6:.1f}, {cx - hem_flare:.1f} {hem_y:.1f} Q {cx - hem_flare:.1f} {sk_ctrl_y:.1f}, {l_waist:.1f} {top_y} Z')
    else:
        path = (f'M {l_waist:.1f} {top_y} L {r_waist:.1f} {top_y} Q {r_hip:.1f} {hip_ctrl_y:.1f}, {r_outer_hem:.1f} {hem_y:.1f} L {r_inner_hem:.1f} {hem_y:.1f} Q {r_inner_hem:.1f} {inseam_ctrl_y:.1f}, {cx:.1f} {crotch_y:.1f} Q {l_inner_hem:.1f} {inseam_ctrl_y:.1f}, {l_inner_hem:.1f} {hem_y:.1f} L {l_outer_hem:.1f} {hem_y:.1f} Q {l_hip:.1f} {hip_ctrl_y:.1f}, {l_waist:.1f} {top_y} Z')

    waistband = (f'<line x1="{l_waist:.1f}" y1="{top_y + waistband_h:.1f}" x2="{r_waist:.1f}" y2="{top_y + waistband_h:.1f}" stroke="#000" stroke-width="{STROKE["INTERNAL"]}"/><path d="M {l_waist + 4:.1f} {top_y + 6:.1f} L {r_waist - 4:.1f} {top_y + 6:.1f}" fill="none" stroke="{stitch_c}" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="3,2"/>')

    belt_loops = ''
    has_loops = not is_skirt and 'jogger' not in g and 'sweatpant' not in g and 'track' not in g
    if has_loops:
        lw, lh = 6, waistband_h + 2
        positions = (-0.42, -0.15, 0, 0.15, 0.42) if not is_front else (-0.42, -0.18, 0.18, 0.42)
        for pos in positions:
            lx = cx + pos * waist - lw / 2
            belt_loops += f'<rect x="{lx:.1f}" y="{top_y - 1:.1f}" width="{lw}" height="{lh}" rx="1" fill="{dark}" stroke="#000" stroke-width="0.8"/>'

    fly = ''
    if is_front and not is_skirt and has_loops:
        fly_len = min(rise * 0.6, 48)
        fly = (f'<line x1="{cx:.1f}" y1="{top_y + waistband_h:.1f}" x2="{cx:.1f}" y2="{top_y + fly_len:.1f}" stroke="#000" stroke-width="{STROKE["INTERNAL"]}"/><path d="M {cx:.1f} {top_y + waistband_h:.1f} Q {cx:.1f} {top_y + fly_len + 8:.1f}, {cx - 10:.1f} {top_y + fly_len:.1f}" fill="none" stroke="{stitch_c}" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,2"/>')
        # waistband button: jeans always, anything with button closure too
        if is_jeans or 'button' in closure:
            btn_y = top_y + waistband_h / 2
            fly += (
                f'<circle cx="{cx:.1f}" cy="{btn_y:.1f}" r="2.6" fill="{_lighten(fill, 30)}" stroke="#222" stroke-width="0.8"/>'
                f'<line x1="{cx-1.2:.1f}" y1="{btn_y:.1f}" x2="{cx+1.2:.1f}" y2="{btn_y:.1f}" stroke="#222" stroke-width="0.5"/>'
            )

    seams = ''
    if not is_skirt:
        seams = (f'<path d="M {r_inner_hem - 2:.1f} {hem_y - 3} Q {r_inner_hem - 2:.1f} {inseam_ctrl_y:.1f}, {cx - 2:.1f} {crotch_y:.1f}" fill="none" stroke="{stitch_c}" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,2"/><path d="M {l_inner_hem + 2:.1f} {hem_y - 3} Q {l_inner_hem + 2:.1f} {inseam_ctrl_y:.1f}, {cx + 2:.1f} {crotch_y:.1f}" fill="none" stroke="{stitch_c}" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,2"/>')

    hem = ''
    if not is_skirt:
        hem = (f'<line x1="{l_outer_hem + 3:.1f}" y1="{hem_y - 6}" x2="{l_inner_hem - 3:.1f}" y2="{hem_y - 6}" stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,1"/><line x1="{r_inner_hem + 3:.1f}" y1="{hem_y - 6}" x2="{r_outer_hem - 3:.1f}" y2="{hem_y - 6}" stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,1"/>')

    extras = ''

    # front scoop pockets for jeans/chinos/trousers. Curved opening from a
    # point on the waistband down and out to the side seam below the hip.
    # Drawn solid + parallel topstitch (jeans get gold contrast thread).
    if has_5pocket and is_front:
        for side_sign in (-1, 1):
            x1 = cx + side_sign * half_waist * 0.40
            y1 = top_y + waistband_h
            seam_t = 0.38
            seam_y = top_y + rise * seam_t
            waist_x = cx + side_sign * half_waist
            hip_x   = cx + side_sign * half_hip
            x2 = waist_x + (hip_x - waist_x) * seam_t - side_sign * 1.5
            # control pulled toward the side seam to give the scoop its shape
            cp_x = x1 + (x2 - x1) * 0.65
            cp_y = y1 + (seam_y - y1) * 0.85
            # parallel topstitch sits ~3px inside the pocket opening
            ix1 = x1 - side_sign * 3
            ix2 = x2 - side_sign * 3
            icp_x = cp_x - side_sign * 3
            extras += (
                f'<path d="M {x1:.1f} {y1:.1f} Q {cp_x:.1f} {cp_y:.1f}, {x2:.1f} {seam_y:.1f}" '
                f'fill="none" stroke="#000" stroke-width="{STROKE["INTERNAL"]}"/>'
                f'<path d="M {ix1:.1f} {y1+1:.1f} Q {icp_x:.1f} {cp_y:.1f}, {ix2:.1f} {seam_y-1:.1f}" '
                f'fill="none" stroke="{stitch_c}" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,2"/>'
            )

    if is_jeans:
        if is_front:
            # coin pocket inside the right front pocket opening
            coin_x = cx + half_waist * 0.42 - half_waist * 0.42 + 6
            coin_w = half_waist * 0.20
            extras += (
                f'<path d="M {coin_x:.1f} {top_y + waistband_h + 2:.1f} '
                f'L {coin_x + coin_w:.1f} {top_y + waistband_h + 2:.1f} '
                f'L {coin_x + coin_w:.1f} {top_y + waistband_h + 18:.1f}" '
                f'fill="none" stroke="{stitch_c}" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="1,1"/>'
            )
            # subtle wash marks: soft light patches on outer thighs and knees
            wash_rx  = half_hip * 0.18
            wash_ry  = inseam  * 0.10
            wash_rx2 = half_hip * 0.12
            wash_ry2 = inseam  * 0.07
            for leg_cx in (cx - half_hip * 0.60, cx + half_hip * 0.60):
                thigh_y = crotch_y + inseam * 0.28
                knee_y  = crotch_y + inseam * 0.60
                extras += (
                    f'<ellipse cx="{leg_cx:.1f}" cy="{thigh_y:.1f}" rx="{wash_rx:.1f}" ry="{wash_ry:.1f}" fill="white" opacity="0.10"/>'
                    f'<ellipse cx="{leg_cx:.1f}" cy="{knee_y:.1f}"  rx="{wash_rx2:.1f}" ry="{wash_ry2:.1f}" fill="white" opacity="0.09"/>'
                )
        else:
            yoke_drop = rise * 0.28
            extras += (f'<path d="M {l_waist:.1f} {top_y + waistband_h + 4} Q {cx:.1f} {top_y + waistband_h + yoke_drop:.1f}, {r_waist:.1f} {top_y + waistband_h + 4}" fill="none" stroke="{stitch_c}" stroke-width="{STROKE["INTERNAL"]}"/><path d="M {l_waist:.1f} {top_y + waistband_h + 7} Q {cx:.1f} {top_y + waistband_h + yoke_drop + 3:.1f}, {r_waist:.1f} {top_y + waistband_h + 7}" fill="none" stroke="{stitch_c}" stroke-width="{STROKE["INTERNAL"]}" stroke-dasharray="2,2"/>')
            # 5-pocket back patch pockets
            bp_w = half_waist * 0.42
            bp_h = rise * 0.42
            bp_y = top_y + waistband_h + yoke_drop + 6
            for side_sign in (-1, 1):
                bp_cx = cx + side_sign * half_waist * 0.45
                ax = bp_cx - bp_w / 2
                # pentagon-shaped back pocket (flat top, tapered bottom)
                pt_y = bp_y + bp_h
                mid_y = pt_y - bp_h * 0.30
                extras += (
                    f'<path d="M {ax:.1f} {bp_y:.1f} '
                    f'L {ax + bp_w:.1f} {bp_y:.1f} '
                    f'L {ax + bp_w:.1f} {mid_y:.1f} '
                    f'L {ax + bp_w / 2:.1f} {pt_y:.1f} '
                    f'L {ax:.1f} {mid_y:.1f} Z" '
                    f'fill="{fill}" stroke="#000" stroke-width="{STROKE["INTERNAL"]}"/>'
                    f'<path d="M {ax + 2:.1f} {bp_y + 2:.1f} '
                    f'L {ax + bp_w - 2:.1f} {bp_y + 2:.1f} '
                    f'L {ax + bp_w - 2:.1f} {mid_y - 1:.1f} '
                    f'L {ax + bp_w / 2:.1f} {pt_y - 2:.1f} '
                    f'L {ax + 2:.1f} {mid_y - 1:.1f} Z" '
                    f'fill="none" stroke="{stitch_c}" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,2"/>'
                )
            # back seat fade: single wide soft patch
            seat_cx = cx
            seat_cy = crotch_y + inseam * 0.12
            extras += f'<ellipse cx="{seat_cx:.1f}" cy="{seat_cy:.1f}" rx="{half_waist * 0.55:.1f}" ry="{rise * 0.30:.1f}" fill="white" opacity="0.08"/>'

    if is_cargo:
        # outer thigh cargo pockets: 2 front (per leg) + 2 back patch pockets
        pocket_w = half_hip * 0.28
        pocket_h = inseam * 0.18
        thigh_y  = crotch_y + inseam * 0.18
        if is_front:
            for px in (l_hip * 0.55 + cx * 0.45, r_hip * 0.55 + cx * 0.45):
                ax = px - pocket_w / 2
                extras += (
                    f'<rect x="{ax:.1f}" y="{thigh_y:.1f}" width="{pocket_w:.1f}" height="{pocket_h:.1f}" rx="2" fill="{fill}" stroke="#000" stroke-width="{STROKE["INTERNAL"]}"/>'
                    f'<line x1="{ax + 3:.1f}" y1="{thigh_y + 3:.1f}" x2="{ax + pocket_w - 3:.1f}" y2="{thigh_y + 3:.1f}" stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,2"/>'
                    f'<rect x="{ax + pocket_w * 0.25:.1f}" y="{thigh_y - 8:.1f}" width="{pocket_w * 0.5:.1f}" height="10" rx="1" fill="{fill}" stroke="#000" stroke-width="{STROKE["INTERNAL"]}"/>'
                )
        else:
            back_y = crotch_y + inseam * 0.08
            bw = half_waist * 0.38
            bh = rise * 0.45
            for px in (l_hip * 0.5 + cx * 0.5, r_hip * 0.5 + cx * 0.5):
                ax = px - bw / 2
                extras += (
                    f'<rect x="{ax:.1f}" y="{back_y:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="2" fill="{fill}" stroke="#000" stroke-width="{STROKE["INTERNAL"]}"/>'
                    f'<line x1="{ax + 3:.1f}" y1="{back_y + 3:.1f}" x2="{ax + bw - 3:.1f}" y2="{back_y + 3:.1f}" stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,2"/>'
                )

    has_drawstring = is_soft_bottom or 'drawstring' in closure
    if is_soft_bottom:
        # elastic ankle cuffs only on hemmed soft bottoms (not leggings)
        if 'legging' not in g:
            l_leg_cx = (l_outer_hem + l_inner_hem) / 2
            r_leg_cx = (r_inner_hem + r_outer_hem) / 2
            hw1, hw2 = abs(l_inner_hem - l_outer_hem) / 2, abs(r_outer_hem - r_inner_hem) / 2
            extras += f'<path d="M {l_leg_cx - hw1} {hem_y} Q {l_leg_cx - hw1 * 0.7} {hem_y + 4}, {l_leg_cx - hw1 * 0.7} {hem_y + 12} L {l_leg_cx + hw1 * 0.7} {hem_y + 12} Q {l_leg_cx + hw1 * 0.7} {hem_y + 4}, {l_leg_cx + hw1} {hem_y}" fill="none" stroke="#000" stroke-width="{STROKE["INTERNAL"]}"/><line x1="{l_leg_cx - hw1 * 0.7 + 3}" y1="{hem_y + 3}" x2="{l_leg_cx - hw1 * 0.7 + 3}" y2="{hem_y + 10}" stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="1,2"/><line x1="{l_leg_cx + hw1 * 0.7 - 3}" y1="{hem_y + 3}" x2="{l_leg_cx + hw1 * 0.7 - 3}" y2="{hem_y + 10}" stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="1,2"/>'
            extras += f'<path d="M {r_leg_cx - hw2} {hem_y} Q {r_leg_cx - hw2 * 0.7} {hem_y + 4}, {r_leg_cx - hw2 * 0.7} {hem_y + 12} L {r_leg_cx + hw2 * 0.7} {hem_y + 12} Q {r_leg_cx + hw2 * 0.7} {hem_y + 4}, {r_leg_cx + hw2} {hem_y}" fill="none" stroke="#000" stroke-width="{STROKE["INTERNAL"]}"/><line x1="{r_leg_cx - hw2 * 0.7 + 3}" y1="{hem_y + 3}" x2="{r_leg_cx - hw2 * 0.7 + 3}" y2="{hem_y + 10}" stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="1,2"/><line x1="{r_leg_cx + hw2 * 0.7 - 3}" y1="{hem_y + 3}" x2="{r_leg_cx + hw2 * 0.7 - 3}" y2="{hem_y + 10}" stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="1,2"/>'

    if has_drawstring and is_front:
        extras += f'<circle cx="{cx - 8}" cy="{top_y + 8}" r="2" fill="none" stroke="#555" stroke-width="0.8"/><circle cx="{cx + 8}" cy="{top_y + 8}" r="2" fill="none" stroke="#555" stroke-width="0.8"/><line x1="{cx - 8}" y1="{top_y + 8}" x2="{cx - 14}" y2="{top_y + 28}" stroke="#555" stroke-width="1" stroke-linecap="round"/><line x1="{cx + 8}" y1="{top_y + 8}" x2="{cx + 14}" y2="{top_y + 28}" stroke="#555" stroke-width="1" stroke-linecap="round"/><rect x="{cx - 16}" y="{top_y + 27}" width="4" height="6" rx="1" fill="#555"/><rect x="{cx + 12}" y="{top_y + 27}" width="4" height="6" rx="1" fill="#555"/>'

    # side-seam slip pockets for joggers/sweatpants (front view only)
    if is_soft_bottom and is_front and 'legging' not in g:
        slip_top_y = top_y + waistband_h + rise * 0.18
        slip_bot_y = slip_top_y + rise * 0.45
        for side_sign in (-1, 1):
            # follow the side seam between waist and hip x positions
            t1 = (slip_top_y - top_y) / rise if rise else 0.4
            t2 = (slip_bot_y - top_y) / rise if rise else 0.7
            waist_x = cx + side_sign * half_waist
            hip_x   = cx + side_sign * half_hip
            x1 = waist_x + (hip_x - waist_x) * t1 - side_sign * 1.5
            x2 = waist_x + (hip_x - waist_x) * t2 - side_sign * 1.5
            extras += (
                f'<path d="M {x1:.1f} {slip_top_y:.1f} Q {x1 - side_sign * 3:.1f} {(slip_top_y+slip_bot_y)/2:.1f}, {x2:.1f} {slip_bot_y:.1f}" '
                f'fill="none" stroke="#555" stroke-width="{STROKE["STITCH"]}" stroke-dasharray="2,2"/>'
            )

    view_id = 'front' if is_front else 'back'
    return f'<g id="{view_id}"><path d="{path}" fill="{fill}" stroke="#000" stroke-width="{STROKE["OUTLINE"]}" stroke-linejoin="round"/>{waistband}{belt_loops}{fly}{seams}{hem}{extras}</g>'

# spec sheet boxes underneath the tops drawing: measurements + material + construction
def _draw_info_boxes(tech_pack, box_y):
    row_h, box_w, header_h = 18, 350, 22
    lbl = 'font-family="Arial" font-size="10" font-weight="bold" fill="#333"'
    val = 'font-family="Arial" font-size="10" fill="#000"'
    hdr = 'font-family="Arial" font-size="11" font-weight="bold" fill="#FFF"'

    # one card with a dark header strip + label/value rows separated by hairlines
    def box(col_x, title, rows):
        rows = [(l, v) for l, v in rows if v]
        h = header_h + row_h * len(rows) + 6
        out = f'<rect x="{col_x}" y="{box_y}" width="{box_w}" height="{h}" rx="3" fill="#FFF" stroke="#CCC" stroke-width="1"/><rect x="{col_x}" y="{box_y}" width="{box_w}" height="{header_h}" rx="3" fill="#333"/><text x="{col_x+box_w/2}" y="{box_y+15}" {hdr} text-anchor="middle">{title}</text>'
        for i, (label, value) in enumerate(rows):
            ry = box_y + header_h + 14 + i * row_h
            out += f'<text x="{col_x+12}" y="{ry}" {lbl}>{label}</text><text x="{col_x+box_w-12}" y="{ry}" {val} text-anchor="end">{value}</text>'
            if i < len(rows) - 1: out += f'<line x1="{col_x+8}" y1="{ry+6}" x2="{col_x+box_w-8}" y2="{ry+6}" stroke="#EEE" stroke-width="0.5"/>'
        return out

    measurements = box(45, 'MEASUREMENTS', [
        ('Chest', f'{getattr(tech_pack, "chest", "-")}"'),
        ('Waist', f'{getattr(tech_pack, "waist", "-")}"'),
        ('Shoulder', f'{getattr(tech_pack, "shoulder", "-")}"'),
        ('Sleeve Length', f'{getattr(tech_pack, "sleeve_length", "-")}"'),
        ('Sleeve Width', f'{getattr(tech_pack, "sleeve_width", "-")}"'),
        ('Body Length', f'{getattr(tech_pack, "body_length", "-")}"'),
    ])
    material = box(425, 'MATERIAL', [('Fabric', getattr(tech_pack, 'fabric_type', '').title()), ('Colour', getattr(tech_pack, 'colour', '').title()), ('Weight', getattr(tech_pack, 'fabric_weight', ''))])
    construction = box(805, 'CONSTRUCTION', [('Fit', getattr(tech_pack, 'fit', '').title()), ('Seam Type', getattr(tech_pack, 'seam_type', '')), ('Closure', getattr(tech_pack, 'closure_type', '')), ('Pockets', getattr(tech_pack, 'pockets', ''))])
    return f'<g id="info-boxes">{measurements}{material}{construction}</g>'

# spec sheet boxes for bottoms - same layout, different field set
def _draw_bottoms_info_boxes(tech_pack, box_y):
    row_h, box_w, header_h = 18, 350, 22
    lbl = 'font-family="Arial" font-size="10" font-weight="bold" fill="#333"'
    val = 'font-family="Arial" font-size="10" fill="#000"'
    hdr = 'font-family="Arial" font-size="11" font-weight="bold" fill="#FFF"'

    def box(col_x, title, rows):
        rows = [(l, v) for l, v in rows if v]
        h = header_h + row_h * len(rows) + 6
        out = f'<rect x="{col_x}" y="{box_y}" width="{box_w}" height="{h}" rx="3" fill="#FFF" stroke="#CCC" stroke-width="1"/><rect x="{col_x}" y="{box_y}" width="{box_w}" height="{header_h}" rx="3" fill="#333"/><text x="{col_x+box_w/2}" y="{box_y+15}" {hdr} text-anchor="middle">{title}</text>'
        for i, (label, value) in enumerate(rows):
            ry = box_y + header_h + 14 + i * row_h
            out += f'<text x="{col_x+12}" y="{ry}" {lbl}>{label}</text><text x="{col_x+box_w-12}" y="{ry}" {val} text-anchor="end">{value}</text>'
            if i < len(rows) - 1: out += f'<line x1="{col_x+8}" y1="{ry+6}" x2="{col_x+box_w-8}" y2="{ry+6}" stroke="#EEE" stroke-width="0.5"/>'
        return out

    measurements = box(45, 'MEASUREMENTS', [
        ('Waist', f'{getattr(tech_pack, "waist", "-")}"'), 
        ('Hips', f'{getattr(tech_pack, "hips", "-")}"'), 
        ('Rise', f'{getattr(tech_pack, "rise", "-")}"'), 
        ('Inseam', f'{getattr(tech_pack, "inseam", "-")}"'), 
        ('Thigh', f'{getattr(tech_pack, "thigh", "-")}"'), 
        ('Leg Opening', f'{getattr(tech_pack, "leg_opening", "-")}"')
    ])
    material = box(425, 'MATERIAL', [('Fabric', getattr(tech_pack, 'fabric_type', '').title()), ('Colour', getattr(tech_pack, 'colour', '').title()), ('Weight', getattr(tech_pack, 'fabric_weight', ''))])
    construction = box(805, 'CONSTRUCTION', [('Fit', getattr(tech_pack, 'fit', '').title()), ('Seam Type', getattr(tech_pack, 'seam_type', '')), ('Closure', getattr(tech_pack, 'closure_type', '')), ('Pockets', getattr(tech_pack, 'pockets', ''))])
    return f'<g id="info-boxes">{measurements}{material}{construction}</g>'