from django.contrib import messages
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import authenticate, login
from django.contrib.auth.models import User
from django.http import HttpResponse, JsonResponse
from django.urls import reverse
from django.views.decorators.http import require_POST
import types

from .forms import TechPackModifyForm
from .models import TechPack
from .svg_generator import generate_garment_svg

# unpacks json_data into flat attributes so the svg generator doesn't have to
class _SvgData:
    def __init__(self, techpack):
        d   = techpack.json_data
        m   = d['measurements']
        mat = d['material']
        sty = d['style']
        con = d['construction']
        self.garment_type  = d['garment_type']
        # tops
        self.chest         = int(m['chest'])         if 'chest'         in m else 0
        self.waist         = int(m['waist'])         if 'waist'         in m else 0
        self.sleeve_length = int(m['sleeve_length']) if 'sleeve_length' in m else 0
        self.body_length   = int(m['body_length'])   if 'body_length'   in m else 0
        self.shoulder      = int(m['shoulder'])      if 'shoulder'      in m else 0
        self.sleeve_width  = int(m['sleeve_width'])  if 'sleeve_width'  in m else 8
        self.tank_divet    = int(m['tank_divet'])    if 'tank_divet'    in m else 10
        # bottoms
        self.hips        = int(m['hips'])        if 'hips'        in m else 0
        self.inseam      = int(m['inseam'])      if 'inseam'      in m else 0
        self.outseam     = int(m['outseam'])     if 'outseam'     in m else 0
        self.rise        = int(m['rise'])        if 'rise'        in m else 0
        self.thigh       = int(m['thigh'])       if 'thigh'       in m else 0
        self.leg_opening = int(m['leg_opening']) if 'leg_opening' in m else 0
        self.colour        = mat['colour']
        self.fabric_type   = mat['fabric_type']
        self.fabric_weight = mat['fabric_weight']
        self.fit           = sty['fit']
        self.length        = sty['length']
        self.seam_type     = con['seam_type']
        self.closure_type  = con['closure_type']
        self.pockets       = con['pockets']


# try loading the ai model at startup
try:
    from .ml_model.inference import TechPackGenerator, extract_tech_pack_fields
    ai_generator = TechPackGenerator()
    AI_AVAILABLE = True
except Exception as e:
    print(f"AI not available: {e}")
    AI_AVAILABLE = False


def _build_preview_svg(tech_pack_data):
    tp  = tech_pack_data.get('tech_pack', tech_pack_data)
    m   = tp['measurements']
    mat = tp['material']
    sty = tp['style']
    con = tp['construction']
    d = types.SimpleNamespace(
        garment_type  = tp['garment_type'],
        chest         = int(m['chest']),
        waist         = int(m['waist']),
        sleeve_length = int(m['sleeve_length']),
        body_length   = int(m['body_length']),
        shoulder      = int(m['shoulder']),
        sleeve_width  = int(m['sleeve_width'])  if 'sleeve_width' in m else 8,
        tank_divet    = int(m['tank_divet'])    if 'tank_divet'   in m else 10,
        hips          = int(m['hips']),
        inseam        = int(m['inseam']),
        outseam       = int(m['outseam']),
        rise          = int(m['rise']),
        thigh         = int(m['thigh']),
        leg_opening   = int(m['leg_opening']),
        colour        = mat['colour'],
        fabric_type   = mat['fabric_type'],
        fabric_weight = mat['fabric_weight'],
        fit           = sty['fit'],
        length        = sty['length'],
        seam_type     = con['seam_type'],
        closure_type  = con['closure_type'],
        pockets       = con['pockets'],
    )
    return generate_garment_svg(d)


def _build_preview_fields(tech_pack_data):
    tp  = tech_pack_data.get('tech_pack', tech_pack_data)
    mat = tp['material']
    sty = tp['style']
    con = tp['construction']
    m   = tp['measurements']
    fields = []
    for key, label, src in [
        ('garment_type', 'Garment',  tp),
        ('colour',       'Colour',   mat),
        ('fabric_type',  'Fabric',   mat),
        ('fabric_weight','Weight',   mat),
        ('fit',          'Fit',      sty),
        ('length',       'Length',   sty),
        ('seam_type',    'Seam',     con),
        ('closure_type', 'Closure',  con),
        ('pockets',      'Pockets',  con),
    ]:
        val = src[key] if isinstance(src, dict) else src
        if val and str(val).lower() not in ('null', 'none', ''):
            fields.append({'key': label, 'val': str(val)})
    from .svg_generator import _is_bottoms
    garment = tp['garment_type']
    if _is_bottoms(garment):
        meas_fields = [
            ('waist','Waist'), ('hips','Hips'), ('rise','Rise'),
            ('inseam','Inseam'), ('outseam','Outseam'),
            ('thigh','Thigh'), ('leg_opening','Leg Open'),
        ]
    else:
        meas_fields = [
            ('chest','Chest'), ('waist','Waist'), ('shoulder','Shoulder'),
            ('sleeve_length','Sleeve'), ('sleeve_width','Sleeve Width'), ('body_length','Body'),
        ]
    for key, label in meas_fields:
        val = m[key] if key in m else None
        if val and int(val) > 0:
            fields.append({'key': label, 'val': f'{val}"'})
    return fields


def get_sidebar_context(request, current_pk=None):
    if request.user.is_authenticated:
        return {
            'all_techpacks': TechPack.objects.filter(user=request.user),
            'current_techpack_id': current_pk
        }
    else:
        return {
            'all_techpacks': [],
            'current_techpack_id': None
        }

def home(request):
    #home page with create and modify and sidebar
    context = get_sidebar_context(request)
    return render(request, 'techpack_generator/home.html', context)

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            next_url = request.POST.get('next') or request.GET.get('next') or '/'
            return redirect(next_url)
        else:
            messages.error(request, 'Invalid username or password.')
    return redirect('/')

@login_required
def techpack_list(request):
    #show all tech packs
    techpacks = TechPack.objects.filter(user=request.user)
    return render(request, 'techpack_generator/list.html', {'techpacks': techpacks })

@login_required
def techpack_detail(request, pk):
    # one tech pack's details
    techpack = get_object_or_404(TechPack, pk=pk, user=request.user)

    from .ml_model.validation import _garment_schema
    garment_svg = generate_garment_svg(_SvgData(techpack))
    is_bottoms  = _garment_schema(techpack.json_data['garment_type']) == 'bottoms'
    context = get_sidebar_context(request, pk)
    context['techpack']     = techpack
    context['garment_svg']  = garment_svg
    context['is_bottoms']   = is_bottoms
    return render(request, 'techpack_generator/detail.html', context)

def _save_techpack_from_result(request, result, description):
    from .ml_model.validation import _garment_schema
    fields = extract_tech_pack_fields(result['tech_pack'], user_input=description)
    is_bottoms = _garment_schema(fields['garment_type']) == 'bottoms'

    if is_bottoms:
        measurements = {
            'waist':       fields['waist'],
            'hips':        fields['hips'],
            'inseam':      fields['inseam'],
            'outseam':     fields['outseam'],
            'rise':        fields['rise'],
            'thigh':       fields['thigh'],
            'leg_opening': fields['leg_opening'],
        }
    else:
        measurements = {
            'chest':         fields['chest'],
            'waist':         fields['waist'],
            'sleeve_length': fields['sleeve_length'],
            'body_length':   fields['body_length'],
            'shoulder':      fields['shoulder'],
            'sleeve_width':  8,
            'tank_divet':    10,
        }

    json_data = {
        'garment_type': fields['garment_type'],
        'material': {
            'fabric_type':   fields['fabric_type'],
            'colour':        fields['colour'],
            'fabric_weight': fields['fabric_weight'],
        },
        'measurements': measurements,
        'construction': {
            'seam_type':    fields['seam_type'],
            'closure_type': fields['closure_type'],
            'pockets':      fields['pockets'],
        },
        'style': {
            'fit':     fields['fit'],
            'length':  fields['length'],
            'details': fields['details'],
        },
    }

    return TechPack.objects.create(
        user=request.user,
        name=f"{fields['garment_type'].title()}",
        is_ai_generated=True,
        prompt=description,
        json_data=json_data,
    )


@login_required
def create_techpack(request):
    if request.method == 'POST':
        action = request.POST.get('action')

        # ── AJAX: generate ──────────────────────────────────────
        if action == 'generate':
            if not AI_AVAILABLE:
                return JsonResponse({'type': 'error', 'message': 'AI model not available.'}, status=503)
            description = request.POST.get('description', '').strip()
            if not description:
                return JsonResponse({'type': 'error', 'message': 'Please enter a description.'}, status=400)
            result = ai_generator.generate(description)
            if not result['success']:
                return JsonResponse({'type': 'error', 'message': result['error']}, status=500)
            # always store the inner dict (without 'tech_pack' wrapper) for consistency
            tp_raw = result['tech_pack']
            tp = tp_raw.get('tech_pack', tp_raw)
            # apply garment-specific measurement defaults so preview matches what will be saved
            corrected = extract_tech_pack_fields(tp, user_input=description)
            m = tp.setdefault('measurements', {})
            for key in ('chest', 'waist', 'sleeve_length', 'body_length', 'shoulder',
                        'hips', 'inseam', 'outseam', 'rise', 'thigh', 'leg_opening'):
                try:
                    if not m.get(key) or int(str(m[key])) == 0:
                        m[key] = corrected[key]
                except (ValueError, TypeError):
                    m[key] = corrected[key]
            # write back user-input-detected style/material fields that the model ignores
            sty = tp.setdefault('style', {})
            mat = tp.setdefault('material', {})
            if corrected['fit']:
                sty['fit'] = corrected['fit']
            if corrected['length']:
                sty['length'] = corrected['length']
            if corrected['colour']:
                mat['colour'] = corrected['colour']
            request.session['ajax_description']    = description
            request.session['ajax_tech_pack']      = tp
            request.session['user_answered_fields'] = []
            follow_ups = result['follow_up_questions']
            if follow_ups:
                from .ml_model.followup import QUESTION_TO_FIELD
                questions = [
                    {'field': QUESTION_TO_FIELD.get(q), 'question': q}
                    for q in follow_ups if QUESTION_TO_FIELD.get(q)
                ]
                return JsonResponse({'type': 'followup', 'questions': questions})
            return JsonResponse({
                'type':   'preview',
                'svg':    _build_preview_svg(tp),
                'fields': _build_preview_fields(tp),
            })

        # ── AJAX: follow-up ─────────────────────────────────────
        if action == 'followup':
            tp = request.session.get('ajax_tech_pack')
            if not tp:
                return JsonResponse({'type': 'error', 'message': 'Session expired. Please try again.'}, status=400)
            answered = {
                k[3:]: v.strip()
                for k, v in request.POST.items()
                if k.startswith('fq_') and v.strip()
            }
            from .ml_model.followup import apply_followup_answers, get_remaining_questions
            updated = apply_followup_answers(tp, answered)
            tp = updated.get('tech_pack', updated)  # always store unwrapped
            request.session['ajax_tech_pack'] = tp
            user_answered = set(request.session.get('user_answered_fields', []))
            user_answered.update(answered.keys())
            request.session['user_answered_fields'] = list(user_answered)
            remaining = get_remaining_questions(tp, user_answered_fields=user_answered)
            if remaining:
                return JsonResponse({'type': 'followup', 'questions': remaining})
            return JsonResponse({
                'type':   'preview',
                'svg':    _build_preview_svg(tp),
                'fields': _build_preview_fields(tp),
            })

        # ── AJAX: save ──────────────────────────────────────────
        if action in ('save', 'save_and_modify'):
            tp_raw      = request.session.get('ajax_tech_pack')
            description = request.session.get('ajax_description', '')
            if not tp_raw:
                return JsonResponse({'type': 'error', 'message': 'Nothing to save.'}, status=400)
            # unwrap in case session still holds the {'tech_pack': {...}} form
            tp = tp_raw.get('tech_pack', tp_raw)
            from .ml_model.validation import _garment_schema
            from .ml_model.inference import _garment_meas_defaults
            garment_type = tp['garment_type']
            is_bottoms   = _garment_schema(garment_type) == 'bottoms'
            mat = tp['material']
            m   = dict(tp['measurements'])
            con = tp['construction']
            sty = tp['style']
            # fill in any measurements the model left out or set to 0
            for field, default_val in _garment_meas_defaults(garment_type).items():
                try:
                    if not m.get(field) or int(m[field]) == 0:
                        m[field] = default_val
                except (TypeError, ValueError):
                    m[field] = default_val
            if is_bottoms:
                measurements = {
                    'waist':       m['waist'],
                    'hips':        m['hips'],
                    'inseam':      m['inseam'],
                    'outseam':     m['outseam'],
                    'rise':        m['rise'],
                    'thigh':       m['thigh'],
                    'leg_opening': m['leg_opening'],
                }
            else:
                measurements = {
                    'chest':         m['chest'],
                    'waist':         m['waist'],
                    'sleeve_length': m['sleeve_length'],
                    'body_length':   m['body_length'],
                    'shoulder':      m['shoulder'],
                    'sleeve_width':  m['sleeve_width'] if 'sleeve_width' in m else 8,
                    'tank_divet':    m['tank_divet']   if 'tank_divet'   in m else 10,
                }
            json_data = {
                'garment_type': garment_type,
                'material': {
                    'fabric_type':   mat['fabric_type'],
                    'colour':        mat['colour'],
                    'fabric_weight': mat['fabric_weight'],
                },
                'measurements': measurements,
                'construction': {
                    'seam_type':    con['seam_type'],
                    'closure_type': con['closure_type'],
                    'pockets':      con['pockets'],
                },
                'style': {
                    'fit':     sty['fit'],
                    'length':  sty['length'],
                    'details': sty['details'],
                },
            }
            techpack = TechPack.objects.create(
                user=request.user,
                name=garment_type.title(),
                is_ai_generated=True,
                prompt=description,
                json_data=json_data,
            )
            request.session.pop('ajax_tech_pack', None)
            request.session.pop('ajax_description', None)
            if action == 'save_and_modify':
                url = reverse('techpack_generator:modify', args=[techpack.pk])
            else:
                url = reverse('techpack_generator:detail', args=[techpack.pk])
            return JsonResponse({'type': 'redirect', 'url': url})

        # ── legacy form: follow-up answers ──────────────────────
        if request.POST.get('answering_followup') == '1':
            original_description = request.session.pop('followup_description', '')
            follow_up_fields     = request.session.pop('followup_fields', [])
            stored_tech_pack     = request.session.pop('followup_tech_pack', None)
            if not original_description:
                messages.error(request, 'Session expired. Please try again.')
                return redirect('techpack_generator:create')

            answered = {field: request.POST.get(f'fq_{field}', '').strip()
                        for field in follow_up_fields}
            answered = {k: v for k, v in answered.items() if v}

            if stored_tech_pack:
                # patch answers into the existing tech pack instead of re-running
                # inference, otherwise fabric/seam/weight from run 1 get clobbered
                from .ml_model.followup import apply_followup_answers
                updated = apply_followup_answers(stored_tech_pack, answered)
                result  = {'success': True, 'tech_pack': updated, 'confidences': {}}
            else:
                from .ml_model.followup import build_enriched_description
                enriched = build_enriched_description(original_description, answered)
                result   = ai_generator.generate(enriched)

            if result['success']:
                techpack = _save_techpack_from_result(request, result, original_description)
                return redirect('techpack_generator:detail', pk=techpack.pk)
            else:
                messages.error(request, f'Generation failed: {result["error"]}')

        else:
            # standard path
            description = request.POST.get('description', '').strip()

            if not description:
                messages.error(request, 'Please enter a description.')
            elif not AI_AVAILABLE:
                context = get_sidebar_context(request)
                context['ai_available'] = False
                context['error'] = 'AI model not available.'
                return render(request, 'techpack_generator/create.html', context)
            else:
                result = ai_generator.generate(description)

                if result['success']:
                    follow_ups = result['follow_up_questions']

                    if follow_ups:
                        from .ml_model.followup import QUESTION_TO_FIELD
                        # work out which fields have questions so the template can
                        # name the inputs as fq_{field} for the follow-up POST
                        follow_up_fields = [QUESTION_TO_FIELD.get(q) for q in follow_ups
                                            if QUESTION_TO_FIELD.get(q)]
                        request.session['followup_description'] = description
                        request.session['followup_fields']      = follow_up_fields
                        request.session['followup_tech_pack']   = result['tech_pack']

                        context = get_sidebar_context(request)
                        context['ai_available'] = AI_AVAILABLE
                        context['follow_up_questions'] = list(zip(follow_up_fields, follow_ups))
                        context['partial_tech_pack']   = result['tech_pack']
                        return render(request, 'techpack_generator/create.html', context)

                    techpack = _save_techpack_from_result(request, result, description)
                    return redirect('techpack_generator:detail', pk=techpack.pk)
                else:
                    messages.error(request, f'Generation failed: {result["error"]}')

    context = get_sidebar_context(request)
    context['ai_available'] = AI_AVAILABLE
    return render(request, 'techpack_generator/create.html', context)

@login_required
def modify_techpack(request, pk):
    techpack = get_object_or_404(TechPack, pk=pk, user=request.user)
    from .ml_model.validation import _garment_schema
    g          = techpack.json_data['garment_type'].lower()
    is_bottoms = _garment_schema(g) == 'bottoms'
    is_tank    = any(t in g for t in ('tank', 'singlet', 'vest'))
    is_skirt   = 'skirt' in g

    if request.method == 'POST':
        mat = techpack.json_data.setdefault('material', {})
        con = techpack.json_data.setdefault('construction', {})
        mat['fabric_type']   = request.POST.get('fabric_type',   mat['fabric_type'])
        mat['fabric_weight'] = request.POST.get('fabric_weight', mat['fabric_weight'])
        mat['colour']        = request.POST.get('colour',        mat['colour'])
        con['seam_type']     = request.POST.get('seam_type',     con['seam_type'])

        if is_bottoms:
            def _pi(name, default=0):
                try: return int(request.POST.get(name, default))
                except (ValueError, TypeError): return default
            techpack.json_data['measurements'] = {
                'waist':       _pi('waist'),
                'hips':        _pi('hips'),
                'rise':        _pi('rise'),
                'inseam':      _pi('inseam'),
                'outseam':     _pi('outseam'),
                'thigh':       _pi('thigh'),
                'leg_opening': _pi('leg_opening'),
            }
            techpack.save()
            messages.success(request, f'Tech Pack "{techpack.name}" updated successfully!')
            return redirect('techpack_generator:detail', pk=techpack.pk)
        else:
            form = TechPackModifyForm(request.POST)
            if form.is_valid():
                techpack.json_data['measurements'] = {
                    'chest':         form.cleaned_data['chest'],
                    'waist':         form.cleaned_data['waist'],
                    'sleeve_length': form.cleaned_data['sleeve_length'],
                    'body_length':   form.cleaned_data['body_length'],
                    'shoulder':      form.cleaned_data['shoulder'],
                    'sleeve_width':  form.cleaned_data['sleeve_width'],
                    'tank_divet':    form.cleaned_data['tank_divet'],
                }
                techpack.save()
                messages.success(request, f'Tech Pack "{techpack.name}" updated successfully!')
                return redirect('techpack_generator:detail', pk=techpack.pk)

    m   = techpack.json_data['measurements']
    mat = techpack.json_data['material']
    con = techpack.json_data['construction']

    if is_bottoms:
        form = None
    else:
        form = TechPackModifyForm(initial={
            'chest':         m['chest'],
            'waist':         m['waist'],
            'sleeve_length': m['sleeve_length'],
            'body_length':   m['body_length'],
            'shoulder':      m['shoulder'],
            'sleeve_width':  m['sleeve_width'] if 'sleeve_width' in m else 8,
            'tank_divet':    m['tank_divet']   if 'tank_divet'   in m else 10,
            'fabric_type':   mat['fabric_type'],
            'fabric_weight': mat['fabric_weight'],
            'colour':        mat['colour'],
            'seam_type':     con['seam_type'],
        })

    context = get_sidebar_context(request, pk)
    context['techpack']     = techpack
    context['form']         = form
    context['is_bottoms']   = is_bottoms
    context['is_tank']      = is_tank
    context['is_skirt']     = is_skirt
    context['measurements'] = m
    context['material']     = mat
    context['construction'] = con
    return render(request, 'techpack_generator/modify.html', context)

@login_required
def preview_svg(request, pk):
    # called by the JS on the modify page every time a slider moves
    # keeps colour/fit/fabric from the saved techpack, swaps in the slider values
    techpack = get_object_or_404(TechPack, pk=pk, user=request.user)
    svg_data = _SvgData(techpack)

    try:
        for attr in ('chest', 'waist', 'body_length', 'sleeve_length', 'shoulder',
                     'sleeve_width', 'tank_divet',
                     'hips', 'rise', 'inseam', 'outseam', 'thigh', 'leg_opening'):
            if attr in request.GET:
                setattr(svg_data, attr, int(request.GET[attr]))
    except (ValueError, TypeError):
        pass

    return HttpResponse(generate_garment_svg(svg_data), content_type='image/svg+xml')


@login_required
def download_svg(request, pk):
    # download the garment svg as a file
    techpack = get_object_or_404(TechPack, pk=pk, user=request.user)
    svg_content = generate_garment_svg(_SvgData(techpack))

    response = HttpResponse(svg_content, content_type='image/svg+xml')
    response['Content-Disposition'] = f'attachment; filename="techpack_{pk}.svg"'
    return response


@login_required
@require_POST
def delete_techpack(request, pk):
    # delete a tech pack owned by the current user
    techpack = get_object_or_404(TechPack, pk=pk, user=request.user)
    techpack.delete()
    return redirect('techpack_generator:home')


def register_view(request):
    # register a new user account
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '').strip()
        password2 = request.POST.get('password2', '').strip()

        if not username or not password:
            messages.error(request, 'Username and password are required.')
        elif password != password2:
            messages.error(request, 'Passwords do not match.')
        elif User.objects.filter(username=username).exists():
            messages.error(request, 'Username already taken.')
        else:
            user = User.objects.create_user(username=username, password=password)
            login(request, user)
            messages.success(request, f'Account created. Welcome, {username}!')
            return redirect('techpack_generator:home')

    return render(request, 'techpack_generator/register.html')
