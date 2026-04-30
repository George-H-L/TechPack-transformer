from django import forms 


class TechPackManualForm(forms.Form):
    """Form for manually entering tech pack details."""

    name = forms.CharField(max_length=200, 
            widget=forms.TextInput
            (attrs={'class': 'form-input', 
            'placeholder': 'i.e, blue jeans, black puffer'}), 
            label='Tech Pack Name')
    
    garment_type = forms.CharField(max_length=100,
            widget=forms.TextInput(
            attrs={'class': 'form-input', 
            'placeholder': 'i.e, jeans, puffer jacket'}), 
            label='Garment Type')
    
    fabric_type = forms.CharField(max_length=100,
            widget=forms.TextInput(attrs={'class': 'form-input', 
            'placeholder': 'i.e, denim, nylon'}), 
            label='Fabric Type')
    
    fabric_weight = forms.CharField(max_length=100,
            widget=forms.TextInput(attrs={'class': 'form-input', 
            'placeholder': 'i.e, 100gsm, 200gsm'}), 
            label='Fabric Weight')
    
    colour = forms.CharField(max_length=50,
            widget=forms.TextInput(attrs={'class': 'form-input', 
            'placeholder': 'i.e, dark blue, black'}), 
            label='Colour')

    chest = forms.CharField(max_length=10,
            widget=forms.TextInput(attrs={'class': 'form-input',
            'placeholder': 'i.e, 38'}),
            label='Chest (cm)')

    waist = forms.CharField(max_length=10,
            widget=forms.TextInput(attrs={'class': 'form-input',
            'placeholder': 'i.e, 80'}),
            label='Waist (cm)')

    sleeve_length = forms.CharField(max_length=10,
            widget=forms.TextInput(attrs={'class': 'form-input',
            'placeholder': 'i.e, 25'}),
            label='Sleeve Length (cm)')

    body_length = forms.CharField(max_length=10,
            widget=forms.TextInput(attrs={'class': 'form-input',
            'placeholder': 'i.e, 28'}),
            label='Body Length (cm)')
    
    seam_type = forms.CharField(max_length=100,
            widget=forms.TextInput(attrs={'class': 'form-input', 
            'placeholder': 'i.e, Overlock, flat-felled'}), 
            label='Seam Type')
    
    closure_type = forms.CharField(max_length=100,
            widget=forms.TextInput(attrs={'class': 'form-input', 
            'placeholder': 'i.e, zipper, buttons'}), 
            label='Closure Type')
    
    pockets = forms.CharField(max_length=200,
            widget=forms.TextInput(attrs={'class': 'form-input',
            'placeholder': 'i.e, patch pockets, welt pockets'}),
            label='Pockets')


class TechPackModifyForm(forms.Form):
    """Form for modifying existing tech pack details."""

    chest = forms.CharField(max_length=10,
            widget=forms.TextInput(attrs={'class': 'form-input',
            'placeholder': 'i.e, 38'}),
            label='Chest (cm)')

    waist = forms.CharField(max_length=10,
            widget=forms.TextInput(attrs={'class': 'form-input',
            'placeholder': 'i.e, 80'}),
            label='Waist (cm)')

    sleeve_length = forms.CharField(max_length=10,
            widget=forms.TextInput(attrs={'class': 'form-input',
            'placeholder': 'i.e, 25'}),
            label='Sleeve Length (cm)')

    body_length = forms.CharField(max_length=10,
            widget=forms.TextInput(attrs={'class': 'form-input',
            'placeholder': 'i.e, 28'}),
            label='Body Length (cm)')

    fabric_type = forms.CharField(max_length=100,
            widget=forms.TextInput(attrs={'class': 'form-input',
            'placeholder': 'i.e, denim, nylon'}),
            label='Fabric Type')

    fabric_weight = forms.CharField(max_length=100,
            widget=forms.TextInput(attrs={'class': 'form-input',
            'placeholder': 'i.e, 100gsm, 200gsm'}),
            label='Fabric Weight')

    colour = forms.CharField(max_length=50,
            widget=forms.TextInput(attrs={'class': 'form-input',
            'placeholder': 'i.e, dark blue, black'}),
            label='Colour')

    shoulder = forms.CharField(max_length=10,
            widget=forms.TextInput(attrs={'class': 'form-input',
            'placeholder': 'i.e, 18'}),
            label='Shoulder (cm)')

    sleeve_width = forms.CharField(max_length=10,
            widget=forms.TextInput(attrs={'class': 'form-input',
            'placeholder': 'i.e, 13'}),
            label='Sleeve Width (cm)')

    tank_divet = forms.CharField(max_length=10,
            widget=forms.TextInput(attrs={'class': 'form-input',
            'placeholder': '0–10'}),
            label='Armhole Style',
            required=False)

    seam_type = forms.CharField(max_length=100,
            widget=forms.TextInput(attrs={'class': 'form-input',
            'placeholder': 'i.e, Overlock, flat-felled'}),
            label='Seam Type')


