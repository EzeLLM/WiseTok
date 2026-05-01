from django import forms

from .models import CoachApplication

PAYMENT_METHODS = {
    ('visa', 'VISA'),
    ('paypal', 'PAYPAL')
}

class UpgradeProfileForm(forms.Form):
    payment_method = forms.ChoiceField(choices=PAYMENT_METHODS,
                                       required=True,
                                       widget=forms.Select(attrs={
                                           'class':'form-control',
                                           'id': 'payment_id'})
                                       )
    note = forms.CharField(max_length=500,
                           required=False,
                           widget=forms.TextInput(attrs={
                               'class':'form-control',
                               'id': 'note_id',
                               'rows':'5',
                               'cols': '100'})
                           )
