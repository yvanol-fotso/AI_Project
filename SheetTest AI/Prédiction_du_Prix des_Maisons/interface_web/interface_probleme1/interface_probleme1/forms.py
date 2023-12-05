from django import forms


# formulaire prediction de pri


class PredictionForm(forms.Form):

  #parametre d'entrer du modele pour la prediction

   rm = forms.FloatField()
   ltsat = forms.FloatField()
   ptratio = forms.FloatField()
