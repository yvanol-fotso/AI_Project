from django import forms


# formulaire prediction de pri


class PredictionForm(forms.Form):

  # si je veux predire les dis dernier mois il me faut plutot un input int pour dix par contre si c'est pour une date precise alors juste la date
   nombre_prediction = forms.FloatField() 
   



class PredictionFormTwo(forms.Form):
    start_date = forms.DateField(widget=forms.DateInput(attrs={"type":"date"}))
    end_date = forms.DateField(widget=forms.DateInput(attrs={"type":"date"}))
    nombre_prediction = forms.IntegerField()

