from django.http import HttpResponse

from django.shortcuts import render,redirect

from .forms import PredictionForm

from django.http import JsonResponse

import pickle


# Chargement du modèle
with open('modeleTemporelle/modele_holt_winters.pkl', 'rb') as model_file:



    loaded_model = pickle.load(model_file)




#home page
def home(request):
    return render(request,'index.html')




def predictions_view(request):

  if request.method == 'POST':

     form = PredictionForm(request.POST)

     # la validation du formulaire

     if form.is_valid():

       
       #start_date = form.cleaned_data['start_date']
       #end_date = form.cleaned_data['end_date']

       nbprediction = form.cleaned_data['nombre_prediction']

       # Utilisez le modèle pour faire des prédictions


       #predictions = loaded_model.predict(start=start_date, end=end_date)

       predictions = loaded_model.forecast(steps=nbprediction)



       # cas ou je veux le faire avec les données du dataser


       import pandas as pd

       # Chargez les données d'entraînement
       
       #train = pd.read_csv('data/train.csv')

       

       #reset le formulaire

       form = PredictionForm()

       context={'predictions': predictions.tolist(),'form':form}

       template = 'resultat.html'

       #return JsonResponse({'predictions': predictions.tolist()})  

       return render(request,template,context)

  else:
     
     form = PredictionForm()
     context={'form':form}

     template = 'resultat.html'

     return render(request, template,context)     
    
    