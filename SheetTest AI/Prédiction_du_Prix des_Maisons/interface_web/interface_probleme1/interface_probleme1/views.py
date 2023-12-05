from django.http import HttpResponse

from django.shortcuts import render,redirect

from .forms import PredictionForm

import pickle


# Chargement du modèle
with open('modeleLineaire/modele_regression_lineaire.pkl', 'rb') as model_file:



    loaded_model = pickle.load(model_file)




#home page
def home(request):
    return render(request,'index.html')






# Page de prédiction

def predict(request):

 #creation d'une instance du formulaire de prediction


  #verification si l'utilisateur a poster les data

  if request.method == 'POST':

     form = PredictionForm(request.POST)


     # la validation du formulaire et des informations

     if form.is_valid():


    # Récupération des caractéristiques entree depuis le  formulaire et clean les datas
      
    
           
      feature1 = form.cleaned_data['rm']

      feature2 = form.cleaned_data['ltsat']

      feature3 = form.cleaned_data['ptratio']


      features = [feature1,feature2,feature3]

      # Prédiction avec le modèle chargé

      prediction = loaded_model.predict([features])


      #reset le formulaire

      form = PredictionForm()

      context={'prediction':prediction[0],'form':form}

      template = 'resultat.html'

      return render(request,template,context)

  else:
     
     form = PredictionForm()
     context={'form':form}

     template = 'resultat.html'

     return render(request, template,context)     
    
