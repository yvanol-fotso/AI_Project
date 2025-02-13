import json
import random

def add_fields_to_questions(input_file, output_file):
    # Lire le fichier JSON
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Niveaux de difficulté possibles
    difficulties = ['facile', 'moyen', 'difficile']
    
    # Durées de réponse possibles en minutes
    response_durations = [1, 2, 3]
    
    # Ajouter les nouveaux champs à chaque question
    for question in data['questions']:
        question['difficulte'] = random.choice(difficulties)
        question['duree_reponse'] = f"{random.choice(response_durations)} minute(s)"
    
    # Écrire les modifications dans un nouveau fichier JSON
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Utilisation de la fonction
input_file = 'data.json'
output_file = 'data_modified.json'
add_fields_to_questions(input_file, output_file)
