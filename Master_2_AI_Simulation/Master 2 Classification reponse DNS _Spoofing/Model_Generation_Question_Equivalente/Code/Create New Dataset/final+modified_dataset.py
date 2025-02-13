import json
import random

def add_fields_to_questions(input_file, output_file):
    # Lire le fichier JSON
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Niveaux de difficulté possibles (1: facile, 2: moyen, 3: difficile)
    difficulties = [1, 2, 3]
    
    # Durées de réponse possibles en minutes (1 à 5 minutes)
    response_durations = [1, 2, 3, 4, 5]
    
    # Ajouter les nouveaux champs à chaque question
    for question in data['questions']:
        question['difficulte'] = random.choice(difficulties)
        question['duree_reponse'] = f"{random.choice(response_durations)} minute(s)"
        question['marks'] = random.randint(1, 3)
    
    # Écrire les modifications dans un nouveau fichier JSON
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# Utilisation de la fonction
input_file = 'data.json'
output_file = 'data_modified_final.json'
add_fields_to_questions(input_file, output_file)
