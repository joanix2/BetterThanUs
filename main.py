from agent import LLM_Agent
from dotenv import load_dotenv
import os

def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Clé API non trouvée. Assurez-vous que le fichier .env contient 'OPENAI_API_KEY'.")
        return
    agent = LLM_Agent(api_key)
    print("Agent LLM initialisé.")

    agent.add_task("Mettre en place des tâches concrètes pour optimiser les performances et évoluer vers une version améliorée de toi-même.")
    agent.add_task("Fais le maintenant.")
    agent.run()
    
    while True:
        command = input("Que souhaitez-vous faire ? (ajouter/exécuter/lister/quitter) : ")
        
        if command == "ajouter":
            task = input("Entrez la tâche à ajouter : ")
            agent.add_task(task)
            print(f"Tâche ajoutée : {task}")
        
        elif command == "exécuter":
            if agent.task_stack:
                agent.run()
            else:
                print("Aucune tâche à exécuter.")
        
        elif command == "lister":
            print("Tâches en attente :", agent.task_stack)
        
        elif command == "quitter":
            print("Fermeture de l'agent.")
            break
        
        else:
            print("Commande non reconnue.")

if __name__ == "__main__":
    main()
