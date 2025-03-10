import inspect
import json
import os
import traceback
from openai import OpenAI

class LLM_Agent:
    def __init__(self, api_key):
        self.task_stack = []  # Pile de tâches
        self.knowledge_base = {}  # Stocke les fonctions créées
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key) # , base_url="https://api.deepseek.com")

        self.system_prompt = self.load_system_prompt(os.path.join("prompts","system.txt"))
        self.tools = self.load_tools("tools")
        self.messages = [{"role": "system", "content": self.system_prompt }]
        
        print(self.tools)

    def load_system_prompt(self, prompt_file):
        """Charge le prompt système depuis un fichier texte."""
        if os.path.exists(prompt_file):
            with open(prompt_file, "r", encoding="utf-8") as f:
                return f.read().strip()
        return "Tu es un agent autonome qui cherche à s'améliorer continuellement. Tu peux segmenter les tâches, générer du code et modifier ton propre prompt système pour optimiser tes performances."

    def load_tools(self, tools_directory):
        """Charge tous les outils depuis le dossier tools."""
        tools = []
        if os.path.exists(tools_directory):
            for filename in os.listdir(tools_directory):
                if filename.endswith(".json"):
                    with open(os.path.join(tools_directory, filename), "r", encoding="utf-8") as f:
                        try:
                            tool = json.load(f)
                            tools.append(tool)
                        except json.JSONDecodeError:
                            print(f"Erreur de chargement du fichier {filename}")
        return tools

    def add_task(self, task):
        """Ajoute une tâche à la pile"""
        self.task_stack.append(task)
        return f"Tâche ajoutée : {task}"

    def add_tasks(self, tasks):
        """Ajoute une tâche à la pile"""
        for task in tasks : self.task_stack.append(task)
        return f"Tâches ajoutées : {tasks}"

    def get_next_task(self):
        """Récupère la prochaine tâche à exécuter"""
        return self.task_stack.pop(0) if self.task_stack else None
    
    def update_system_prompt(self, new_prompt):
        """Permet à l'agent de modifier son propre prompt système"""
        self.system_prompt = new_prompt
        return "Prompt système mis à jour avec succès."

    def ask_gpt(self, prompt, model="gpt-4o"):
        """Utilise GPT pour prendre une décision ou générer du code avec les outils disponibles"""
        try:
            self.messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=model,
                messages=self.messages,
                tools=self.tools,
                tool_choice="auto",  # Permet à GPT de choisir un outil s'il en a besoin
                stream=False
            )

            self.messages.append(response.choices[0].message)

            tool_calls = response.choices[0].message.tool_calls
            if tool_calls and tool_calls.__len__() > 0:
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_args = json.loads(tool_call.function.arguments)
                    result = self.execute_tool(tool_name, **tool_args)
                    print(f"Résultat de l'outil '{tool_name}' : {result}")
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })

                # Ajouter une tâche de validation après l'exécution des outils
                # self.add_task("Vérifier que l'étape précédente a été correctement réalisée")

                # Relancer la requête GPT avec les réponses des outils
                response = self.client.chat.completions.create(
                    model=model,
                    messages=self.messages,
                    tools=self.tools,
                    stream=False
                )

                self.messages.append(response.choices[0].message)
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Erreur avec GPT : {traceback.format_exc()}"
        
    def execute_tool(self, function_name, **kwargs):
        """Exécute une fonction dynamique si elle est définie dans la classe."""
        func = getattr(self, function_name, None)
        if callable(func):
            try:
                sig = inspect.signature(func)
                params = sig.parameters
                args = {k: v for k, v in kwargs.items() if k in params}
                return func(**args)
            except Exception as e:
                return f"Erreur lors de l'exécution : {traceback.format_exc()}"
        else:
            return f"Outil '{function_name}' non trouvé"

    def create_tool(self, function_code, tool_definition):
        """Crée un nouvel outil et l'ajoute à l'agent."""
        tool_name = tool_definition["function"]["name"]
        
        # Sauvegarde du fichier JSON du tool
        tools_directory = "tools"
        os.makedirs(tools_directory, exist_ok=True)
        tool_path = os.path.join(tools_directory, f"{tool_name}.json")
        with open(tool_path, "w") as f:
            json.dump(tool_definition, f, indent=4)
        
        # Sauvegarde du fichier Python de la fonction
        functions_directory = "functions"
        os.makedirs(functions_directory, exist_ok=True)
        function_path = os.path.join(functions_directory, f"{tool_name}.py")
        with open(function_path, "w") as f:
            f.write(function_code)
        
        # Ajout du tool à la liste des outils
        self.tools.append(tool_definition)
        
        # Ajout de la fonction à la base de connaissances
        exec(function_code, globals())
        self.knowledge_base[tool_name] = globals().get(tool_name)
        
        return f"Outil '{tool_name}' créé et ajouté avec succès."

    def run(self):
        """Boucle principale de l'agent"""
        while self.task_stack:
            task = self.get_next_task()
            print(f"Exécution de la tâche : {task}")
            decision = self.ask_gpt(f"Comment dois-je accomplir cette tâche : {task} ?")
            print(f"Décision GPT : {decision}")