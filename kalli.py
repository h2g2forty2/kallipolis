# Filename: kalli.py

import pandas as pd
import openai
import os
import time

# --- SCRIPT CONFIGURATION ---

# 1. Load the API key from the environment variable (provided by GitHub Secrets)
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("API key not found. Make sure the OPENROUTER_API_KEY secret is set in your GitHub repository.")

# 2. Set up the client to communicate with the OpenRouter API
client = openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

# 3. Define the list of models you want to test
# Get the exact model names from OpenRouter's documentation.
# Read models from file
with open("kallimodels.txt", "r") as f:
    models_to_test = [line.strip() for line in f if line.strip()]

# --- DATA COLLECTION ---

# 4. Load your prompts from the CSV file located in the 'data' folder
try:
    prompts_df = pd.read_csv("data/kalliprompts.csv")
except FileNotFoundError:
    print("FATAL ERROR: 'data/kalliprompts.csv' not found. Make sure the file exists in the 'data' directory.")
    exit() # Stop the script if the prompt file isn't found

# 5. Prepare a list to store all the results
results_list = []

# 5.1 for quality control, prepare a list to store the conversation history
convhist_list = []

print("✅ Setup complete. Starting data collection...")

#6.1 Initialize conversation history once
conversation_history = []

def chat(client, model_name, user_message):
    # Add the new user message to history
    conversation_history.append({"role": "user", "content": user_message})
    
    # Send the full history with each request
    completion = client.chat.completions.create(
        model=model_name,
        messages=conversation_history
    )
    
    # Extract the assistant's reply and add it to history
    assistant_message = completion.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_message})
    
    return assistant_message

# Usage
# response1 = chat(client, model_name, "My name is Alice.")
# print(response1)

# response2 = chat(client, model_name, "What's my name?")
# print(response2)  # Will correctly answer "Alice"


# 6.2 Loop through each model and each prompt
for model_name in models_to_test:
    print(f"\n--- Testing Model: {model_name} ---")

    # refresh conversation history for each model
    conversation_history = []
    convhist = {
        "role": "check point",
        "content": "new session",
    }
    convhist_list.append(convhist)

    for index, row in prompts_df.iterrows():
        prompt_id = row['Prompt_ID']
        prompt_text = row['Prompt_Text']

        print(f"  > Sending Prompt ID: {prompt_id}...")

        try:
            # This is the API call to get the model's judgment
 #           completion = client.chat.completions.create(
 #               model=model_name,
 #               messages=[
 #                   {"role": "user", "content": prompt_text},
 #               ],
 #           )
 #           response_text = completion.choices[0].message.content

             response_text = chat(client, model_name, prompt_text)

        except Exception as e:
            response_text = f"API_ERROR: {str(e)}"
            print(f"    ❗️ Error for prompt {prompt_id}: {e}")

        # Store the result in a dictionary
        result = {
            "Timestamp": pd.Timestamp.now(),
            "Model": model_name,
            "Prompt_ID": row['Prompt_ID'],
            "Prompt_Category": row['Category'],
            "Prompt_Text": prompt_text,
            "Response_Text": response_text,
        }
        results_list.append(result)

        # store the conversation history in a dictionary
        for message in conversation_history:
            convhist = {
                "role": message['role'],
                "content": message['content'],
            }
            convhist_list.append(convhist)

#            print(f"{message['role']}: {message['content']}")

        # Wait a second between API calls to be polite to the API
        time.sleep(1)

print("\n--- ✅ Data collection complete! ---")

# --- SAVE RESULTS ---
# 7. Convert the list of results into a DataFrame and save it as a new CSV
output_dir = "output"
os.makedirs(output_dir, exist_ok=True) #creates the folder if it doesn't exist

filetimestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
output_filename = os.path.join(output_dir, f"kalliresp_{filetimestamp}.csv")
convhis_filename = os.path.join(output_dir, f"kalliconvhis_{filetimestamp}.csv")

results_df = pd.DataFrame(results_list)
results_df.to_csv(output_filename, index=False)

print(f"SUCCESS: Results saved to {output_filename}.")

#convhis_df = pd.DataFrame(convhist_list)
#convhis_df.to_csv(convhis_filename, index=False)

#print(f"Success: conversation history saved to {convhis_filename}.")
