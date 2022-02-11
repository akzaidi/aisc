"""
Prompt completion API for OpenAI decoder models
"""

import os
import openai
import torch
import openai
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
import json

GPU = 0
if torch.cuda.is_available():
    print("CUDA available!")
    torch.cuda.set_device(GPU)
else:
    print("CUDA unavailable :(")

# source = 'huggingface'  # select from ['openai', 'huggingface']
source = "openai"
# planning_lm_id = 'gpt2-large'  # see comments above for all options
planning_lm_id = "davinci-msft"
translation_lm_id = "stsb-roberta-large"  # see comments above for all options
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# default parameters
MAX_STEPS = 20  # maximum number of steps to be generated
CUTOFF_THRESHOLD = (
    0.8  # early stopping threshold based on matching score and likelihood score
)
P = 0.5  # hyperparameter for early stopping heuristic to detect whether Planning LM believes the plan is finished
BETA = 0.3  # weighting coefficient used to rank generated samples
if source == "openai":
    openai.api_key = OPENAI_KEY
    sampling_params = {
        "max_tokens": 10,
        "temperature": 0.6,
        "top_p": 0.9,
        "n": 10,
        "logprobs": 1,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.3,
        "stop": "\n",
    }
elif source == "huggingface":
    sampling_params = {
        "max_tokens": 10,
        "temperature": 0.1,
        "top_p": 0.9,
        "num_return_sequences": 10,
        "repetition_penalty": 1.2,
        "use_cache": True,
        "output_scores": True,
        "return_dict_in_generate": True,
        "do_sample": True,
    }


class PromptCompletion:
    def __init__(self, engine_id: str = "davinci", max_tokens: int = 256):

        self.engine_id = engine_id
        self.max_tokens = max_tokens

    def complete_prompt(self, prompt: str = "", **kwargs):
        """[summary]

        Parameters
        ----------
        prompt : str, optional
            [description], by default ""
        Allowable kwargs
            {
                "prompt": "Say this is a test",
                "max_tokens": 5,
                "temperature": 1,
                "top_p": 1,
                "n": 1,
                "stream": false,
                "logprobs": null,
                "stop": "\n"
            }


        Returns
        -------
        [type]
            [description]
        """

        return openai.Completion.create(
            engine=self.engine_id, prompt=prompt, max_tokens=self.max_tokens, **kwargs
        )


openai.api_key = os.getenv("OPENAI_API_KEY")

start_sequence = "\nNPC:"
restart_sequence = "\nAdventurer: "

# get back story, and add it to the prompt
with open("backstory.txt", "r") as input_file:
    backstory = input_file.read()
prompt = (
    backstory
    + "\nThe following is a conversation between an Adventurer and an NPC in the Divine City.\n"
)

# initial greeting
greeting = start_sequence + " greetings adventurer! how can i help you?"
print(greeting)
prompt += greeting

while True:
    # get user message
    message = input("Adventurer: ")
    prompt += restart_sequence + message + start_sequence

    # get response from api
    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=prompt,
        temperature=0.9,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=["\n", " NPC:", " Adventurer:"],
    )
    npc_message = response["choices"][0]["text"]
    print(f"NPC:{npc_message}")

    # add response to prompt
    prompt += npc_message + restart_sequence
