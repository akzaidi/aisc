"""
Prompt completion API for OpenAI decoder models
"""

from typing import Dict

import os
import openai
import torch
import openai
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_utils
from dotenv import load_dotenv

load_dotenv()

# default parameters
source = "openai"
# planning_lm_id = 'gpt2-large'  # see comments above for all options
# available_engines = openai.Engine.list()
planning_lm_id = "davinci-msft"


MAX_STEPS = 20  # maximum number of steps to be generated
CUTOFF_THRESHOLD = (
    0.8  # early stopping threshold based on matching score and likelihood score
)
P = 0.5  # hyperparameter for early stopping heuristic to detect whether Planning LM believes the plan is finished
BETA = 0.3  # weighting coefficient used to rank generated samples
if source == "openai":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    sampling_params = {
        "max_tokens": 64,
        "temperature": 0.7,
        "top_p": 1.0,
        "n": 10,
        "logprobs": 1,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.3,
        "stop": "\n",
    }
elif source == "huggingface":
    sampling_params = {
        "max_tokens": 256,
        "temperature": 0.1,
        "top_p": 0.9,
        "num_return_sequences": 10,
        "repetition_penalty": 1.2,
        "use_cache": True,
        "output_scores": True,
        "return_dict_in_generate": True,
        "do_sample": True,
    }
else:
    raise ValueError("source must be either 'openai' or 'huggingface'")


def lm_engine(source: str, planning_lm_id: str, device):
    if source == "huggingface":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(planning_lm_id)
        model = AutoModelForCausalLM.from_pretrained(
            planning_lm_id, pad_token_id=tokenizer.eos_token_id
        ).to(device)

    def _generate(prompt, sampling_params: Dict = sampling_params):
        if source == "openai":
            response = openai.Completion.create(
                engine=planning_lm_id, prompt=prompt, **sampling_params
            )
            generated_samples = [
                response["choices"][i]["text"] for i in range(sampling_params["n"])
            ]
            # calculate mean log prob across tokens
            mean_log_probs = [
                np.mean(response["choices"][i]["logprobs"]["token_logprobs"])
                for i in range(sampling_params["n"])
            ]
        elif source == "huggingface":
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            prompt_len = input_ids.shape[-1]
            output_dict = model.generate(
                input_ids,
                max_length=prompt_len + sampling_params["max_tokens"],
                **sampling_params,
            )
            # discard the prompt (only take the generated text)
            generated_samples = tokenizer.batch_decode(
                output_dict.sequences[:, prompt_len:]
            )
            # calculate per-token logprob
            vocab_log_probs = torch.stack(output_dict.scores, dim=1).log_softmax(
                -1
            )  # [n, length, vocab_size]
            token_log_probs = (
                torch.gather(
                    vocab_log_probs, 2, output_dict.sequences[:, prompt_len:, None]
                )
                .squeeze(-1)
                .tolist()
            )  # [n, length]
            # truncate each sample if it contains '\n' (the current step is finished)
            # e.g. 'open fridge\n<|endoftext|>' -> 'open fridge'
            for i, sample in enumerate(generated_samples):
                stop_idx = sample.index("\n") if "\n" in sample else None
                generated_samples[i] = sample[:stop_idx]
                token_log_probs[i] = token_log_probs[i][:stop_idx]
            # calculate mean log prob across tokens
            mean_log_probs = [
                np.mean(token_log_probs[i])
                for i in range(sampling_params["num_return_sequences"])
            ]
        generated_samples = [sample.strip().lower() for sample in generated_samples]
        return generated_samples, mean_log_probs

    return _generate


if __name__ == "__main__":

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    default_prompt = """Hello, my name is GPT and"""
    import argparse

    parser = argparse.ArgumentParser(description="Language Model completion arena")
    parser.add_argument(
        "--prompt",
        type=str,
        default=default_prompt,
        help="Prompt to initialize language model",
    )
    parser.add_argument(
        "--engine-id",
        type=str,
        default=planning_lm_id,
        help="Engine ID, i.e., davinici, babbage",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="openai",
        help="Source of langauge model, i.e., openai, huggingface",
    )

    args, _ = parser.parse_known_args()
    generator = lm_engine(args.source, args.engine_id, device)
    response = generator(args.prompt, sampling_params)
    print(response)

    def generate_tree(
        tree_len: int = 10,
        prompt: str = default_prompt,
        lm_model=generator,
        params: Dict = sampling_params,
    ):

        iteration = 0
        while iteration < tree_len:
            prompt += lm_model(prompt, params)[0][0]
            iteration += 1

        return prompt

    # TODO: create a tree structure to capture prompt trees
    generation = generate_tree()
    print(generation)
