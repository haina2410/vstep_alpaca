import together
from typing import Optional, Sequence, Union
from dataclasses import dataclass
import tqdm
import math
import logging
import copy
import dotenv
import time
import sys

dotenv.load_dotenv()

@dataclass
class TogetherDecodingArguments:
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[Sequence[str]] = None
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

def together_completion(
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: TogetherDecodingArguments,
    model_name="meta-llama/Llama-Vision-Free",
    sleep_time=2,
    batch_size=1,
    max_instances=sys.maxsize,
    return_text=False,
    **decoding_kwargs,
):
    """Complete prompts using Together's API with Llama model"""
    
    is_single_prompt = isinstance(prompts, (str, dict))
    if is_single_prompt:
        prompts = [prompts]

    prompts = prompts[:max_instances]
    num_prompts = len(prompts)
    prompt_batches = [
        prompts[batch_id * batch_size : (batch_id + 1) * batch_size]
        for batch_id in range(int(math.ceil(num_prompts / batch_size)))
    ]

    completions = []
    for prompt_batch in tqdm.tqdm(prompt_batches, desc="prompt_batches"):
        batch_decoding_args = copy.deepcopy(decoding_args)

        while True:
            try:
                for prompt in prompt_batch:
                    response = together.Complete.create(
                        prompt=prompt,
                        model=model_name,
                        max_tokens=batch_decoding_args.max_tokens,
                        temperature=batch_decoding_args.temperature,
                        top_p=batch_decoding_args.top_p,
                        n=batch_decoding_args.n,
                        stop=batch_decoding_args.stop,
                        frequency_penalty=batch_decoding_args.frequency_penalty,
                        presence_penalty=batch_decoding_args.presence_penalty,
                        **decoding_kwargs
                    )
                    
                    if return_text:
                        completions.append(response['output']['choices'][0]['text'])
                    else:
                        completions.append(response)
                break
            except Exception as e:
                logging.warning(f"Together API Error: {e}")
                time.sleep(sleep_time)

    if decoding_args.n > 1:
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:
        (completions,) = completions
        
    return completions