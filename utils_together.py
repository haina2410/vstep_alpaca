from together import Together
from together.types import ChatCompletionResponse
from typing import List, Optional, Sequence, Union
from dataclasses import dataclass
from torch import le
import tqdm
import math
import logging
import copy
import dotenv
import time
import sys

dotenv.load_dotenv()

together = Together()


@dataclass
class TogetherDecodingArguments:
    max_tokens: int = 1800
    temperature: float = 0.2
    top_p: float = 1.0
    n: int = 1
    stream: bool = False
    stop: Optional[List[str]] = None
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
                    response = together.chat.completions.create(
                        messages=[
                            {
                                "role": "system",
                                "content": open("system_prompt.txt", "r").read(),
                            },
                            {
                                "role": "user",
                                "content": prompt,
                            },
                        ],
                        model=model_name,
                        max_tokens=batch_decoding_args.max_tokens,
                        temperature=batch_decoding_args.temperature,
                        top_p=batch_decoding_args.top_p,
                        n=batch_decoding_args.n,
                        stop=batch_decoding_args.stop,
                        frequency_penalty=batch_decoding_args.frequency_penalty,
                        presence_penalty=batch_decoding_args.presence_penalty,
                        **decoding_kwargs,
                    )

                    if isinstance(response, ChatCompletionResponse):
                        if return_text and response.choices:
                            completions.append(response.choices[0].message.content)
                        else:
                            completions.append(response)
                    else:
                        for chunk in response:
                            if return_text and chunk.choices:
                                completions.append(chunk.choices[0].message.content)
                            else:
                                completions.append(chunk)

                    open(f"completions/result_{len(completions)}.txt", "w").write(
                        completions[-1].choices[0].message.content
                    )
                break
            except Exception as e:
                logging.warning(f"Together API Error: {e}")
                time.sleep(sleep_time)

    if decoding_args.n > 1:
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:
        (completions,) = completions

    return completions
