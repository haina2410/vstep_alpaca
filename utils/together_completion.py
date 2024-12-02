import os
import random
import re
import string
import together
from typing import Sequence, Union
import tqdm
import math
import logging
import copy
import dotenv
import time
import sys

from generate_instruction import find_word_in_string
from utils.together_decoding_argument import TogetherDecodingArguments

dotenv.load_dotenv()

sources = os.listdir("sources/")


def encode_prompt(prompt_instructions, request_idx: int):
    """Encode multiple prompt instructions into a single string."""
    prompt = open("./prompt_vi.txt").read() + "\n"

    source = open(f"sources/{sources[request_idx % len(sources)]}").read()
    prompt += source + "\n#######################\n" + "Danh sách 20 câu hỏi:\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Câu hỏi: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Câu hỏi:"
    return prompt


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
                        **decoding_kwargs,
                    )

                    if return_text:
                        completions.append(response["output"]["choices"][0]["text"])
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


def post_process_response(num_prompt_instructions, response):
    if response["choices"] is None:
        return []
    response = response["choices"][0]
    raw_instructions = f"{num_prompt_instructions+1}. Câu hỏi:" + response["text"]
    raw_instructions = response["text"]
    raw_instructions = re.split("###", raw_instructions)
    instructions = []
    for idx, inst in enumerate(raw_instructions):
        # if the decoding stops due to length, the last example is likely truncated so we discard it
        if idx == len(raw_instructions) - 1 and response["finish_reason"] == "length":
            continue

        # filter instruction that has unicode escape
        if re.search(r"\\u[0-9a-f]{4}", inst):
            logging.info("Removing: Unicode escape")
            continue

        idx += num_prompt_instructions + 1
        splitted_data = re.split(f"{idx}\.\s+(Câu hỏi|Input|Output):", inst)

        if len(splitted_data) != 7:
            logging.info("Removing: Length mismatch after splitting")
            continue
        else:
            inst = splitted_data[2].strip()
            input = splitted_data[4].strip()
            input = "" if input.lower() == "<noinput>" else input
            output = splitted_data[6].strip()
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            logging.info("Removing: Length too short or too long")
            continue
        # filter based on keywords that are not suitable for language models.
        blacklist = [
            # vietnamese
            "hình ảnh",
            "đồ thị",
            "hình",
            "file",
            "bản đồ",
            "vẽ",
            "minh họa",
            "đi đến",
            "video",
            "âm thanh",
            "nhạc",
            "biểu đồ",
            "sơ đồ",
        ]
        blacklist += []
        if any(find_word_in_string(word, inst) for word in blacklist):
            logging.info("Removing: Blacklisted keyword")
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if (
            inst.startswith("Write a program")
            or inst.startswith("Viết một chương trình")
            or inst.startswith("Câu hỏi: ")
            or inst.startswith("Input: ")
            or inst.startswith("Trả lời ")
        ):
            logging.info("Removing: Starting pattern")
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append({"instruction": inst, "input": input, "output": output})
    return instructions
