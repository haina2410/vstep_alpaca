import json
import os
import random
import re
import string
from pydantic import BaseModel, Field
from together import Together
from together.types import ChatCompletionResponse
from typing import List, Sequence, Union, Any
import tqdm
import math
import logging
import copy
import dotenv
import time
import sys

from generate_instruction import find_word_in_string

from .together_decoding_argument import TogetherDecodingArguments

dotenv.load_dotenv()

together = Together()


class InstructionEntry(BaseModel):
    instruction: str = Field(description="Câu hỏi về kỳ thi VSTEP")
    input: str = Field(description="input cho câu hỏi, nếu không có input thì để <noinput>")
    output: str = Field(description="Câu trả lời cho câu hỏi")


class GeneratedData(BaseModel):
    list_data: List[InstructionEntry] = Field(description="Danh sách các câu hỏi và câu trả lời")


def encode_prompt(prompt_instructions):
    """Encode multiple prompt instructions into a single string."""
    prompt = open("./prompt_vi.txt").read() + "\n"

    sources = os.listdir("sources/")
    source = open(f"sources/{random.choice(sources)}").read()
    prompt += source + "\n#######################\n" + "Danh sách 20 câu hỏi:\n"

    for idx, task_dict in enumerate(prompt_instructions):
        (instruction, input, output) = task_dict["instruction"], task_dict["input"], task_dict["output"]
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        input = "<noinput>" if input.lower() == "" else input
        prompt += f"###\n"
        prompt += f"{idx + 1}. Instruction: {instruction}\n"
        prompt += f"{idx + 1}. Input:\n{input}\n"
        prompt += f"{idx + 1}. Output:\n{output}\n"
    prompt += f"###\n"
    prompt += f"{idx + 2}. Instruction:"
    return prompt


def together_completion(
    prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
    decoding_args: TogetherDecodingArguments,
    model_name="meta-llama/Llama-Vision-Free",
    sleep_time=2,
    batch_size=1,
    request_idx=0,
    max_instances=sys.maxsize,
    **decoding_kwargs,
) -> List[dict[str, Any]]:
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
                                "content": "Hãy tiếp tục sinh các bộ data dưới định dạng JSON.",
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
                        response_format={
                            "type": "json_object",
                            "schema": GeneratedData.model_json_schema(),
                        },
                        **decoding_kwargs,
                    )

                    if isinstance(response, ChatCompletionResponse):
                        if response.choices:
                            json_data = json.loads(response.choices[0].message.content)
                            completions += json_data["list_data"]

                            open(f"completions/response_{request_idx}_{len(completions)}.json", "w").write(
                                json.dumps(json_data, indent=2, ensure_ascii=False)
                            )
                    else:
                        for chunk in response:
                            if chunk.choices:
                                completions.append(chunk.choices[0].message.content)

                break
            except Exception as e:
                logging.warning(f"Together API Error: {e}")
                time.sleep(sleep_time)

    if decoding_args.n > 1:
        completions = [completions[i : i + decoding_args.n] for i in range(0, len(completions), decoding_args.n)]
    if is_single_prompt:
        (completions,) = completions

    return completions


def post_process_response(num_prompt_instructions, responses: List[dict[str, str]]):
    instructions = []
    for inst in responses:
        instruction = inst["instruction"]

        if instruction.strip() == "":
            logging.info("Removing: Empty")
            continue

        # filter instruction that has unicode escape
        if re.search(r"\\u[0-9a-f]{4}", instruction):
            logging.info("Removing: Unicode escape")
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
        if any(find_word_in_string(word, instruction) for word in blacklist):
            logging.info("Removing: Blacklist")
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result.
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if (
            instruction.startswith("Write a program")
            or instruction.startswith("Viết một chương trình")
            or instruction.startswith("Câu hỏi: ")
            or instruction.startswith("Input: ")
            or instruction.startswith("Trả lời ")
        ):
            continue
        # filter those starting with punctuation
        if instruction[0] in string.punctuation:
            continue

        instructions.append(inst)
    return instructions
