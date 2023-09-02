import os
import re
import json
import openai
import backoff
import tiktoken
import argparse

from pathlib import Path
from datasets import load_dataset
from tenacity import retry, stop_after_attempt, wait_random_exponential


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt-3.5-turbo', choices=['gpt-3.5-turbo', 'gpt-4'], type=str)
    parser.add_argument('--data_load_name', default='code_smell_data.jsonl',
                        choices=['code_review_data.jsonl', 'code_smell_data.jsonl', 'code_test_data.jsonl'], type=str)
    parser.add_argument('--result_save_name', default='code_smell_eval_gpt3.jsonl',
                        choices=['code_review_eval_gpt3.jsonl', 'code_smell_eval_gpt3.jsonl',
                                 'code_test_data_gpt3.jsonl', 'code_review_eval_gpt4.jsonl',
                                 'code_smell_eval_gpt4.jsonl', 'code_test_data_gpt4.jsonl'], type=str)
    args = parser.parse_args()

    return args


# References: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_format_inputs_to_ChatGPT_models.ipynb
def num_tokens_from_messages(messages, model):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print('Warning: model not found. Using cl100k_base encoding.')
        encoding = tiktoken.get_encoding('cl100k_base')

    tokens_per_message = 4
    tokens_per_name = -1
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == 'name':
                num_tokens += tokens_per_name
    num_tokens += 3

    return num_tokens


# References: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
# @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def generate_text(model, prompt, temperature):
    messages = [{'role': 'user', 'content': prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature
    )

    print('input tokens:', response['usage']['prompt_tokens'])
    print('output tokens:', response['usage']['completion_tokens'])
    print('total tokens:', response['usage']['total_tokens'])

    return response['choices'][0]['message']['content']


def add_smell(example):
    code_uid = example['code_uid']
    lang_cluster = example['lang_cluster']
    smell_code = example['smell_code']
    source_code = example['source_code']
    prompt = f"""As an expert software developer with years of experience, please meticulously inspect the following smell code segment and categorize it into one of the following categories:
- large class
- long method
- data class
- blob
- feature envy
The detailed information are as follows:
1. Programming language: {lang_cluster} 
2. Smell code segment: 
```
{smell_code.strip()}
```
3. Source code containing code smells:
```
{source_code.strip()}
```
Respond only with one of the specified categories."""

    try:
        response = generate_text(
            model=args.model,
            prompt=prompt,
            temperature=temperature
        )
        print('response:', response)

        if response is not None:
            supported_smells = ['large class', 'long method', 'data class', 'blob', 'feature envy']
            if all(supported_smell not in response.lower() for supported_smell in supported_smells):
                print('Respond content is invalid value.')
                smell = ''
            else:
                smell = ''
                for supported_smell in supported_smells:
                    if supported_smell in response.lower():
                        smell = supported_smell
                        break
        else:
            print('Respond content is none.')
            smell = ''

    except Exception as e:
        print('Failed to generate text:', e)
        smell = ''

    print('smell:', smell)
    example['smell'] = smell

    return example


def add_review_comment(example):
    return example


def add_hidden_unit_tests(example):
    return example


def main():
    load_path = Path(__file__).parent.parent / Path('data') / Path(args.data_load_name)
    save_path = Path(__file__).parent.parent / Path('results') / Path(args.result_save_name)

    dataset = load_dataset('json', split='train', data_files=str(load_path))
    print(dataset)

    if args.data_load_name == 'code_review_data.jsonl':
        dataset = dataset.map(add_review_comment)
    elif args.data_load_name == 'code_smell_data.jsonl':
        dataset = dataset.map(add_smell)
    elif args.data_load_name == 'code_test_data.jsonl':
        dataset = dataset.map(add_hidden_unit_tests)
    print(dataset)

    dataset.to_json(save_path, lines=True)


if __name__ == '__main__':
    args = parse_arguments()
    temperature = 0
    # References: https://github.com/chatanywhere/GPT_API_free
    openai.api_key = ''
    openai.api_base = 'https://api.chatanywhere.com.cn/v1'
    print(generate_text(args.model, 'Why did Zhou Shuren beat up Lu Xun?', temperature))
    # main()
    # python scripts/eval_gpt.py
