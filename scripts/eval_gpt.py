import re
import json
import openai
import backoff
import logging
import tiktoken
import argparse

from pathlib import Path
from datasets import load_dataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', default='', type=str)
    parser.add_argument('--model', default='gpt-3.5-turbo-0613',
                        choices=['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k-0613',
                                 'gpt-3.5-turbo-0301', 'gpt-4', 'gpt-4-0613', 'gpt-4-32k', 'gpt-4-32k-0613',
                                 'gpt-4-0314', 'gpt-4-32k-0314'],
                        type=str)
    parser.add_argument('--data_load_name', default='code_test_data.jsonl',
                        choices=['code_review_data.jsonl', 'code_smell_data.jsonl', 'code_test_data.jsonl'], type=str)
    parser.add_argument('--result_save_name', default='code_test_data_gpt3.jsonl',
                        choices=['code_review_eval_gpt3.jsonl', 'code_smell_eval_gpt3.jsonl',
                                 'code_test_data_gpt3.jsonl', 'code_review_eval_gpt4.jsonl',
                                 'code_smell_eval_gpt4.jsonl', 'code_test_data_gpt4.jsonl'], type=str)
    parser.add_argument('--log_file_name', default='code_test_data_gpt3.log',
                        choices=['code_review_eval_gpt3.log', 'code_smell_eval_gpt3.log', 'code_test_data_gpt3.log',
                                 'code_review_eval_gpt4.log', 'code_smell_eval_gpt4.log', 'code_test_data_gpt4.log'],
                        type=str)
    args = parser.parse_args()

    return args


# References: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def generate_text(model, prompt, temperature):
    messages = [{'role': 'user', 'content': prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature
    )

    return response['choices'][0]['message']['content']


# References: https://github.com/openai/openai-cookbook/blob/5783656852d507c335955d14875ebc9902f628ef/examples/How_to_count_tokens_with_tiktoken.ipynb
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def count_message_tokens(content, model, type):
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print('Model not found, using cl100k_base encoding.')
        encoding = tiktoken.get_encoding('cl100k_base')

    num_tokens = 0
    if type == 'input':
        messages = [{'role': 'user', 'content': content}]
        tokens_per_message = 4
        tokens_per_name = -1
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == 'name':
                    num_tokens += tokens_per_name
        num_tokens += 3
    elif type == 'output':
        num_tokens = len(encoding.encode(content))

    return num_tokens


def add_smell(example):
    code_uid = example['code_uid']
    lang_cluster = example['lang_cluster']
    smell_code = example['smell_code']
    source_code = example['source_code']
    if lang_cluster == 'C#':
        prompt = f"""As an expert software developer with years of experience, please meticulously inspect the following smell code segment and categorize it into one of the following categories:
- large class
- long method
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
    else:
        prompt = f"""As an expert software developer with years of experience, please meticulously inspect the following smell code segment and categorize it into one of the following categories:
- data class
- blob
- feature envy
- long method
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

    logging.info('code uid: ' + str(code_uid))

    input_tokens = count_message_tokens(prompt, args.model, 'input')
    logging.info('input tokens: ' + str(input_tokens))

    try:
        response = generate_text(
            model=args.model,
            prompt=prompt,
            temperature=temperature
        )
        logging.info('response: ' + str(response))

        if response is not None:
            output_tokens = count_message_tokens(response, args.model, 'output')
            logging.info('output tokens: ' + str(output_tokens))
            if input_tokens + output_tokens > max_tokens:
                logging.warning('Over total tokens limit ' + str(code_uid))

            supported_smells = ['large class', 'long method', 'data class', 'blob', 'feature envy']
            if all(supported_smell not in response.lower() for supported_smell in supported_smells):
                logging.warning('Respond content is invalid value.')
                smell = ''
            else:
                smell = ''
                # Find the smell that first occurs in the response.
                min_index = float('inf')
                for supported_smell in supported_smells:
                    first_index = response.lower().find(supported_smell)
                    if first_index != -1 and first_index < min_index:
                        min_index = first_index
                        smell = supported_smell
        else:
            logging.warning('Respond content is none.')
            smell = ''

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        smell = ''

    logging.info('smell: ' + str(smell))
    example['smell'] = smell

    return example


def add_diff_tag(example):
    code_uid = example['code_uid']
    lang_cluster = example['lang_cluster']
    old_code = example['old_code']
    diff_hunk = example['diff_hunk']

    prompt = f"""As an expert code reviewer with years of experience, please meticulously inspect the following code change and categorize its quality into one of the following categories:
- 0: Good quality that no review comments required.
- 1: Poor quality that requires review comments.
The detailed information are as follows:
1. Programming language: {lang_cluster} 
2. Original version code: 
```
{old_code.strip()}
```
3. Code diff chunk:
```
{diff_hunk.strip()}
```
Respond only with the number: 0 or 1."""

    logging.info('code uid: ' + str(code_uid))

    input_tokens = count_message_tokens(prompt, args.model, 'input')
    logging.info('input tokens: ' + str(input_tokens))

    try:
        response = generate_text(
            model=args.model,
            prompt=prompt,
            temperature=temperature
        )
        logging.info('response: ' + str(response))

        if response is not None:
            output_tokens = count_message_tokens(response, args.model, 'output')
            logging.info('output tokens: ' + str(output_tokens))
            if input_tokens + output_tokens > max_tokens:
                logging.warning('Over total tokens limit ' + str(code_uid))

            supported_diff_tags = ['0', '1']
            if all(supported_diff_tag not in response for supported_diff_tag in supported_diff_tags):
                logging.warning('Respond content is invalid value.')
                diff_tag = 2
            else:
                diff_tag = 2
                # Find the diff tag that first occurs in the response.
                min_index = float('inf')
                for supported_diff_tag in supported_diff_tags:
                    first_index = response.find(supported_diff_tag)
                    if first_index != -1 and first_index < min_index:
                        min_index = first_index
                        diff_tag = int(supported_diff_tag)
        else:
            logging.warning('Respond content is none.')
            diff_tag = 2

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        diff_tag = 2

    logging.info('diff_tag: ' + str(diff_tag))
    example['diff_tag'] = diff_tag

    return example


def add_review_comment(example):
    code_uid = example['code_uid']
    lang_cluster = example['lang_cluster']
    old_code = example['old_code']
    diff_hunk = example['diff_hunk']
    prompt = f"""As an expert code reviewer with years of experience, please meticulously inspect the following code change and provide a concise review comment.
The detailed information are as follows:
1. Programming language: {lang_cluster} 
2. Original version code: 
```
{old_code.strip()}
```
3. Code diff chunk:
```
{diff_hunk.strip()}
```
Respond only with a string that represents review comment."""

    logging.info('code uid: ' + str(code_uid))

    input_tokens = count_message_tokens(prompt, args.model, 'input')
    logging.info('input tokens: ' + str(input_tokens))

    try:
        response = generate_text(
            model=args.model,
            prompt=prompt,
            temperature=temperature
        )
        logging.info('response: ' + str(response))

        if response is not None:
            output_tokens = count_message_tokens(response, args.model, 'output')
            logging.info('output tokens: ' + str(output_tokens))
            if input_tokens + output_tokens > max_tokens:
                logging.warning('Over total tokens limit ' + str(code_uid))

            review_comment = response
        else:
            logging.warning('Respond content is none.')
            review_comment = ''

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        review_comment = ''

    logging.info('review_comment: ' + str(review_comment))
    example['review_comment'] = review_comment

    return example


def add_hidden_unit_tests(example):
    code_uid = example['code_uid']
    prob_desc_description = example['prob_desc_description']
    prob_desc_time_limit = example['prob_desc_time_limit']
    prob_desc_memory_limit = example['prob_desc_memory_limit']
    prob_desc_input_spec = example['prob_desc_input_spec']
    prob_desc_output_spec = example['prob_desc_output_spec']
    prob_desc_sample_inputs = example['prob_desc_sample_inputs']
    prob_desc_sample_outputs = example['prob_desc_sample_outputs']
    prob_desc_notes = example['prob_desc_notes']
    source_code = example['source_code']
    lang_cluster = example['lang_cluster']
    num_hidden_unit_tests = example['num_hidden_unit_tests']
    prompt = f"""As an expert code test developer with years of experience, please provide multiple test cases for a given problem along and its solution.
The detailed information are as follows:
1. Problem description: {prob_desc_description}
2. Time limit: {prob_desc_time_limit}
3. Memory limit: {prob_desc_memory_limit}
4. Input specification: {prob_desc_input_spec}
5. Output specification: {prob_desc_output_spec}
6. Sample inputs: {prob_desc_sample_inputs}
7. Sample outputs: {prob_desc_sample_outputs}
8. Sample explanations: {prob_desc_notes}
9. Programming language: {lang_cluster} 
10. Solution source code: 
```
{source_code.strip()}
```
Craft {num_hidden_unit_tests} test cases with these criteria:
1. Each test case contains a string for both input and output.
2. The solution source code successfully processes the test case's input with no errors.
3. The solution source code's outcome aligns with the test case's output.
4. All test cases are simple and achieve optimal branch and line coverage.
Respond only with a string in the following JSON format:
[{{"input": input string, "output": output string}}]"""

    logging.info('code uid: ' + str(code_uid))

    input_tokens = count_message_tokens(prompt, args.model, 'input')
    logging.info('input tokens: ' + str(input_tokens))

    try:
        response = generate_text(
            model=args.model,
            prompt=prompt,
            temperature=temperature
        )
        logging.info('response: ' + str(response))

        if response is not None:
            output_tokens = count_message_tokens(response, args.model, 'output')
            logging.info('output tokens: ' + str(output_tokens))
            if input_tokens + output_tokens > max_tokens:
                logging.warning('Over total tokens limit ' + str(code_uid))

            pattern = r'\[\s*\{.*?\}\s*\]'
            matches = re.search(pattern, response, re.DOTALL)
            if matches:
                json_array_string = matches.group().replace("'", '"')
                try:
                    json_array = json.loads(json_array_string, strict=False)
                    if isinstance(json_array, list):
                        for json_item in json_array:
                            if isinstance(json_item['input'], list):
                                json_item['input'] = str(json_item['input'][0])
                            if isinstance(json_item['output'], str):
                                json_item['output'] = [json_item['output']]
                        hidden_unit_tests = str(json_array)
                    else:
                        logging.warning('Respond content is not a list.')
                        hidden_unit_tests = "[{'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}]"
                except json.JSONDecodeError as e:
                    logging.warning('Failed to load json:', e)
                    hidden_unit_tests = "[{'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}]"
            else:
                logging.warning('JSON array object not found.')
                hidden_unit_tests = "[{'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}]"

        else:
            logging.warning('Respond content is none.')
            hidden_unit_tests = "[{'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}]"

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        hidden_unit_tests = "[{'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}, {'input': '', 'output': ['']}]"

    logging.info('hidden_unit_tests: ' + str(hidden_unit_tests))
    example['hidden_unit_tests'] = hidden_unit_tests

    return example


def main():
    load_path = Path(__file__).parent.parent / Path('data') / Path(args.data_load_name)
    save_path = Path(__file__).parent.parent / Path('results') / Path(args.result_save_name)

    dataset = load_dataset('json', split='train', data_files=str(load_path))
    dataset.cleanup_cache_files()  # for multiple evaluation
    print(dataset)

    if args.data_load_name == 'code_review_data.jsonl':
        dataset = dataset.map(add_diff_tag)
        dataset = dataset.map(add_review_comment)
    elif args.data_load_name == 'code_smell_data.jsonl':
        dataset = dataset.map(add_smell)
    elif args.data_load_name == 'code_test_data.jsonl':
        dataset = dataset.map(add_hidden_unit_tests)
    print(dataset)

    dataset.to_json(save_path, lines=True)


if __name__ == '__main__':
    args = parse_arguments()

    log_file_path = Path(__file__).parent.parent / Path('logs') / Path(args.log_file_name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(filename)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(filename=log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    # References: https://platform.openai.com/docs/api-reference/authentication
    openai.api_key = args.api_key
    model_max_tokens = {
        'gpt-3.5-turbo': 4097,
        'gpt-3.5-turbo-16k': 16385,
        'gpt-3.5-turbo-0613': 4097,
        'gpt-3.5-turbo-16k-0613': 16385,
        'gpt-3.5-turbo-0301': 4097,
        'gpt-4': 8192,
        'gpt-4-0613': 8192,
        'gpt-4-32k': 32768,
        'gpt-4-32k-0613': 32768,
        'gpt-4-0314': 8192,
        'gpt-4-32k-0314': 32768
    }
    temperature = 0
    max_tokens = model_max_tokens.get(args.model) if model_max_tokens.get(args.model) is not None else 0

    main()
    # python scripts/eval_gpt.py
