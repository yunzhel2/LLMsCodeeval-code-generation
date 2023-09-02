import re
import json
import argparse
import google.generativeai as palm

from pathlib import Path
from datasets import load_dataset
from google.api_core import retry


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_load_name', default='code_review_data.jsonl',
                        choices=['code_review_data.jsonl', 'code_smell_data.jsonl', 'code_test_data.jsonl'], type=str)
    parser.add_argument('--result_save_name', default='code_review_eval_palm.jsonl',
                        choices=['code_review_eval_palm.jsonl', 'code_smell_eval_palm.jsonl',
                                 'code_test_data_palm.jsonl'], type=str)
    args = parser.parse_args()

    return args


@retry.Retry()
def generate_text(*args, **kwargs):
    return palm.generate_text(*args, **kwargs)


@retry.Retry()
def count_message_tokens(*args, **kwargs):
    return palm.count_message_tokens(*args, **kwargs)


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

    input_tokens = count_message_tokens(prompt=prompt)['token_count']
    print('input tokens:', input_tokens)
    if input_tokens > max_input_tokens:
        print('Over input tokens limit:', code_uid)

    try:
        response = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
        print('response:', response.result)

        if response.result is not None:
            output_tokens = count_message_tokens(prompt=response.result)['token_count']
            print('output tokens:', output_tokens)
            if output_tokens > max_output_tokens:
                print('Over output tokens limit:', code_uid)

            supported_smells = ['large class', 'long method', 'data class', 'blob', 'feature envy']
            if all(supported_smell not in response.result.lower() for supported_smell in supported_smells):
                print('Respond content is invalid value.')
                smell = ''
            else:
                smell = ''
                for supported_smell in supported_smells:
                    if supported_smell in response.result.lower():
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
    code_uid = example['code_uid']
    lang_cluster = example['lang_cluster']
    old_code = example['old_code']
    diff_hunk = example['diff_hunk']

    prompt1 = f"""As an expert code reviewer with years of experience, please meticulously inspect the following code change and categorize its quality:
0: High quality, no review comments needed.
1: Low quality, needs review comments.
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

    prompt2 = f"""As an expert code reviewer with years of experience, please meticulously inspect the following code change and provide a concise review comment.
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

    input_tokens = count_message_tokens(prompt=prompt1)['token_count']
    print('input tokens:', input_tokens)
    if input_tokens > max_input_tokens:
        print('Over input tokens limit:', code_uid)

    try:
        response1 = generate_text(
            prompt=prompt1,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
        print('response1:', response1.result)

        if response1.result is not None:
            output_tokens = count_message_tokens(prompt=response1.result)['token_count']
            print('output tokens:', output_tokens)
            if output_tokens > max_output_tokens:
                print('Over output tokens limit:', code_uid)

            supported_diff_tags = ['0', '1']
            if all(supported_diff_tag not in response1.result for supported_diff_tag in supported_diff_tags):
                print('First respond content is invalid value.')
                diff_tag = 2
                review_comment = ''
            else:
                diff_tag = 2
                for supported_diff_tag in supported_diff_tags:
                    if supported_diff_tag in response1.result:
                        diff_tag = int(supported_diff_tag)
                        break

                if diff_tag == 0:
                    review_comment = ''
                else:
                    input_tokens = count_message_tokens(prompt=prompt2)['token_count']
                    print('input tokens:', input_tokens)
                    if input_tokens > max_input_tokens:
                        print('Over input tokens limit:', code_uid)

                    try:
                        response2 = generate_text(
                            prompt=prompt2,
                            temperature=temperature,
                            max_output_tokens=max_output_tokens
                        )
                        print('response2:', response2.result)

                        if response2.result is not None:
                            output_tokens = count_message_tokens(prompt=response2.result)['token_count']
                            print('output tokens:', output_tokens)
                            if output_tokens > max_output_tokens:
                                print('Over output tokens limit:', code_uid)
                            review_comment = response2.result
                        else:
                            print('Second respond content is none.')
                            review_comment = ''

                    except Exception as e:
                        print('Failed to generate text:', e)
                        review_comment = ''
        else:
            print('First respond content is none.')
            diff_tag = 2
            review_comment = ''

    except Exception as e:
        print('Failed to generate text:', e)
        diff_tag = 2
        review_comment = ''

    print('diff_tag:', diff_tag)
    print('review_comment:', review_comment)
    example['diff_tag'] = diff_tag
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
9. Solution source code: 
```
{source_code.strip()}
```
10. Programming language: {lang_cluster} 
Craft {num_hidden_unit_tests} test cases with these criteria:
1. Each test case contains a string for both input and output.
2. The solution source code successfully processes the test case's input with no errors.
3. The solution source code's outcome aligns with the test case's output.
4. All test cases are simple and achieve optimal branch and line coverage.
Respond only with a string in the following JSON format:
[{{"input": input string, "output": output string}}]"""

    input_tokens = count_message_tokens(prompt=prompt)['token_count']
    print('input tokens:', input_tokens)
    if input_tokens > max_input_tokens:
        print('Over input tokens limit:', code_uid)

    try:
        response = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
        print('response:', response.result)

        if response.result is not None:
            output_tokens = count_message_tokens(prompt=response.result)['token_count']
            print('output tokens:', output_tokens)
            if output_tokens > max_output_tokens:
                print('Over output tokens limit:', code_uid)

            pattern = r'\[{.*?\}]'
            matches = re.search(pattern, response.result, re.DOTALL)
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
                        print('Respond content is not a list.')
                        hidden_unit_tests = "[]"
                except json.JSONDecodeError as e:
                    print('Failed to load json:', e)
                    hidden_unit_tests = "[]"
            else:
                print('JSON array not found.')
                hidden_unit_tests = "[]"

        else:
            print('Respond content is none.')
            hidden_unit_tests = "[]"

    except Exception as e:
        print('Failed to generate text:', e)
        hidden_unit_tests = "[]"

    print('hidden_unit_tests:', hidden_unit_tests)
    example['hidden_unit_tests'] = hidden_unit_tests

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
    # References: https://github.com/google/generative-ai-python/issues/29
    palm.configure(api_key='', transport='rest')
    models = [model for model in palm.list_models() if 'generateText' in model.supported_generation_methods]
    max_input_tokens = models[0].input_token_limit
    max_output_tokens = models[0].output_token_limit
    main()
    # python scripts/eval_palm.py
