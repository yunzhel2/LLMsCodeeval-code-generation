import re
import json
import logging
import argparse
import google.generativeai as palm

from pathlib import Path
from datasets import load_dataset
from google.api_core import retry


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', default=None, type=str)
    parser.add_argument('--data_load_name', default='code_smell_data.jsonl',)
    parser.add_argument('--result_save_name', default='code_smell_eval_llama.jsonl')
    parser.add_argument('--log_file_name', default='code_smell_eval_llama.log'),
    args = parser.parse_args()

    return args

lang_cluster = ['C++', 'Java', 'Python', 'C', 'C#', 'Ruby', 'delphi', 'Go',
                'Javascript', 'Kotlin', 'PHP', 'd', 'perl', 'Rust']

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
    prompt = f"""As an expert software developer with years of experience, please meticulously inspect the following smell code snippet and categorize it into one of the following categories:
- large class
- data class
- blob
- feature envy
- long method
The detailed information are as follows:
1. Programming language: {lang_cluster} 
2. Smell code snippet: 
```
{smell_code.strip()}
```
3. Source code containing code smells:
```
{source_code.strip()}
```
Respond only with one of the specified categories."""

    logging.info('code uid: ' + str(code_uid))

    input_tokens = count_message_tokens(prompt=prompt)['token_count']
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning('Over input tokens limit: ' + str(code_uid))

    try:
        response = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
        logging.info('response: ' + str(response.result))

        if response.result is not None:
            output_tokens = count_message_tokens(prompt=response.result)['token_count']
            logging.info('output tokens: ' + str(output_tokens))
            if output_tokens > max_output_tokens:
                logging.warning('Over output tokens limit ' + str(code_uid))

            supported_smells = ['large class', 'long method', 'data class', 'blob', 'feature envy']
            if all(supported_smell not in response.result.lower() for supported_smell in supported_smells):
                logging.warning('Respond content is invalid value.')
                smell = ''
            else:
                smell = ''
                # Find the smell that first occurs in the response.
                min_index = float('inf')
                for supported_smell in supported_smells:
                    first_index = response.result.lower().find(supported_smell)
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

    input_tokens = count_message_tokens(prompt=prompt)['token_count']
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning('Over input tokens limit: ' + str(code_uid))

    try:
        response = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
        logging.info('response: ' + str(response.result))

        if response.result is not None:
            output_tokens = count_message_tokens(prompt=response.result)['token_count']
            logging.info('output tokens: ' + str(output_tokens))
            if output_tokens > max_output_tokens:
                logging.warning('Over output tokens limit ' + str(code_uid))

            supported_diff_tags = ['0', '1']
            if all(supported_diff_tag not in response.result for supported_diff_tag in supported_diff_tags):
                logging.warning('Respond content is invalid value.')
                diff_tag = 2
            else:
                diff_tag = 2
                # Find the diff tag that first occurs in the response.
                min_index = float('inf')
                for supported_diff_tag in supported_diff_tags:
                    first_index = response.result.find(supported_diff_tag)
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

    input_tokens = count_message_tokens(prompt=prompt)['token_count']
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning('Over input tokens limit: ' + str(code_uid))

    try:
        response = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
        logging.info('response: ' + str(response.result))

        if response.result is not None:
            output_tokens = count_message_tokens(prompt=response.result)['token_count']
            logging.info('output tokens: ' + str(output_tokens))
            if output_tokens > max_output_tokens:
                logging.warning('Over output tokens limit ' + str(code_uid))

            review_comment = response.result
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

    input_tokens = count_message_tokens(prompt=prompt)['token_count']
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning('Over input tokens limit: ' + str(code_uid))

    try:
        response = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
        logging.info('response: ' + str(response.result))

        if response.result is not None:
            output_tokens = count_message_tokens(prompt=response.result)['token_count']
            logging.info('output tokens: ' + str(output_tokens))
            if output_tokens > max_output_tokens:
                logging.warning('Over output tokens limit ' + str(code_uid))

            pattern = r'\[\s*\{.*?\}\s*\]'
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

env_map = {
    'C++': ['GNU C++11', 'GNU C++14', 'MS C++', 'GNU C++0x', 'GNU C++', 'MS C++ 2017', 'Clang++17 Diagnostics',
            'GNU C++17'],
    'C#': ['MS C#', 'Mono C#', '.NET Core C#'],
    'Java': ['Java 11', 'Java 7', 'Java 6', 'Java 8', 'Java 17', ],
    'Javascript': ['JavaScript', 'Node.js'],
    'C': ['GNU C', 'GNU C11'],
    'Python': ['Python 2', 'PyPy 3', 'Python 3', 'PyPy 2'],
    'PHP': ['PHP', 'PHP 8.1'],
    'Ruby': ['Ruby', 'Ruby 3'],
    'Kotlin': ['Kotlin', 'Kotlin 1.4', 'Kotlin 1.5', 'Kotlin 1.6', 'Kotlin 1.7', ],
    'Rust': ['Rust', 'Rust 2015', 'Rust 2021'],
    'Go': ['Go 1.19.5'],
    'd': ['dmd 2.105.0 win32'],
    'delphi': ['Delphi7 win32'],
    'perl': ['Perl v5.20.3']
}


def add_program_synthesis(example):
    """
    Generate corresponding code based on the problem description

    problem_attributes = ['title', 'description', 'input_from', 'output_to', 'time_limit',
           'memory_limit', 'input_spec', 'output_spec', 'notes', 'sample_inputs',
           'sample_outputs', 'id', 'difficulty', 'tags', 'src_uid']
    """

    # supported languages:
    lang_cluster = ['C++', 'Java', 'Python', 'C', 'C#', 'Ruby', 'delphi', 'Go',
                    'Javascript', 'Kotlin', 'PHP', 'd', 'perl', 'Rust']

    prob_uid = example['src_uid']
    prob_desc_description = example['description']
    prob_desc_input_spec = example['input_spec']
    prob_desc_output_spec = example['output_spec']
    prob_desc_sample_inputs = example['sample_inputs']
    prob_desc_sample_outputs = example['sample_outputs']
    prob_desc_notes = example['notes']

    for lang in lang_cluster:
        prompt = f"""As an experienced code developer with years of expertise, please provide the source code based on the problem description. Here are the specifics:
1. Problem description: {prob_desc_description}
2. Input specification: {prob_desc_input_spec}
3. Output specification: {prob_desc_output_spec}
4. Sample inputs: {prob_desc_sample_inputs}
5. Sample outputs: {prob_desc_sample_outputs}
6. Sample explanations: {prob_desc_notes}
7. Programming language: {lang} 
8. support programming language version: {env_map[lang]}
Respond should only with a string in the following JSON format:
[{{"version": specific version used in the programming language, "target code":  the code you produced in the respective programming language version."}}] """

        logging.info('problem src_id: ' + str(prob_uid))

        input_tokens = count_message_tokens(prompt=prompt)['token_count']
        logging.info('input tokens: ' + str(input_tokens))
        if input_tokens > max_input_tokens:
            logging.warning('Over input tokens limit: ' + str(prob_uid))

        try:
            response = generate_text(
                prompt=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens
            )
            logging.info('response: ' + str(response.result))

            if response.result is not None:
                output_tokens = count_message_tokens(prompt=response.result)['token_count']
                logging.info('output tokens: ' + str(output_tokens))
                if output_tokens > max_output_tokens:
                    logging.warning('Over total tokens limit ' + str(prob_uid) + ' lang: ' + str(lang))
                    program_sythesis = ''
                else:
                    program_sythesis = response.result
            else:
                logging.warning('Respond content is none.')
                program_sythesis = ''

        except Exception as e:
            logging.error('Failed to generate text: ' + e.__str__())
            program_sythesis = ''

        logging.info('program_synthesis in: ' + lang + ' :' + str(program_sythesis))
        example[lang] = program_sythesis

    return example


def add_code_translation(example):
    """
     Generate corresponding code in specific language based on the given code

     problem_attributes = ['title', 'description', 'input_from', 'output_to', 'time_limit',
            'memory_limit', 'input_spec', 'output_spec', 'notes', 'sample_inputs',
            'sample_outputs', 'id', 'difficulty', 'tags', 'src_uid']

    submission_attributes = ['lang', 'source_code', 'tags', 'lang_cluster', 'src_uid', 'code_uid',
       'difficulty', 'exec_outcome', 'verdict', 'time', 'memory', 'sent',
       'judged', 'id', 'submission_id', 'participant_id']
     """

    # supported languages:
    lang_cluster = ['C++', 'Java', 'Python', 'C', 'C#', 'Ruby', 'delphi', 'Go',
                    'Javascript', 'Kotlin', 'PHP', 'd', 'perl', 'Rust']
    source_lang = example['lang_cluster']
    target_lang = example['target_lang_cluster']
    prob_uid = example['src_uid']
    source_code = example['source_code']

    prompt = f"""As an expert code developer proficient in multiple programming languages with years of experience, please translate the source code in {source_lang} to programming language {target_lang} within our supported version. 
The detailed information are as follows:
1. Target programming language: {target_lang}
2. support programming language version: {env_map[target_lang]}
3. Source code\n: {source_code}

Respond should only with a string in the following JSON format:
[{{"version": specific version used in the programming language, "target code":  the code you produced in the respective programming language version."}}] """

    logging.info('problem src_id: ' + str(prob_uid))

    input_tokens = count_message_tokens(prompt=prompt)['token_count']
    logging.info('input tokens: ' + str(input_tokens))

    try:
        response = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
        logging.info('response: ' + str(response))

        if response is not None:
            output_tokens = count_message_tokens(prompt=response.result)['token_count']
            logging.info('output tokens: ' + str(output_tokens))
            if input_tokens > max_input_tokens:
                logging.warning('Over total tokens limit ' + str(prob_uid) + ' lang: ' + str(lang_cluster))
                translation_outcome = ''
            else:
                translation_outcome = response.result
        else:
            logging.warning('Respond content is none.')
            translation_outcome = ''

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        translation_outcome = ''

    logging.info('Code translation in: ' + target_lang + ' :' + str(translation_outcome))
    example['translation_result'] = translation_outcome

    return example


def add_code_repairing(example):
    """
     Generate corresponding code based on the problem description

     problem_attributes = ['title', 'description', 'input_from', 'output_to', 'time_limit',
            'memory_limit', 'input_spec', 'output_spec', 'notes', 'sample_inputs',
            'sample_outputs', 'id', 'difficulty', 'tags', 'src_uid']

    submission_attributes = ['lang', 'source_code', 'tags', 'lang_cluster', 'src_uid', 'code_uid',
       'difficulty', 'exec_outcome', 'verdict', 'time', 'memory', 'sent',
       'judged', 'id', 'submission_id', 'participant_id']
     """

    # supported languages:
    lang_cluster = ['C++', 'Java', 'Python', 'C', 'C#', 'Ruby', 'delphi', 'Go',
                    'Javascript', 'Kotlin', 'PHP', 'd', 'perl', 'Rust']
    source_lang = example['lang']

    prob_uid = example['src_uid']
    source_code = example['source_code']
    prob_desc_description = example['description']
    prob_desc_input_spec = example['input_spec']
    prob_desc_output_spec = example['output_spec']
    prob_desc_sample_inputs = example['sample_inputs']
    prob_desc_sample_outputs = example['sample_outputs']
    error_msg = example['exec_outcome']
    prompt = f"""As an expert code developer with years of experience, please debug the source code in {source_lang} based on the corresponding problem description and show the correct code. 
The detailed information are shown as follows:  
1. Problem description: {prob_desc_description}
2. Input specification: {prob_desc_input_spec}
3. Output specification: {prob_desc_output_spec}
4. Sample inputs: {prob_desc_sample_inputs} 
5. Sample outputs: {prob_desc_sample_outputs}   
6. Programming language: {source_lang}
7. Buggy code :\n {source_code}
8. Error message: {error_msg}
Please note that use complex header files as little as possible. 

Respond should only with a string in the following JSON format:
[{{"version": specific version used in the programming language, "target code":  the code you produced in the respective programming language version."}}] """
    logging.info('problem src_id: ' + str(prob_uid))

    input_tokens = count_message_tokens(prompt=prompt)['token_count']
    logging.info('input tokens: ' + str(input_tokens))

    try:
        response = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_output_tokens=max_output_tokens
        )
        logging.info('response: ' + str(response))

        if response is not None:
            output_tokens = count_message_tokens(prompt=response.result)['token_count']
            logging.info('output tokens: ' + str(output_tokens))
            if input_tokens > max_input_tokens:
                logging.warning('Over total tokens limit ' + str(prob_uid) + ' lang: ' + str(lang_cluster))
                repair_outcome = ''
            else:
                repair_outcome = response.result
        else:
            logging.warning('Respond content is none.')
            repair_outcome = ''

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        repair_outcome = ''

    logging.info('Code repairing in: ' +  str(repair_outcome))
    example['debug_result'] = repair_outcome

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
    elif 'program_synthesis' in args.data_load_name:
        dataset = dataset.map(add_program_synthesis)
    elif 'translation' in args.data_load_name:
        dataset = dataset.map(add_code_translation)
    elif 'code_debugging' in args.data_load_name:
        dataset = dataset.map(add_code_repairing)
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

    # References: https://github.com/google/generative-ai-python/issues/29
    palm.configure(api_key=args.api_key, transport='rest')
    models = [model for model in palm.list_models() if 'generateText' in model.supported_generation_methods]
    temperature = 0
    max_input_tokens = models[0].input_token_limit  # 8192
    max_output_tokens = models[0].output_token_limit  # 1024

    main()
    # python scripts/eval_palm.py
