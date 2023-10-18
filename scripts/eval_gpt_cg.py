import re
import json
import openai
import backoff
import logging
import os
import pandas as pd
import tiktoken
import argparse

from pathlib import Path
from datasets import load_dataset, Dataset


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', default='', type=str)
    parser.add_argument('--model', default='gpt-3.5-turbo',
                        choices=['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k-0613',
                                 'gpt-3.5-turbo-0301', 'gpt-4', 'gpt-4-0613', 'gpt-4-32k', 'gpt-4-32k-0613',
                                 'gpt-4-0314', 'gpt-4-32k-0314'],
                        type=str)
    parser.add_argument('--data_load_name', default='program_synthesis_data.jsonl', type=str)
    parser.add_argument('--result_save_name', default='program_synthesis_eval_gpt3.jsonl', type=str)
    parser.add_argument('--log_file_name', default='program_synthesis_eval_gpt3.log', type=str)
    parser.add_argument('--temperature', default=0.5, type=float)
    parser.add_argument('--candidate_num', default=1, type=int)
    args = parser.parse_args()

    return args


# References: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def generate_text(model, prompt, temperature,candidate):
    messages = [{'role': 'user', 'content': prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        n=candidate
    )
    results = []
    for i in range(candidate):
        results.append(response['choices'][i]['message']['content'])
    return results


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




env_map = {
    'c++': ['GNU C++11', 'GNU C++14', 'MS C++', 'GNU C++0x', 'GNU C++', 'MS C++ 2017', 'Clang++17 Diagnostics',
            'GNU C++17'],
    'c#': ['MS C#', 'Mono C#', '.NET Core C#'],
    'java': ['Java 11', 'Java 7', 'Java 6', 'Java 8'],
    'javascript': ['JavaScript', 'Node.js'],
    'c': ['GNU C', 'GNU C11'],
    'python': ['Python 2', 'PyPy 3', 'Python 3', 'PyPy 2'],
    'php': ['PHP'],
    'ruby': ['Ruby'],
    'kotlin': ['Kotlin'],
    'rust': ['Rust'],
    'go': ['Go'],
    'd': ['dmd 2.105.0 win32'],
    'delphi': ['Delphi7 win32'],
    'perl': ['Perl v5.20.3']
}


# supported languages:
lang_cluster = ['c++', 'java', 'python', 'c', 'c#', 'ruby', 'delphi', 'go',
                'javascript', 'kotlin', 'php', 'd', 'perl', 'rust']

def add_data_augment_v2(example):
    """
    Generate corresponding code based on the problem description and corresponding language for new data generation

    problem_attributes = ['title', 'description', 'input_from', 'output_to', 'time_limit',
           'memory_limit', 'input_spec', 'output_spec', 'notes', 'sample_inputs',
           'sample_outputs', 'id', 'difficulty', 'tags', 'src_uid']
    """

    repeat_time = 5
    lang = example['target_lang']
    prob_uid = example['src_uid']
    prob_desc_description = example['description']
    prob_desc_input_spec = example['input_spec']
    prob_desc_output_spec = example['output_spec']
    prob_desc_sample_inputs = example['sample_inputs']
    prob_desc_sample_outputs = example['sample_outputs']
    prob_desc_notes = example['notes']

    prompt = f"""
As a professional code developer with years of experience, please provide the corresponding code solution based on the problem description. Detailed information is given below:
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

    input_tokens = count_message_tokens(prompt, args.model, 'input')
    logging.info('input tokens: ' + str(input_tokens))
    for time in range(repeat_time):
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
                    logging.warning('Over total tokens limit ' + str(prob_uid) + ' lang: ' + str(lang))
                    program_synthesis = ''
                else:
                    program_synthesis = response
            else:
                logging.warning('Respond content is none.')
                program_synthesis = ''
        except Exception as e:
            logging.error('Failed to generate text: ' + e.__str__())
            program_synthesis = ''

        logging.info('program_synthesis in: ' + lang + ' :' + str(program_synthesis))
        example[str(time)] = program_synthesis

    return example


def add_program_synthesis(example):
    """
    Generate corresponding code based on the problem description

    problem_attributes = ['title', 'description', 'input_from', 'output_to', 'time_limit',
           'memory_limit', 'input_spec', 'output_spec', 'notes', 'sample_inputs',
           'sample_outputs', 'id', 'difficulty', 'tags', 'src_uid']
    """

    prob_uid = example['src_uid']
    prob_desc_description = example['description']
    prob_desc_input_spec = example['input_spec']
    prob_desc_output_spec = example['output_spec']
    prob_desc_sample_inputs = example['sample_inputs']
    prob_desc_sample_outputs = example['sample_outputs']
    prob_desc_notes = example['notes']
    lang = example['lang_cluster']

    prompt = f"""
As a professional code developer with years of experience, please provide the corresponding code solution based on the problem description. Detailed information is given below:
1. Problem description: {prob_desc_description}
2. Input specification: {prob_desc_input_spec}
3. Output specification: {prob_desc_output_spec}
4. Sample inputs: {prob_desc_sample_inputs}
5. Sample outputs: {prob_desc_sample_outputs}
6. Sample explanations: {prob_desc_notes}
7. Programming language: {lang} 
8. support programming language version: {env_map[lang]}
Please take care to minimize the use of complex header files.

Respond should only with a string in the following JSON format:
[{{"version": specific version used in the programming language, "target code":  the code you produced in the respective programming language version."}}] """

    logging.info('problem src_id: ' + str(prob_uid))

    input_tokens = count_message_tokens(prompt, args.model, 'input')
    logging.info('input tokens: ' + str(input_tokens))

    try:
        response = generate_text(
            model=args.model,
            prompt=prompt,
            temperature=temperature,
            candidate=candidate_num
        )
        logging.info('response: ' + str(response))

        if response is not None:
            output_tokens = count_message_tokens(response, args.model, 'output')
            logging.info('output tokens: ' + str(output_tokens))
            if input_tokens + output_tokens > max_tokens:
                logging.warning('Over total tokens limit ' + str(prob_uid) + ' lang: ' + str(lang))
                program_sythesis = ''
            else:
                program_sythesis = response
        else:
            logging.warning('Respond content is none.')
            program_sythesis = ''

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        program_sythesis = ''

    logging.info('program_synthesis in: ' + lang + ' :' + str(program_sythesis))
    example['program_synthesis'] = program_sythesis

    for i, generated_code in enumerate(program_sythesis):
        logging.info('program_synthesis  in: ' + lang + ' :' + generated_code)
        example['program_synthesis_' + str(i)] = generated_code
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

    source_lang = example['lang_cluster']
    target_lang = example['target_lang_cluster']
    prob_uid = example['src_uid']
    source_code = example['source_code']


    prompt = f"""As an expert code developer proficient in multiple programming languages with years of experience, please translate the source code in {source_lang} to programming language {target_lang} within our supported version. 
The detailed information are as follows:
1. Target programming language: {target_lang}
2. support programming language version: {env_map[target_lang]}
3. Source code\n: {source_code}
Please take care to minimize the use of complex header files.

Respond should only with a string in the following JSON format:
[{{"version": specific version used in the programming language, "target code":  the code you produced in the respective programming language version."}}] """

    logging.info('problem src_id: ' + str(prob_uid))

    input_tokens = count_message_tokens(prompt, args.model, 'input')
    logging.info('input tokens: ' + str(input_tokens))

    try:
        response = generate_text(
            model=args.model,
            prompt=prompt,
            temperature=temperature,
            candidate=candidate_num
        )
        logging.info('response: ' + str(response))

        if response is not None:
            output_tokens = count_message_tokens(response, args.model, 'output')
            logging.info('output tokens: ' + str(output_tokens))
            if input_tokens + output_tokens > max_tokens:
                logging.warning('Over total tokens limit ' + str(prob_uid) + ' lang: ' + str(lang_cluster))
                translation_outcome = ''
            else:
                translation_outcome = response
        else:
            logging.warning('Respond content is none.')
            translation_outcome = ''

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        translation_outcome = ''

    for i in range(len(translation_outcome)):
        logging.info('code translation  in: ' + target_lang + ' :' + str(translation_outcome[i]))
        example[target_lang + '_' + str(i)] = translation_outcome[i]

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
Please take care to minimize the use of complex header files.

Respond should only with a string in the following JSON format:
[{{"version": specific version used in the programming language, "target code":  the code you produced in the respective programming language version."}}] """

    logging.info('problem src_id: ' + str(prob_uid))

    input_tokens = count_message_tokens(prompt, args.model, 'input')
    logging.info('input tokens: ' + str(input_tokens))

    try:
        response = generate_text(
            model=args.model,
            prompt=prompt,
            temperature=temperature,
            candidate=candidate_num
        )
        logging.info('response: ' + str(response))

        if response is not None:
            output_tokens = count_message_tokens(response, args.model, 'output')
            logging.info('output tokens: ' + str(output_tokens))
            if input_tokens + output_tokens > max_tokens:
                logging.warning('Over total tokens limit ' + str(prob_uid) + ' lang: ' + str(lang_cluster))
                repair_outcome = []
            else:
                repair_outcome = response
        else:
            logging.warning('Respond content is none.')
            repair_outcome = []

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        repair_outcome = []

    for i in range(len(repair_outcome)):
        logging.info('Code repairing in: ' + source_lang + ' :' + str(repair_outcome[i]))
        example['Code repairing_'+str(i)] = repair_outcome[i]

    return example


def main():
    load_path = Path(__file__).parent.parent / Path('data') / Path(args.data_load_name)
    save_path = Path(__file__).parent.parent / Path('results') / Path(args.result_save_name)
    if 'augment_problem_list' in args.data_load_name:
        data = []
        for cluster in lang_cluster:
            file_name = f'''1007_before_{cluster.lower()}.jsonl'''
            if os.path.exists(os.path.join(load_path,file_name)):
                data.append(pd.read_json(os.path.join(load_path,file_name), lines=True))
        data = pd.concat(data,axis=0)
        dataset = Dataset.from_pandas(data)
    elif 'program_synthesis_v3' in args.data_load_name:
            data = []
            for cluster in lang_cluster:
                file_name = f'''1014_each30_{cluster}.jsonl'''
                if os.path.exists(os.path.join(load_path, file_name)):
                    data.append(pd.read_json(os.path.join(load_path, file_name), lines=True))
            data = pd.concat(data, axis=0)
            dataset = Dataset.from_pandas(data)
    else:
        dataset = load_dataset('json', split='train', data_files=str(load_path))
        dataset.cleanup_cache_files()  # for multiple evaluation

    # For debug
    dataset = dataset.select([0])

    if  'augment_problem_list'  in args.data_load_name:
        dataset = dataset.map(add_data_augment_v2)
    elif 'program_synthesis' in args.data_load_name:
        dataset = dataset.map(add_program_synthesis)
    elif 'translation' in args.data_load_name:
        dataset = dataset.map(add_code_translation)
    elif 'code_debug' in args.data_load_name:
        dataset = dataset.map(add_code_repairing)
    else:
        print("please use corresponding task as file name")
    dataset.to_json(save_path, orient='records',lines=True)


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

    candidate_num = args.candidate_num
    temperature = args.temperature
    max_tokens = model_max_tokens.get(args.model) if model_max_tokens.get(args.model) is not None else 0

    main()
    # python scripts/eval_gpt.py