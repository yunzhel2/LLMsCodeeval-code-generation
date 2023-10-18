import re
import json
import torch
import logging
import argparse
import warnings
import pandas as pd
import os
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--access_token', default=None, type=str)
    parser.add_argument('--cache_dir', default=None, type=str)
    parser.add_argument('--checkpoint', default='HuggingFaceH4/starchat-beta',
                        choices=['HuggingFaceH4/starchat-alpha', 'HuggingFaceH4/starchat-beta'], type=str)
    parser.add_argument('--data_load_name', default='code_smell_data.jsonl',)
    parser.add_argument('--result_save_name', default='code_smell_eval_llama.jsonl')
    parser.add_argument('--log_file_name', default='code_smell_eval_llama.log'),
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--candidate_num', default=1, type=int)
    args = parser.parse_args()

    return args

lang_cluster = ['c++', 'java', 'python', 'c', 'c#', 'ruby', 'delphi', 'go',
                'javascript', 'kotlin', 'php', 'd', 'perl', 'rust']

class StopAtSpecificTokenCriteria(StoppingCriteria):

    def __init__(self, token_id_list: List[int] = None):
        self.token_id_list = token_id_list

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def generate_text(prompt, temperature, max_new_tokens):
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(device)
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        # References: https://github.com/bigcode-project/starcoder/issues/73
        stopping_criteria=StoppingCriteriaList([
            StopAtSpecificTokenCriteria(token_id_list=[
                tokenizer.encode("<|end|>", return_tensors='pt').tolist()[0][0]
            ])
        ])
    ).to('cpu')
    response = [tokenizer.decode(output).split('<|assistant|>')[-1].replace('<|end|>', '').strip()
                for output in outputs]

    return response


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def count_message_tokens(content):
    tokens = tokenizer(content)['input_ids']
    num_tokens = len(tokens)

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




def add_program_synthesis(example):
    """
    Generate corresponding code based on the problem description (for subset)

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

    user_message = f"""As an expert code developer with years of experience, please provide the source code based on the problem description. The detailed information are as follows:
1. Problem description: {prob_desc_description}
2. Input specification: {prob_desc_input_spec}
3. Output specification: {prob_desc_output_spec}
4. Sample inputs: {prob_desc_sample_inputs}
5. Sample outputs: {prob_desc_sample_outputs}
6. Sample explanations: {prob_desc_notes}
7. Programming language: {lang} 
8. support programming language version: {env_map[lang.lower()]}
Respond should only with a string in the following JSON format:

[{{"version": specific version used in the programming language, "target code":  the code you produced in the respective programming language version."}}] """

    system_message = 'Below is a dialogue between a human and an AI assistant called StarChat.'
    prompt = f'<|system|>\n{system_message.strip()}<|end|>\n<|user|>\n{user_message.strip()}<|end|>\n<|assistant|>'
    logging.info('problem src_id: ' + str(prob_uid))

    input_tokens = count_message_tokens(prompt)
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning('Over input tokens limit: ' + str(prob_uid))
    try:
        response = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            candidate_num=candidate_num
        )
        logging.info('response: ' + str(response))

        if response is not None:
            output_tokens = count_message_tokens(response)
            logging.info('output tokens: ' + str(output_tokens))
            if output_tokens > max_new_tokens:
                logging.warning('Over total tokens limit ' + str(prob_uid) + ' lang: ' + str(lang))
                program_sythesis = []

            else:
                program_sythesis = response
        else:
            logging.warning('Respond content is none.')
            program_sythesis = []

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        program_sythesis = []

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

    user_message = f"""As an expert code developer proficient in multiple programming languages with years of experience, please translate the source code in {source_lang} to programming language {target_lang} within our supported version. 
The detailed information are as follows:
1. Target programming language: {target_lang}
2. support programming language version: {env_map[target_lang]}
3. Source code\n: {source_code}

Respond should only with a string in the following JSON format:

[{{"version": specific version used in the programming language, "target code":  the code you produced in the respective programming language version."}}] """

    system_message = 'Below is a dialogue between a human and an AI assistant called StarChat.'
    prompt = f'<|system|>\n{system_message.strip()}<|end|>\n<|user|>\n{user_message.strip()}<|end|>\n<|assistant|>'
    logging.info('problem src_id: ' + str(prob_uid))

    input_tokens = count_message_tokens(prompt)
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning('Over input tokens limit: ' + str(prob_uid))
    try:
        response = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            candidate_num=candidate_num
        )
        logging.info('response: ' + str(response))

        if response is not None:
            output_tokens = count_message_tokens(response)
            logging.info('output tokens: ' + str(output_tokens))
            if output_tokens > max_new_tokens:
                logging.warning('Over total tokens limit ' + str(prob_uid) + ' lang: ' + str(lang_cluster))
                translation_outcome = []
            else:
                translation_outcome = response
        else:
            logging.warning('Respond content is none.')
            translation_outcome = []

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        translation_outcome = []

    for i in range(len(translation_outcome)):
        logging.info('code translation  in: ' + target_lang + ' :' + str(translation_outcome[i]))
        example['code translation_' + str(i)] = translation_outcome[i]

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
    user_message = f"""As an expert code developer with years of experience, please debug the source code in {source_lang} based on the corresponding problem description and show the correct code. 
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

    system_message = 'Below is a dialogue between a human and an AI assistant called StarChat.'
    prompt = f'<|system|>\n{system_message.strip()}<|end|>\n<|user|>\n{user_message.strip()}<|end|>\n<|assistant|>'

    logging.info('problem src_id: ' + str(prob_uid))

    input_tokens = count_message_tokens(prompt)
    logging.info('input tokens: ' + str(input_tokens))
    if input_tokens > max_input_tokens:
        logging.warning('Over input tokens limit: ' + str(prob_uid))
    try:
        response = generate_text(
            prompt=prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            candidate_num=candidate_num
        )
        logging.info('response: ' + str(response))

        if response is not None:
            output_tokens = count_message_tokens(response)
            logging.info('output tokens: ' + str(output_tokens))
            if output_tokens > max_new_tokens:
                logging.warning('Over total tokens limit ' + str(prob_uid) + ' lang: ' + str(lang_cluster))
                repair_outcome = ''
            else:
                repair_outcome = response
        else:
            logging.warning('Respond content is none.')
            repair_outcome = ''

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        repair_outcome = ''

    for i in range(len(repair_outcome)):
        logging.info('Code repairing in: ' + source_lang + ' :' + str(repair_outcome[i]))
        example['Code repairing_' + str(i)] = repair_outcome[i]

    return example


def main():
    load_path = Path(__file__).parent.parent / Path('data') / Path(args.data_load_name)
    save_path = Path(__file__).parent.parent / Path('results') / Path(args.result_save_name)

    if 'program_synthesis_v3' in args.data_load_name:
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
    print(dataset)

    if 'program_synthesis' in args.data_load_name:
        dataset = dataset.map(add_program_synthesis)
    elif 'code_translation' in args.data_load_name:
        dataset = dataset.map(add_code_translation)
    elif 'code_debug' in args.data_load_name:
        dataset = dataset.map(add_code_repairing)
    print(dataset)

    dataset.to_json(save_path, lines=True)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # References: https://huggingface.co/blog/starcoder
    # References: https://huggingface.co/datasets/bigcode/ta-prompt
    # References: https://github.com/bigcode-project/starcoder/issues/101
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint,
        use_fast=True,
        trust_remote_code=True,
        token=args.access_token,
        cache_dir=args.cache_dir
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.float16,
        # load_in_4bit=True,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map='auto',
        token=args.access_token,
        cache_dir=args.cache_dir
    )
    print(f'Memory footprint: {model.get_memory_footprint() / 1e6:.2f} MB')

    candidate_num = args.candidate_candidate_num
    temperature = args.temperature

    max_input_tokens = tokenizer.model_max_length  # 1000000000000000019884624838656
    # The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
    max_new_tokens = 1024

    main()
    # python scripts/eval_starcoder.py