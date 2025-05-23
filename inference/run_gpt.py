
import openai
import backoff
import logging

import tiktoken
import argparse

from pathlib import Path
from datasets import load_dataset, Dataset

import os


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gpt-3.5-turbo',
                        choices=['gpt-3.5-turbo', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k-0613',
                                 'gpt-3.5-turbo-0301', 'gpt-4', 'gpt-4-0613', 'gpt-4-32k', 'gpt-4-32k-0613',
                                 'gpt-4-0314', 'gpt-4-32k-0314'],
                        type=str)
    parser.add_argument('--data_load_name', default='program_synthesis_data.jsonl', type=str)
    parser.add_argument('--result_save_name', default='program_synthesis_gpt3.jsonl', type=str)
    parser.add_argument('--log_file_name', default='program_synthesis_gpt3.logs', type=str)
    parser.add_argument('--temperature', default=0.5, type=float)
    parser.add_argument('--candidate_num', default=5, type=int)
    args = parser.parse_args()

    return args


# References: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def generate_text(model, prompt, temperature, candidate_num):
    messages = [{'role': 'user', 'content': prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        n=candidate_num
    )
    results = []
    for i in range(candidate_num):
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
        num_tokens = max(num_tokens, len(encoding.encode(content)))

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


def add_program_synthesis(example):
    """

    Program synthesis: Generate corresponding code based on the problem description

    """

    prob_uid = example['src_uid']
    prob_desc_description = example['description']
    prob_desc_input_spec = example['input_specification']
    prob_desc_output_spec = example['output_specification']
    prob_desc_sample_inputs = example['sample_inputs']
    prob_desc_sample_outputs = example['sample_outputs']
    prob_desc_notes = example['notes']
    lang = example['lang_cluster'].lower()

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

Respond should only with a string in the following JSON format, using Markdown code block:
[{{"version": specific version used in the programming language, "target code":  the code you produced in the respective programming language version."}}] """

    logging.info('problem src_id: ' + str(prob_uid))
    logging.info(prompt)
    input_tokens = count_message_tokens(prompt, args.model, 'input')
    logging.info('input tokens: ' + str(input_tokens))

    try:
        response = generate_text(
            model=args.model,
            prompt=prompt,
            temperature=temperature,
            candidate_num=candidate_num
        )
        logging.info('response: ' + str(response))

        if response is not None:
            program_sythesis = response
        else:
            logging.warning('Respond content is none.')
            program_sythesis = []

    except Exception as e:
        logging.error('Failed to generate text: ' + e.__str__())
        program_sythesis = []

    logging.info('program_synthesis in: ' + lang + ' :' + str(program_sythesis))
    example['program_synthesis'] = program_sythesis

    for i, generated_code in enumerate(program_sythesis):
        output_tokens = count_message_tokens(generated_code, args.model, 'output')
        logging.info('output tokens: ' + str(output_tokens))
        if input_tokens + output_tokens > max_tokens:
            logging.warning('Over total tokens limit ' + str(prob_uid) + ' lang: ' + str(lang))
            generated_code = ''
        logging.info('program_synthesis  in: ' + lang + ' :' + generated_code)
        example['program_synthesis_' + str(i)] = generated_code
    if len(program_sythesis) < candidate_num:
        for i in range(candidate_num - len(program_sythesis)):
            example['program_synthesis_' + str(i + len(program_sythesis))] = ''
    return example


def main():
    load_path = Path(__file__).parent.parent / Path('data') / Path(args.data_load_name)
    save_path = Path(__file__).parent / Path('results') / Path(args.result_save_name)

    dataset = load_dataset('json', split='train', data_files=str(load_path))
    dataset.cleanup_cache_files()  # for multiple evaluation

    dataset = dataset.map(add_program_synthesis)

    dataset.to_json(save_path, lines=True)


if __name__ == '__main__':
    args = parse_arguments()

    log_file_path = Path(__file__).parent / Path('logs') / Path(args.log_file_name)
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
    openai.api_key = os.getenv('OPENAI_API_KEY')
    model_max_tokens = {
        'gpt-3.5-turbo': 4096,
        'gpt-3.5-turbo-16k': 16385,
        'gpt-3.5-turbo-0613': 4096,
        'gpt-3.5-turbo-16k-0613': 16385,
        'gpt-3.5-turbo-0301': 4096,
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
