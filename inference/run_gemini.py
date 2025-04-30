import time
from google import genai
import backoff
import logging
import argparse
from pathlib import Path
from datasets import load_dataset, Dataset
from google.genai import types
import json
from google.api_core.exceptions import GoogleAPIError
import os

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='gemini-2.0-flash', choices=['gemini-2.0-flash'], type=str)
    parser.add_argument('--data_load_name', default='program_synthesis_data.jsonl', type=str)
    parser.add_argument('--result_save_name', default='program_synthesis_gemini.jsonl', type=str)
    parser.add_argument('--log_file_name', default='program_synthesis_gemini.logs', type=str)
    parser.add_argument('--temperature', default=0.5, type=float)
    parser.add_argument('--candidate_num', default=5, type=int)
    args = parser.parse_args()
    return args

@backoff.on_exception(backoff.expo, GoogleAPIError)
def generate_text(client, model, prompt, temperature, candidate_num):
    response = client.models.generate_content(
        model=model,
        contents=[prompt],
        config=types.GenerateContentConfig(
            temperature=temperature,
            candidate_count=candidate_num,
        )
    )
    results = [candidate.content.parts[0].text for candidate in response.candidates]

    return results

env_map = {
    'python': ['python2', 'python3', ],
}

lang_cluster = ['python']

def add_program_synthesis(example, client):
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

Respond should only with a string in the following JSON format:
[{{"version": specific version used in the programming language, "target code": the code you produced in the respective programming language version."}}] """

    logging.info('problem src_id: ' + str(prob_uid))
    logging.info(prompt)

    try:
        response = generate_text(
            client=client,
            model=args.model,
            prompt=prompt,
            temperature=temperature,
            candidate_num=candidate_num
        )
        logging.info('response: ' + str(response))
        time.sleep(5)

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
        logging.info('program_synthesis  in: ' + lang + ' :' + generated_code)
        example['program_synthesis_' + str(i)] = generated_code

    if len(program_sythesis) < candidate_num:
        for i in range(candidate_num - len(program_sythesis)):
            example['program_synthesis_' + str(i + len(program_sythesis))] = ''

    return example

def main(client):
    load_path = Path(__file__).parent.parent / Path('data') / Path(args.data_load_name)
    save_path = Path(__file__).parent / Path('results') / Path(args.result_save_name)

    dataset = load_dataset('json', split='train', data_files=str(load_path))
    dataset.cleanup_cache_files()

    dataset = dataset.map(lambda example: add_program_synthesis(example, client))

    dataset.to_json(save_path, lines=True)

if __name__ == '__main__':
    args = parse_arguments()

    log_file_path = Path(__file__).parent / Path('logs') / Path(args.log_file_name)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='%(asctime)s - %(filename)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(filename=log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    client = genai.Client(api_key=os.getenv('GEMINI_API_KEY')) # Initialize with API key

    candidate_num = args.candidate_num
    temperature = args.temperature

    main(client)