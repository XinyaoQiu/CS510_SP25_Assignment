import json
import subprocess
import tempfile
import os
import argparse
import pandas as pd
import re

def clean_float_output(output):
    """Ensures floating-point outputs like -0.000000 are converted to 0.000000."""
    lines = output.strip().split("\n")
    cleaned_lines = []
    for line in lines:
        tokens = line.strip().split()
        cleaned_tokens = []
        for token in tokens:
            try:
                val = float(token)
                if abs(val) < 1e-9:
                    cleaned_tokens.append('0.000000')
                else:
                    cleaned_tokens.append(f"{val:.6f}")
            except ValueError:
                cleaned_tokens.append(token)
        cleaned_lines.append(" ".join(cleaned_tokens))
    return "\n".join(cleaned_lines)

def preprocess_input(raw_input):
    if isinstance(raw_input, list):
        # Join list to one string, then split cleanly on newlines
        raw_input = "\n".join(raw_input)

    # Normalize all line endings and remove extra surrounding whitespace
    input_data = raw_input.replace('\r\n', '\n').replace('\r', '\n').strip()

    # Split and clean each line
    lines = [line.strip() for line in input_data.split('\n')]

    # Rejoin into final cleaned input string
    return "\n".join(lines) + "\n"


def normalize_output(output):
    """Removes extra spaces, ensures consistent newline formatting, and fixes floating-point edge cases."""
    output = "\n".join(line.rstrip() for line in output.replace("\r\n", "\n").strip().split("\n"))
    return clean_float_output(output)

def run_python_code(data):
    """Runs the provided Python code against test cases and revises version if misclassified."""

    source_code, lang, testcases = data["source_code"], data['lang'], eval(data["testcases"])  # Convert test cases from string to list
    results = []

    filtered_testcases = []
    for testcase in testcases:
        if any('...' in output for output in testcase['output']) or any('...' in input for input in testcase['input']):
            continue
        filtered_testcases.append(testcase)
    testcases = filtered_testcases
    if len(testcases) == 0:
        print("All testcases skipped. No execution needed.")
        return []

    tried_versions = [lang, "python2" if lang == "python3" else "python3"]
    final_lang = None
    success = False

    # Create temporary Python file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode='w', encoding='utf-8') as temp_file:

        temp_file.write(source_code)
        temp_file.flush()
        temp_filename = temp_file.name

    try:
        # Try both versions only once
        for attempt_lang in tried_versions:
            print(f"Trying {attempt_lang}")

            # Test with first test case only to check version compatibility
            first_case = testcases[0]
            input_data = preprocess_input(first_case["input"])

            if attempt_lang == "python2":
                actual_output, error_output = execute_command(["python2", temp_filename], input_data=input_data)
            else:
                actual_output, error_output = execute_command(["python", temp_filename], input_data=input_data)

            # Check if syntax or version errors occurred
            if error_output != '':
                print(f"Execution failed with {attempt_lang}, error message is {error_output}, trying alternative version...\n")
                continue  # Try the other version
            else:
                final_lang = attempt_lang
                success = True
                if final_lang != lang:
                    data['lang'] = attempt_lang
                    print(f"{data['src_uid']}'s lang is changed to {final_lang}. \n")
                break  # Stop trying other versions

        if not success:
            # Both versions failed, return error for all cases
            for testcase in testcases:
                input_data = preprocess_input(testcase["input"])
                results.append({
                    "input": input_data,
                    "expected": testcase["output"],
                    "actual": "",
                    "error": error_output,
                    "exec_outcome": f"ERROR: {error_output}",
                    "final_lang": None
                })
            return results

        # Run all test cases with determined version
        for id, testcase in enumerate(testcases):
            input_data = preprocess_input(testcase["input"])
            expected_outputs = testcase["output"]

            if final_lang == "python2":
                actual_output, error_output = execute_command(["python2", temp_filename], input_data=input_data)
            else:
                actual_output, error_output = execute_command(["python", temp_filename], input_data=input_data)

            # Determine result
            if error_output != '':
                exec_outcome = f"ERROR: {error_output}"
                print(exec_outcome)
            elif any(normalize_output(actual_output) == normalize_output(expected) for expected in expected_outputs):
                exec_outcome = "PASSED"
            else:
                exec_outcome = "FAILED"
                print(f"failed at case ({id}/{len(testcases)})")
                print(f"expected output: {[normalize_output(expected) for expected in expected_outputs]}\n"
                      f"actual output: {normalize_output(actual_output)}\n")
            # Append results
            results.append({
                "input": input_data,
                "expected": expected_outputs,
                "actual": actual_output,
                "error": error_output,
                "exec_outcome": exec_outcome,
            })
            if exec_outcome != "PASSED":
                break
    finally:
        os.remove(temp_filename)

    return results


def execute_command(command, input_data=None):
    """Executes a Python script and ensures input is passed correctly."""

    try:
        outcome = subprocess.run(
            command,
            input=input_data,  # Send input via stdin
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=10,  # Prevent infinite waits
            shell=False
        )

        return outcome.stdout.strip(), outcome.stderr.strip()

    except subprocess.TimeoutExpired:
        return None, "Error: Execution timed out. Check input formatting."

    except Exception as e:
        return None, f"Error: {e}"


def evaluate_code(json_input):
    """Evaluates a Python code snippet based on the provided JSON input."""
    data = json.loads(json_input)
    evaluation_results = run_python_code(data)

    # Construct evaluation record
    evaluation_record = {
        "lang_cluster": data["lang_cluster"],
        "src_uid": data["src_uid"],
        "difficulty": data["difficulty"],
        "exec_outcome": evaluation_results
    }

    return evaluation_record

def load_json_from_markdown(markdown_string):
    """Extracts and loads JSON from a Markdown code block."""
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', markdown_string)
    if json_match:
        json_string = json_match.group(1).strip()
        json_string = json_string.replace('\\""', '\\"\\"')
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            return None
    else:
        json_string = markdown_string
        try:
            return json.loads(json_string)[0]
        except json.JSONDecodeError as e:
            print(f"JSONDecodeError: {e}")
            return None

def count_passed_problems(results_path):
    """Counts the number of passed problems based on execution results."""
    record_dict = {"python": [[], []]}  # Track passed problems for Python only

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            result_content = json.loads(line)
            pass_flag = all(outcome["exec_outcome"] == "PASSED" for outcome in result_content["exec_outcome"])

            if pass_flag:
                if result_content["src_uid"] not in record_dict["python"][0]:
                    record_dict["python"][0].append(result_content["src_uid"])
                    record_dict["python"][1].append(result_content["difficulty"])

    HARD_BAR = 1501
    NON_BAR = 2701

    E_count = sum(1 for diff in record_dict["python"][1] if diff < HARD_BAR)
    H_count = sum(1 for diff in record_dict["python"][1] if HARD_BAR <= diff < NON_BAR)

    print("\nNumber of problems solved by Python:")
    print("Easy:", E_count, "Hard:", H_count)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--basic_dir", type=str, default="./data/")
    parser.add_argument("--inference_dir", type=str, default="./inference/results/")
    parser.add_argument("--model_name", type=str, default="gpt3")
    parser.add_argument("--exec_result_dir", type=str, default="./exec_results/")
    args = parser.parse_args()

    basic_path = os.path.join(args.basic_dir, "program_synthesis_data.jsonl")

    inference_path = os.path.join(args.inference_dir, f"program_synthesis_{args.model_name}.jsonl")
    exec_result_dir = args.exec_result_dir
    os.makedirs(exec_result_dir, exist_ok=True)
    # refine_and_save_data(basic_path, tmp_path, inference_path, exec_result_dir)
    results_path = os.path.join(exec_result_dir, f"program_synthesis_eval_{args.model_name}.jsonl")
    basic_data = []

    with open(basic_path, "r", encoding="utf-8") as f:
        for line in f:
            basic_data.append(json.loads(line))

    inference_data = []
    with open(inference_path, "r", encoding="utf-8") as f:
        for line in f:
            inference_data.append(json.loads(line))

    with open(results_path, "w", encoding="utf-8") as results_file:
        for item in inference_data:
            solved = False
            for i in range(6):  # Check program_synthesis_0 to program_synthesis_5
                code_key = f"program_synthesis_{i}"
                if code_key in item and item[code_key]:
                    try:
                        code_to_evaluate = json.dumps({
                            "source_code": load_json_from_markdown(item[code_key])[0]["target code"],
                            "testcases": item["testcases"],
                            "lang_cluster": item["lang_cluster"],
                            "lang": load_json_from_markdown(item[code_key])[0]["version"],
                            "src_uid": item["src_uid"],
                            "difficulty": item["difficulty"]
                        })
                        evaluation_result = evaluate_code(code_to_evaluate)
                        pass_flag = all(
                            outcome["exec_outcome"] == "PASSED" for outcome in evaluation_result["exec_outcome"])

                        if pass_flag:
                            solved = True
                            results_file.write(json.dumps(evaluation_result) + "\n")
                            break  # If one solution passes, no need to check others

                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON: {e}")
                        print(f"Skipping code {code_key} for item: {item}")
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")
                        print(f"Skipping code {code_key} for item: {item}")

            if not solved:
                print(f"No solution passed all test cases for item: {item}")
                # Optionally, you can write a default result to results_file here.

    # Count passed problems
    count_passed_problems(results_path)


if __name__ == "__main__":
    main()