import subprocess
import json
from openai import OpenAI
import pandas as pd
import re
from termcolor import colored
import os
import shutil
from anthropic import Anthropic
import signal
from datetime import datetime

# Constants
ITERATIONS = 50
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
O1_MODEL = "o1-mini"
MAX_RUNTIME_MINUTES = 30  # Maximum runtime in minutes

print(colored("Phase 1: Setting up API Clients", "cyan"))
openai_client = OpenAI()
claude_client = Anthropic()

# Clear execution outputs file at start
output_file = "execution_outputs.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(f"=== New Run Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
print(colored("Cleared previous execution outputs", "green"))

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Script execution timed out")

def run_with_timeout(cmd, timeout_minutes):
    """Run command with timeout"""
    start_time = datetime.now()
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout_minutes * 60)
        execution_time = (datetime.now() - start_time).total_seconds()
        return stdout, stderr, process.returncode, execution_time
    except subprocess.TimeoutExpired:
        process.kill()
        return "", "Execution timed out after {} minutes".format(timeout_minutes), -1, timeout_minutes * 60

def get_timeout_improvement(code, train_sample, additional_info, execution_time, max_runtime):
    """Get improvements from O1 for timeout issues"""
    print(colored("Getting performance improvement suggestions...", "yellow"))
    try:
        messages = [{
            "role": "user",
            "content": f"""The machine learning code is taking too long to execute. It ran for {execution_time:.2f} seconds but needs to complete within {max_runtime * 60} seconds.
IMPORTANT: Optimize the code to run faster while maintaining accuracy. Consider:
1. Reducing model complexity without sacrificing too much accuracy
2. Using more efficient data processing
3. Optimizing hyperparameters for speed
4. Using faster algorithms if possible
5. Reducing cross-validation folds if necessary
6. DO NOT remove core functionality or important features
7. Keep GPU utilization if present
8. Maintain progress_report.json creation and all essential metrics
9. Always ensure you split the training set into train and test sets to validate the model

Here's the training data sample for context:
{train_sample}

Additional information about the task:
{additional_info}

Current code:
```python
{code}
```

Return ONLY the complete optimized code wrapped in ```python and ``` markers."""
        }]
        
        response = openai_client.chat.completions.create(
            model=O1_MODEL,
            messages=messages
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(colored(f"Error getting timeout improvements: {str(e)}", "red"))
        return None

def is_actual_error(stderr_text):
    """Check if stderr contains actual errors and not just warnings"""
    error_indicators = [
        "Traceback (most recent call last)",
        "Error:",
        "Exception:",
        "SyntaxError:",
        "NameError:",
        "TypeError:",
        "ValueError:",
        "ImportError:",
        "ModuleNotFoundError:",
        "IndexError:",
        "KeyError:",
        "AttributeError:",
        "IndentationError:"
    ]
    return any(indicator in stderr_text for indicator in error_indicators)

def get_claude_fix(error_message, code, train_sample, additional_info):
    """Get code fix from Claude for the error"""
    print(colored("Getting Claude's fix...", "yellow"))
    try:
        # Get the last 5000 characters of the error message
        truncated_error = error_message[-5000:] if len(error_message) > 5000 else error_message
        if len(error_message) > 5000:
            truncated_error = "...(earlier error output truncated)...\n" + truncated_error
            
        message = claude_client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=8000,
            temperature=0,
            system="""You are an expert Python developer. Your task is to fix code errors.
IMPORTANT: You must follow these rules:
1. Return ONLY the complete fixed code
2. Always wrap the entire code in ```python and ``` markers
3. Do not include any explanations or comments outside the code block
4. Return the FULL corrected code, not just the fixed part
5. Make sure all imports and functions are included
6. If using ML frameworks, include GPU checks and utilize GPU when available
7. For ModuleNotFoundError or ImportError:
   - Add necessary pip install commands at the start of the script
   - Use subprocess.check_call to run pip install
   - Keep the rest of the code unchanged unless there are other errors
   - Example:
     ```python
     import subprocess
     try:
         import missing_module
     except ImportError:
         subprocess.check_call(['pip', 'install', 'missing_module'])
         import missing_module

     ```
8. make sure that the script is creating and saving a progress_report.json file with the accuracy and other useful information

     """,
            messages=[{
                "role": "user",
                "content": f"""Fix this Python code error. Return ONLY the complete fixed code wrapped in ```python and ``` markers.
If there are missing imports (ModuleNotFoundError or ImportError), add pip install commands at the start of the script but keep the main machine learning code logic unchanged.
Make sure to check for GPU availability and use it when possible for ML frameworks.

Here's the training data sample for context:
{train_sample}

Additional information about the task:
{additional_info}

Error (last 5000 characters):
{truncated_error}

Code:
{code}"""
            }]
        )
        fixed_code = message.content[0].text
        if not fixed_code.strip().startswith("```python") or not fixed_code.strip().endswith("```"):
            print(colored("Warning: Claude's response is not properly formatted with code blocks", "red"))
            return None
        return fixed_code
    except Exception as e:
        print(colored(f"Error getting Claude's fix: {str(e)}", "red"))
        return None

print(colored("Phase 2: Data Loading and Preparation", "cyan"))
# Read the first 10 rows of train.csv
train_data = pd.read_csv('train.csv', nrows=10)
train_sample = train_data.to_string(index=False)

# Read additional info
with open('additional_info.txt', 'r', encoding='utf-8') as file:
    additional_info = file.read()

print(colored("Phase 3: ML Solution Generation and Execution", "cyan"))
previous_code = ""
output_file = "execution_outputs.txt"

for i in range(ITERATIONS):
    print(colored(f"Iteration {i+1}/{ITERATIONS}", "yellow"))
    
    # Move old solution, progress report, and submission to older_solutions folder
    if i > 0:
        older_solutions_dir = "older_solutions"
        os.makedirs(older_solutions_dir, exist_ok=True)
        
        # Move files with iteration number
        files_to_move = {
            "solution.py": f"solution_{i}.py",
            "progress_report.json": f"progress_report_{i}.json",
            "submission.csv": f"submission_{i}.csv"
        }
        
        for src, dst in files_to_move.items():
            if os.path.exists(src):
                shutil.move(src, os.path.join(older_solutions_dir, dst))
                print(colored(f"Moved {src} to {older_solutions_dir}/{dst}", "green"))

    initial_prompt = f"""You are an expert ML scientist. Your task is to write excellent Python code to solve a difficult data challenge.
IMPORTANT: You must follow these rules:
1. Return ONLY the complete code solution
2. Always wrap the entire code in ```python and ``` markers
3. Do not include any explanations or text outside the code block
4. Include all necessary imports at the top of the code
5. Include descriptive comments within the code
6. Check for GPU availability and use it when possible
7. Print clear messages about whether GPU or CPU is being used
8. make sure that the script is creating and saving a progress_report.json file with the accuracy and other useful information as well as the submission file to a file called submission.csv
9. keep the scripts fast and efficient but very accurate
10. script should be able to run and complete in {MAX_RUNTIME_MINUTES} minutes
11. always ensure you split the training set into train and test sets to validate the model

Here's the information about the challenge:

{additional_info}

Here are the first 10 rows of the training data:

{train_sample}

The data files available are train.csv and test.csv. Pay specific attention to column names.

Based on this information, write complete Python code that:
1. Load and preprocess both train.csv and test.csv
2. Perform exploratory data analysis without using any plots
3. Engineer relevant features
4. Select and train an appropriate machine learning model using the training data
   - Check for GPU availability and use it if available
   - Fall back to CPU if GPU is not available
   - Print clear message about which device is being used
5. Evaluate the model's performance using cross-validation. clearly print the accuracy every so often
6. Make predictions on the test set and print the accuracy
- save the progress report and the final accuracy to a file called progress_report.json as well as the submission file to a file called submission.csv
7. Prepare the submission file in the required format
- do not use any plots
- you can print any analysis or information necessary as this will be passed back to you for the next iteration to improve the code
- make sure you use utf-8 encoding when writing to files
- do not use logging instead print the information necessary

Return ONLY the complete code solution wrapped in ```python and ``` markers."""

    if i == 0:
        messages = [{"role": "user", "content": initial_prompt}]
    else:
        # Read the previous progress report
        prev_progress_path = os.path.join(older_solutions_dir, f"progress_report_{i}.json")
        with open(prev_progress_path, 'r', encoding='utf-8') as f:
            prev_progress = json.load(f)
            
        messages = [
            {"role": "user", "content": initial_prompt},
            {"role": "assistant", "content": f"Here's the previous code I generated:\n\n```python\n{previous_code}\n```"},
            {"role": "user", "content": f"""Improve this code based on the previous results. Your goal is to improve the accuracy as much as you can. Return ONLY the complete improved code wrapped in ```python and ``` markers.
Make sure to maintain or add GPU support with proper availability checks and fallback to CPU.
make sure that the script is creating and saving a progress_report.json file with the accuracy and other useful information
keep the scripts fast and efficient but very accurate
11. always ensure you split the training set into train and test sets to validate the model

Here's the training data sample for context:
{train_sample}

Additional information about the task:
{additional_info}

Previous model performance:
{json.dumps(prev_progress, indent=2)}

Focus on improving:
1. Cross-validation scores: {prev_progress.get('cross_validation_scores', [])}
2. Mean CV accuracy: {prev_progress.get('mean_cv_accuracy', 0)}
3. Feature importance (if available)
4. Model parameters and architecture"""}
        ]

    response = openai_client.chat.completions.create(
        model=O1_MODEL,
        messages=messages
    )

    print(colored("ML Scientist's Solution:", "cyan"))
    solution_content = response.choices[0].message.content
    print(solution_content)

    code_blocks = re.findall(r'```python(.*?)```', solution_content, re.DOTALL)
    if not code_blocks:
        print(colored("Error: No properly formatted Python code blocks found in the solution.", "red"))
        print(colored("Raw response:", "yellow"))
        print(solution_content)
        break

    current_code = '\n'.join(code_blocks).strip()
    has_error = True
    
    while has_error:  # Keep trying with Claude until no errors
        with open('solution.py', 'w', encoding='utf-8') as file:
            file.write(current_code)
        print(colored("Solution saved to solution.py", "green"))

        # Execute the generated code with timeout
        print(colored("Executing generated code:", "cyan"))
        stdout, stderr, returncode, execution_time = run_with_timeout(['python', 'solution.py'], MAX_RUNTIME_MINUTES)
        
        print(colored("Execution output:", "green"))
        print(stdout)
        
        if returncode == -1:  # Timeout occurred
            print(colored(f"Execution timed out after {MAX_RUNTIME_MINUTES} minutes", "red"))
            
            # Get optimization suggestions from O1
            optimized_code = get_timeout_improvement(
                current_code, 
                train_sample, 
                additional_info,
                execution_time,
                MAX_RUNTIME_MINUTES
            )
            
            if optimized_code and "```python" in optimized_code:
                code_blocks = re.findall(r'```python(.*?)```', optimized_code, re.DOTALL)
                if code_blocks:
                    current_code = code_blocks[0].strip()
                    print(colored("Applied performance optimizations. Retrying...", "green"))
                    continue
                else:
                    print(colored("Error: Optimization response was not properly formatted", "red"))
                    break
            else:
                print(colored("Could not get optimization suggestions. Moving to next iteration.", "red"))
                break
        
        if stderr:
            # Check if stderr contains actual errors or just warnings
            if is_actual_error(stderr):
                print(colored("Execution errors:", "red"))
                print(stderr)
                
                # Try to get fix from Claude
                fixed_code = get_claude_fix(stderr, current_code, train_sample, additional_info)
                
                if fixed_code and "```python" in fixed_code:
                    # Extract code from markdown if present
                    code_blocks = re.findall(r'```python(.*?)```', fixed_code, re.DOTALL)
                    if code_blocks:
                        current_code = code_blocks[0].strip()
                        print(colored("Applied Claude's fix. Retrying...", "green"))
                    else:
                        print(colored("Error: Claude's fix was not properly formatted", "red"))
                        break
                else:
                    print(colored("Could not get fix from Claude. Moving to next iteration.", "red"))
                    break
            else:
                print(colored("Warnings (non-critical):", "yellow"))
                print(stderr)
                has_error = False  # Continue if only warnings
        else:
            has_error = False  # No errors or warnings
        
        # Save execution output to file
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(f"Iteration {i+1}/{ITERATIONS}\n")
            f.write("Execution output:\n")
            f.write(stdout)
            f.write("\nExecution errors:\n")
            f.write(stderr)
            if returncode == -1:
                f.write(f"\nExecution timed out after {MAX_RUNTIME_MINUTES} minutes")
            f.write("\n" + "="*50 + "\n\n")
    
    # Update previous_code for the next iteration (O1 improvements)
    previous_code = current_code

print(colored("ML Scientist process completed.", "cyan"))
print(f"All execution outputs have been saved to {output_file}")
print(colored("All solutions, progress reports, and submissions have been saved with iteration numbers in the older_solutions directory.", "green"))
