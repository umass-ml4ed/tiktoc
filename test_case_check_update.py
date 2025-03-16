import pickle
import os
import yaml
import pandas as pd
from ast import literal_eval
import numpy as np
from munch import Munch
import subprocess
import re
import threading
from collections import defaultdict
import pdb
import matplotlib.pyplot as plt
from collections import Counter
from pdb import set_trace
from eval import *

mydict = yaml.safe_load(open("configs_okt.yaml", "r"))
configs = Munch(mydict)

def check_generated_results(configs, now):
    with open(os.path.join(configs.model_save_dir, now, 'eval_logs.pkl'), 'rb') as f:
        data = pickle.load(f)
        print(data.keys())
        

# Checked the available good test cases
# Return: 
#  1. sat_questions: dictionary of (Question Id: Question Description)
#  2. pandas dataframe: subset that contains specific test case input and output for "All Good" test case questions
def test_case_check():
    prompt_df = pd.read_csv(os.path.join('test-case-query-results/prompt_concept_summary.csv'), on_bad_lines='warn')
    problem_id_all = set(prompt_df['ProblemID'])
    all_good = prompt_df.loc[prompt_df['Test Case Status'] == 'All Good']
    problem_id = all_good['ProblemID']

    test_info_df = pd.read_csv(os.path.join('test-case-query-results/test_cases-1-26-24.csv'), on_bad_lines='skip')

    coding_prompt_id = set(test_info_df['coding_prompt_id'].unique())
    ls = [29, 37, 106, 236, 239, 240]
    for i in ls:
        coding_prompt_id.add(i)
    coding_prompt_id = {int(item) for item in coding_prompt_id if not (isinstance(item, float) and np.isnan(item))}

    sat_questions = dict(zip(all_good['ProblemID'], all_good['Requirement']))

    sat_id = sat_questions.keys()
    good_test_case = test_info_df[test_info_df['coding_prompt_id'].isin(sat_id)]

    return sat_questions, good_test_case

def nested_dict():
    return defaultdict(list)    

def split_large_dataset(configs, baseline):
    path = os.path.join(configs.data_path, baseline)

    with open(path, 'rb') as f:
        data = pickle.load(f)

        length = len(data['prompts'])
        cnt = 0
        for i in range(0, length, 2000):
            code_pair_sub = {}
            code_pair_sub['ground_truth_codes'] = data['ground_truth_codes'][i: i+2000]
            code_pair_sub['prompts'] = data['prompts'][i: i+2000]
            code_pair_sub['students'] = data['students'][i: i+2000]
            code_pair_sub['scores'] = data['scores'][i:i+2000]

            filename = 'val_testcase_pairs_sub_' + str(cnt) + '.pkl'
            with open(os.path.join(configs.data_path, filename), 'wb') as sub:
                pickle.dump(code_pair_sub, sub)
            cnt += 1




def validate_prompt_test_set(configs, all_good_test, now, baseline=None):
    if baseline:
        path = os.path.join(configs.data_path, baseline)
    else:
        path = os.path.join(configs.model_save_dir, now, 'eval_logs.pkl')

    with open(path, 'rb') as f:
        data = pickle.load(f)
        
        prompts = data['prompts']

        unique_prompts = set(prompts)
        all_good_question = list(all_good_test.values())
        all_good_qustion_id = list(all_good_test.keys())
        
        desc_id_dict = {}

        valid_questionID = []
        for prompt in unique_prompts:
            if prompt in all_good_question:
                ind = all_good_question.index(prompt)
                valid_questionID.append(all_good_qustion_id[ind])
                desc_id_dict[prompt] = all_good_qustion_id[ind]

        # Create data structure for whole all good student code pair
        students = data['students']

        if not baseline:
            generated_codes = data['generated_codes']
        else:
            scores = data['scores']

        gt_codes = data['ground_truth_codes']
        student_code_pair = defaultdict(nested_dict)

        for i in range(len(students)):
            prompt_i = prompts[i]
            if prompt_i in all_good_question:
                student_i = students[i]
                if not baseline:
                    generated_code_i = generated_codes[i]
                    student_code_pair[student_i]['generated_codes'].append(generated_code_i)
                else:
                    student_code_pair[student_i]['scores'].append(scores[i])
                ground_truth_code_i = gt_codes[i]
                student_code_pair[student_i]['prompts'].append(prompt_i)
                student_code_pair[student_i]['ground_truth_codes'].append(ground_truth_code_i)
                # cnt += 1
        
        # print('total questions:'cnt)

        return student_code_pair, desc_id_dict

# None is returned for method name and output_type when generated code is invalid:
# For example: there is no () in the generated code which lead to compile error. (public boolean isSix)
def extract_function_features(code):
    code = code.split('\n')
    for line in code:
        if 'public' in line or 'private' in line:
            line = line.split(' ')
            for i in range(len(line)):
                word = line[i]
                if '(' in word:
                    method_name = word.split('(')[0]
                    output_type = line[i-1]
                    return method_name, output_type
    return None, None

# extract_function_feature_complete handles the case when there is helper function in the code
# the function name and output type of main function will be returned
def extract_function_feature_complete(code, method_name_gt):
    function_pair = {}
    split_code = re.split(r'(?=(public))', code)
    split_code = [piece for piece in split_code if piece]
    if split_code.count('public') == 1:
        main_code = split_code[-1]
        method_name, output_type = extract_function_features(main_code)
        return method_name, output_type
    else:
        for code_piece in split_code:
            if code_piece != 'public':
                method_name, output_type = extract_function_features(code_piece)
                if method_name == method_name_gt:
                    return method_name, output_type

    return None, None



# Without handling the missing good question(37) from the dataset 
def uniq_test_construct(good_test_case):
    question_input_dict = {}
    grouped = good_test_case.groupby('coding_prompt_id')
    for name, group in grouped:
        if name == 34 or name == 39 or name == 40:
            inp = group['input'].tolist()
            clean_input = [i.rstrip('"').replace('\\', '"') for i in inp]
            question_input_dict[int(name)] = clean_input
        else:
            question_input_dict[int(name)] = group['input'].tolist()
    
    return question_input_dict

def handle_uniq_test_exception(question_input_dict):
    df_q37 = pd.read_csv(os.path.join('test-case-query-results/test_case_37.csv'), on_bad_lines='warn')
    df_q37['total_input'] = df_q37[['input_1', 'input_2']].apply(lambda x: ', '.join(x[x.notnull()]).rstrip('"').replace('\\', ''), axis=1)
    processed = df_q37['total_input'].tolist()
    cleaned_37 = ['"'+i+'"' for i in processed]
    # print(cleaned_37)
    question_input_dict[37] = cleaned_37
    
    return question_input_dict

def find_max_test_cases(question_input_dict):
    test_cases = question_input_dict.values()
    t_max = max([len(i) for i in test_cases])
    return t_max


def create_unit_test_case_function(function_name, test_cases_list):
    '''
    Creates a unit test case function for the given function name, code and test cases
    Output is a string representing a valid Java code
    '''
    unit_test_case_function_code = 'public void {:s}_test(String []s)'.format(function_name)
    unit_test_case_function_code += '{\n'
    # add print statements for test cases
    test_cases_prints = ''
    for test_case in test_cases_list:
        test_cases_prints += 'System.out.println({:s}({:s}));\n'.format(function_name, test_case)
    unit_test_case_function_code += test_cases_prints
    unit_test_case_function_code += '}\n'
    return unit_test_case_function_code



def get_main_function_code(class_name, function_name):
    '''
    Creates a main function for the given function name
    Output is a string representing a valid Java code
    '''
    main_function_code = 'public static void main(String []args)'
    main_function_code += '{\n'
    # Create an object of the class
    main_function_code += '{:s} obj = new {:s}();\n'.format(class_name, class_name)
    # Call the test function
    main_function_code += 'obj.{:s}_test(args);\n'.format(function_name)
    main_function_code += '}\n'
    return main_function_code

def get_complete_java_code(class_name, code, unit_test_case_function_code, main_function_code):
    '''
    Weaves everything together into a class
    Output is a string representing a valid Java code
    '''
    complete_java_code = 'public class {:s}'.format(class_name)
    complete_java_code += '{\n'
    complete_java_code += code
    complete_java_code += '\n'
    complete_java_code += unit_test_case_function_code
    complete_java_code += '\n'
    complete_java_code += main_function_code
    complete_java_code += '\n'
    complete_java_code += '}\n'
    return complete_java_code

# TODO: End indedx check
def parse_java_code(java_code):
    '''
    Takes Java code as input and returns the function signature and the main body
    '''
    java_code = java_code.strip()
    signature = re.search(r'(.*?){', java_code, re.DOTALL).group(1).strip()
    body_start_index = java_code.index("{") + 1
    body_end_index = java_code.rindex("}")

    body = java_code[body_start_index : body_end_index].strip()
    return signature, body



# TODO: Modify error information when running for ground truth code
def handle_exception(student_code, output_type=None):
    '''
    Wraps student code in a try-catch block to handle exceptions
    '''
    # extract function signature and main body (in between the outermost two curly braces)
    function_signature, main_body = parse_java_code(student_code)
    # TODO: Handle helper method in main body

    helper_keyword_pattern = re.compile(r'\bpublic\b')

    check_helper = helper_keyword_pattern.search(main_body)
    if bool(check_helper):
        start_pos = check_helper.start()
        end_bracket =  main_body.rfind('}', 0, start_pos)
        main_part = main_body[:end_bracket]
        helper =  main_body[start_pos:] + '\n' + '} \n'
    else:
        main_part = main_body
        helper = ''

    except_code = 'try {\n'
    except_code += main_part
    except_code += '\n'
    except_code += '} catch (Exception e) {\n'
    # except_code += 'System.out.print(e + " ");\n'
    except_code += 'System.out.print("Error ");\n'
    if output_type == 'int':
        except_code += 'return -999;\n'
    elif output_type == 'boolean':
        except_code += 'return false;\n'
    elif output_type == 'String':
        except_code += 'return "ERROR";\n'
    elif output_type == 'int[]':
        except_code += 'return new int[] {-999};\n'
    elif output_type == 'String[]':
        except_code += 'return new String[] {"ERROR"};\n'
    except_code += '}\n'

    # wrap the main code in the function defintion
    new_code = function_signature + '{\n' + except_code + '}\n'
    new_code += helper
    return new_code

def construct_java_code(test_cases, function_name, class_name, student_code, output_type):
    '''
    Constructs the full Java code
    '''
    # create a unit test case function
    unit_test_case_function_code = create_unit_test_case_function(function_name, test_cases)
    # create a main function
    main_function_code = get_main_function_code(class_name, function_name)
    # TODO: Wrap student code in a try-catch block to handle exceptions
    exception_handeled_code = handle_exception(student_code, output_type)
    # Weave everything together into a class
    complete_java_code = get_complete_java_code(class_name, exception_handeled_code, unit_test_case_function_code, main_function_code)
    return complete_java_code


def save_code(group, code, p):
    '''
    Saves the generated code
    '''
    group_tuple = literal_eval(group)
    # Save the code
    if not os.path.exists('compiler_code/{:d}_{:d}'.format(group_tuple[0], group_tuple[1])):
        os.mkdir('compiler_code/{:d}_{:d}'.format(group_tuple[0], group_tuple[1]))
    with open('compiler_code/{:d}_{:d}/{:s}.java'.format(group_tuple[0], group_tuple[1], p), 'w') as f:
        f.write(code)

# TODO: System.out.println() in the original code will be written into the output file.
def run_command_test(java_code_group_path, code_name, stop_event):
    if not os.path.exists('{:s}/output'.format(java_code_group_path)):
        os.mkdir('{:s}/output'.format(java_code_group_path))
    
    output_dir = os.path.join(java_code_group_path, 'output')

    command = f'java {java_code_group_path}/{code_name}.java'
    output_file = f'{output_dir}/output_{code_name}.txt'

    with open(output_file, 'w') as f:
        process = subprocess.Popen(command, shell=True,stdout=f, stderr=subprocess.STDOUT)

        while process.poll() is None:
            if stop_event.is_set():
                process.terminate()
                break

        if stop_event.is_set():
            try:
                process.wait(timeout=5)  # Wait for process to terminate
            except subprocess.TimeoutExpired:
                process.kill()


def run_command(java_code_group_path, code_name, stop_event):
    # execute the code using a system call and record the output as text
    # create a directory for the output
    if not os.path.exists('{:s}/output'.format(java_code_group_path)):
            os.mkdir('{:s}/output'.format(java_code_group_path))
    
    # execute the code
    while not stop_event.is_set():
        try:
            os.system('java {:s}/{:s}.java output > {:s}/output/output_{:s}.txt'.format(java_code_group_path, code_name, java_code_group_path, code_name))
        except Exception as e:
            print(e)
        break


# Returns the list of error, error type indicating compile error or runtime error, and number of errors
def run_command_subprocess(java_code_group_path, code_name):
    compile_command = ['javac', os.path.join(java_code_group_path, f'{code_name}.java')]
    run_command = ['java', '-cp', java_code_group_path, code_name]
    
    # Compile Java code
    compile_result = subprocess.run(compile_command, capture_output=True, text=True)
    if compile_result.returncode != 0:
        matches = re.findall(r'error: (.+?)(\n|$)', compile_result.stderr)
        if matches:
            errors = [match[0] for match in matches]
           
            # print(errors)
            return errors, 0, len(errors)
        return [compile_result.stderr], 0, 1

    # Run compiled Java program
    
    run_result = subprocess.run(run_command, capture_output=True, text=True)
    if run_result.returncode != 0:
        matches = re.findall(r'(Exception|Error): (.+?)(\n|$)', run_result.stderr)
        if matches:
            errors = [match[0] for match in matches]
            return errors, 1, len(errors)
        return [run_result.stderr], 1, 1


def execute_and_store(raw_group, code_name):
    '''
    Executes the code and stores the output in a file
    Returns the output of the execution
    '''
    group_elements = literal_eval(raw_group)
    group = '{:d}_{:d}'.format(group_elements[0], group_elements[1])
    java_code_group_path = 'compiler_code/{:s}'.format(group)
    # java_code_group_path = 'compiler_code_all/{:s}'.format(group)
    
    stop_event = threading.Event()
    # thread = threading.Thread(target=run_command, args=(java_code_group_path, code_name, stop_event))
    thread = threading.Thread(target=run_command_test, args=(java_code_group_path, code_name, stop_event))
    thread.start()
    thread.join(timeout=10)  # Set the timeout in seconds

    if thread.is_alive():
        print('in here')
        stop_event.set()
        thread.join()  # Ensure the thread is terminated
        return "Timeout"
    
    # read the output
    with open('{:s}/output/output_{:s}.txt'.format(java_code_group_path, code_name), 'r') as f:
        output = f.read()
    return output

def execute_and_store_update(raw_group, code_name, timeout=30):
    group_elements = literal_eval(raw_group)
    group = '{:d}_{:d}'.format(group_elements[0], group_elements[1])
    java_code_group_path = 'compiler_code/{:s}'.format(group)

    if not os.path.exists('{:s}/output'.format(java_code_group_path)):
        os.mkdir('{:s}/output'.format(java_code_group_path))
    
    output_dir = os.path.join(java_code_group_path, 'output')

    command = f'java {java_code_group_path}/{code_name}.java'
    output_file = f'{output_dir}/output_{code_name}.txt'


    try:
        with open(output_file, 'w') as f:
            process = subprocess.Popen(command,shell=True, stdout=f, stderr=subprocess.STDOUT)
            process.wait(timeout=timeout)
        
        with open(output_file, 'r') as f:
            output = f.read()
        return output
    
    except subprocess.TimeoutExpired:
        process.kill()
        return 'Timeout'



def execute_and_store_error_subprocess(raw_group, code_name):
    '''
    Executes the code and stores the output in a file
    Returns the output of the execution
    '''
    group_elements = literal_eval(raw_group)
    group = '{:d}_{:d}'.format(group_elements[0], group_elements[1])
    java_code_group_path = 'compiler_code/{:s}'.format(group)
    # java_code_group_path = 'compiler_code_all/{:s}'.format(group)

    error_st = run_command_subprocess(java_code_group_path, code_name)
    return error_st
    


def get_test_case_solution(test_case_df):
    solutions = test_case_df.groupby('coding_prompt_id')
    solution_dict = {}
    for name, group in solutions:
        if name == 34 or name == 40:
            sol = group['expected_output'].tolist()
            cleaned_out = [i.rstrip('"').replace('\\', '') for i in sol]
            solution_dict[int(name)] = cleaned_out
        else:
            solution_dict[int(name)] = group['expected_output'].tolist()
    
    df_q37 = pd.read_csv(os.path.join('test-case-query-results/test_case_37.csv'), on_bad_lines='warn')
    processed = df_q37['expected_output'].tolist()
    processed_convert = [str(i).lower() for i in processed]
    
    solution_dict[37] = processed_convert

    return solution_dict

# TODO: change return objects to get both accuracy score and binary list
def get_score(solution, student_answer):
    if len(student_answer) == 0 or len(student_answer) >= 2 * len(solution):
        return 0, [0]*len(solution)
    
    # if len(solution) > len(student_answer):
    #     set_trace()

    acc_list = [1 if solution[i] == student_answer[i] else 0 for i in range(len(solution))]
    accuracy = sum(acc_list) / len(solution)
    # accuracy = sum(1 for x,y in zip(solution, student_answer) if x == y) / len(solution)
    return accuracy, acc_list

def get_granular_correctness(solution, student_answer, generated_answer):
    if len(generated_answer) == 0 and len(student_answer) == 0:
        return 1
    if len(generated_answer) == 0 or len(student_answer) == 0:
        return 0

    acc_student = [1 if solution[i] == student_answer[i] else 0 for i in range(len(solution))]
    acc_gen = [1 if solution[i] == generated_answer[i] else 0 for i in range(len(solution))]
    granular_score = sum(1 for x, y in zip(acc_student, acc_gen) if x == y) / len(solution)

    return granular_score


def is_valid_code(code):
    is_valid = True
    code_parts = code.split()
    if code_parts.count('int') > 200 or code_parts.count('String') > 200 or code_parts.count('boolean') > 200 or len(code_parts) < 4:
        is_valid = False
    return is_valid

def preprocess_code(code):
    code_lines = code.split('\n')
    code_remove_print = [i for i in code_lines if 'System.out.print' not in i]
    full_code = ''.join(code_remove_print).replace('\r', '\n')
    return full_code


def get_test_result_for_generated_code(good_test_case, student_code_pair, desc_id_pair, metric='MSE', eval=True):
    question_input_dict = uniq_test_construct(good_test_case)
    question_input_dict = handle_uniq_test_exception(question_input_dict)
    solution = get_test_case_solution(good_test_case)

    full_question_id_dict = {1: '439, 1', 3: '439, 3', 5: '439, 5', 12: '439, 12', 13: '439, 13', 17: '487, 17', 20: '487, 20', 21: '487, 21', 22: '487, 22', 24: '487, 24', 25: '487, 25', 34: '492, 34', 37: '492, 37', 39: '492, 39', 40: '492, 40', 46: '494, 46', 71: '502, 71'}

    # Used dictionary to store ground truth code function name.
    gt_function_name_dict = {1: ('sortaSum', 'int'), 3: ('in1To10', 'boolean'), 5: ('answerCell', 'boolean'), 12: ('squirrelPlay', 'boolean'), 13: ('caughtSpeeding', 'int'), 17: ('redTicket', 'int'), 20: ('loneSum', 'int'), 21: ('luckySum', 'int'), 22: ('noTeenSum', 'int'), 24: ('blackjack', 'int'), 25: ('evenlySpaced', 'boolean'), 34: ('zipZap', 'String'), 37: ('endOther', 'boolean'), 39: ('xyBalance', 'boolean'), 40: ('getSandwich', 'String'), 46: ('isEverywhere', 'boolean'), 71: ('canBalance', 'boolean')}
    total_cnt, error_cnt, time_out_cnt = 0, 0, 0


    final_gt_ls, final_compiled_ls = [], []


    if eval:
        edge_cnt, mse_overall, granular_correctness_overall = 0, 0, 0

        # Compile and Doesn't Compile Situation Stats Calculation
        all_compile_cnt = 0
        mse_all_compile, granular_error_compile = 0, 0

        mse_gt_compile, granular_error_gt_compile, gt_compile_cnt = 0, 0, 0

        mse_gen_compile, granular_error_gen_compile, gen_compile_cnt = 0, 0, 0

        mse_no_compile, granular_error_no_compile, no_compile_cnt = 0, 0, 0

        group_cases_dict = {0: {0 : {'MSE':0.0, 'cnt':0, 'granular': 0.0, 'stats_ls':[]}, 1: {'MSE':0.0, 'cnt':0, 'granular': 0.0, 'stats_ls':[]}, 2: {'MSE':0.0, 'cnt':0, 'granular': 0.0, 'stats_ls':[]}}, 1: {0 : {'MSE':0.0, 'cnt':0, 'granular': 0.0, 'stats_ls':[]}, 1: {'MSE':0.0, 'cnt':0, 'granular': 0.0, 'stats_ls':[]}, 2: {'MSE':0.0, 'cnt':0, 'granular': 0.0, 'stats_ls':[]}}, 2: {0 : {'MSE':0.0, 'cnt':0, 'granular': 0.0, 'stats_ls':[]}, 1: {'MSE':0.0, 'cnt':0, 'granular': 0.0, 'stats_ls':[]}, 2: {'MSE':0.0, 'cnt':0, 'granular': 0.0, 'stats_ls':[]}}}
        gt_status, gen_status = 0, 0

        generated_error_ls, gt_error_ls = [], []

        valid_cnt, name_mismatch_cnt, both_pass_debug_ls = 0, 0, []

    for student in student_code_pair.keys():
        question_code_pair = student_code_pair[student]
        for i in range(len(question_code_pair['prompts'])):
            question_i = question_code_pair['prompts'][i]
            q_id = desc_id_pair[question_i]
            solution_i = solution[q_id]
            group_i = full_question_id_dict[q_id]
            spec_test_case = question_input_dict[q_id]
            method_name_gt, output_type_gt = gt_function_name_dict[q_id]

            if eval:
                generated_code_i = question_code_pair['generated_codes'][i]
                valid_gen, gen_status, accuracy, student_answer_i = True, 0, 0, []

                method_name, output_type = extract_function_feature_complete(generated_code_i, method_name_gt) 
                if method_name is None or output_type is None:
                    edge_cnt += 1
                    valid_gen = False
                elif generated_code_i.rfind("}") == -1:
                    print('} missing code:', generated_code_i)
                    valid_gen = False
                    
                if valid_gen:
                    valid_cnt += 1
                    full_part = construct_java_code(spec_test_case, method_name, method_name.upper()+student, generated_code_i, output_type)

                    # Might need to change during full submission case, since same function will be submitted more than once for same student
                    save_code(group_i, full_part, method_name.upper()+student)
                    # output_i = execute_and_store(group_i, method_name.upper()+student)
                    output_i = execute_and_store_update(group_i, method_name.upper()+student)

                    # added to handle compilation error in full submission
                    if 'compilation failed' in output_i or output_i == 'TIMEOUT':
                        print(output_i)
                    else:
                        student_answer_i = output_i.split('\n')
                    
                    # print('Student: ', student)
                    # print('Question No: ', group_i)
                    # print('Expected Output: ', solution_i)
                    # print('Student Answer: ', student_answer_i[:-1])
                    accuracy, bi_acc_ls = get_score(solution_i, student_answer_i[:-1])
                    # print('Accuracy on {:s}: {:.2f}%'.format(group_i, accuracy*100))

                    if len(student_answer_i[:-1]) > 0 and accuracy == 1:
                        gen_status = 2
                    elif len(student_answer_i[:-1]) > 0:
                        gen_status = 1
                    else:
                        generated_error_ls.append((group_i, method_name.upper()+student))
            
                else:
                    bi_acc_ls = [0] * len(solution_i)
            else:
                score_i = question_code_pair['scores'][i]

            # Compile the Grount Truth code for granular test case check and final score validation on original data set
            gt_status = 0
            gt_code_i = question_code_pair['ground_truth_codes'][i]

            gt_code_i = preprocess_code(gt_code_i)

            full_part_gt = construct_java_code(spec_test_case, method_name_gt, method_name_gt.upper()+student+'_gt', gt_code_i, output_type_gt)

            # Might need to change during full submission case, since same function will be submitted more than once for same student
            save_code(group_i, full_part_gt, method_name_gt.upper()+student+'_gt')

            # output_i_gt = execute_and_store(group_i, method_name_gt.upper()+student+'_gt')
            output_i_gt = execute_and_store_update(group_i, method_name_gt.upper()+student+'_gt')

            if 'compilation failed' in output_i_gt or output_i_gt.upper() == 'TIMEOUT':
                print(output_i_gt)
                student_ground_truth_i = []
            else:
                student_ground_truth_i = output_i_gt.split('\n')

            accuracy_gt, bi_acc_ls_gt = get_score(solution_i, student_ground_truth_i[:-1])
            
            # Used for final F1 and accuracy
            if len(bi_acc_ls_gt) == len(bi_acc_ls):
                final_gt_ls.append(bi_acc_ls_gt)
                final_compiled_ls.append(bi_acc_ls)
            else:
                set_trace()
                print(bi_acc_ls_gt)

            # print('Accuracy on Ground Truth {:s}: {:.2f}%'.format(group_i, accuracy_gt*100))
            # print('--------------')

            if eval:
                # Compile and Dosn't Compile Case
                if len(student_ground_truth_i[:-1]) > 0 and len(student_answer_i[:-1]) > 0:
                    mse_all_compile += (accuracy_gt - accuracy)**2
                    granular_error_compile += get_granular_correctness(solution_i, student_ground_truth_i[:-1], student_answer_i[:-1])
                    all_compile_cnt += 1
                
                elif len(student_ground_truth_i[:-1]) == 0 and len(student_answer_i[:-1]) > 0:
                    mse_gen_compile += (accuracy_gt - accuracy)**2
                    granular_error_gen_compile += get_granular_correctness(solution_i, student_ground_truth_i[:-1], student_answer_i[:-1])
                    gen_compile_cnt += 1
                
                elif len(student_ground_truth_i[:-1]) > 0 and len(student_answer_i[:-1]) == 0:
                    mse_gt_compile += (accuracy_gt - accuracy)**2
                    granular_error_gt_compile += get_granular_correctness(solution_i, student_ground_truth_i[:-1], student_answer_i[:-1])
                    gt_compile_cnt += 1
                
                elif len(student_ground_truth_i[:-1]) == 0 and len(student_answer_i[:-1]) == 0:
                    mse_no_compile += (accuracy_gt - accuracy)**2
                    granular_error_no_compile += get_granular_correctness(solution_i, student_ground_truth_i[:-1], student_answer_i[:-1])
                    no_compile_cnt += 1

                ## Group Cases Calculation
                if len(student_ground_truth_i[:-1]) > 0 and accuracy_gt == 1:
                    gt_status = 2
                elif len(student_ground_truth_i[:-1]) > 0:
                    gt_status = 1
                else:
                    gt_error_ls.append((group_i, method_name_gt.upper()+student+'_gt'))
                
                if gt_status == 2 and gen_status == 0:
                    both_pass_debug_ls.append((group_i, method_name_gt.upper()+student))

                if method_name_gt != method_name:
                    name_mismatch_cnt += 1
                
                
                mse = (accuracy_gt - accuracy)**2
                mse_overall += mse
                group_cases_dict[gt_status][gen_status]['MSE'] += mse
                group_cases_dict[gt_status][gen_status]['cnt'] += 1
                granular_error = get_granular_correctness(solution_i, student_ground_truth_i[:-1], student_answer_i[:-1])
                granular_correctness_overall += granular_error
                group_cases_dict[gt_status][gen_status]['granular'] += granular_error
                if metric == 'MSE':
                    group_cases_dict[gt_status][gen_status]['stats_ls'].append((mse, student, group_i))
                else:
                    group_cases_dict[gt_status][gen_status]['stats_ls'].append((granular_error, student, group_i))
                
                student_code_pair[student]['score'].append(accuracy)
                student_code_pair[student]['answer'].append(student_answer_i[:-1])
                student_code_pair[student]['ground_truth_result'].append(student_ground_truth_i[:-1])
                student_code_pair[student]['binary_correctness'].append(bi_acc_ls)

            else:
                student_code_pair[student]['binary_correctness'].append(bi_acc_ls_gt)
            
            total_cnt += 1

    if eval:
        final_compiled = [pred for pred_ls in final_compiled_ls for pred in pred_ls]
        final_gt = [gt for gt_ls in final_gt_ls for gt in gt_ls]

        f1 = f1_score(final_gt, final_compiled)
        print('F1:', f1)

        acc = accuracy = sum([1 for true, pred in zip(final_gt, final_compiled) if true == pred]) / len(final_gt)
        print('Accuracy:', acc)

        set_trace()

        mse_overall /= total_cnt
        print('MSE overall:', mse_overall)

        granular_correctness_overall /= total_cnt
        print('Granular Correctness Score Overall:', granular_correctness_overall)
        print('--------------')


        return student_code_pair, generated_error_ls, gt_error_ls
    
    else:
        print('No. of score mismatch case for GranularDKT')
        return student_code_pair

def find_code_by_cnt(student_code_pair, target_cnt):
    cnt = 0
    student_code_temp = defaultdict(nested_dict)
    for student in student_code_pair.keys():
        question_code_pair = student_code_pair[student]
        for i in range(len(question_code_pair['prompts'])):
            cnt += 1

            if cnt > target_cnt - 3 and cnt < target_cnt + 3:
                student_code_temp[student]['prompts'].append(question_code_pair['prompts'][i])
                student_code_temp[student]['ground_truth_codes'].append(question_code_pair['ground_truth_codes'][i])

    return student_code_temp


def generated_code_error_analyze(gen_error_ls):
    error_ls = []
    error_cnt = 0

    incomp_ls = []

    print('Total Number of Error: ', len(gen_error_ls))
    compile_error_cnt = 0
    runtime_error_cnt = 0

    # Used to check parameter mismatch generation amount
    param_cnt = 0
    cannot_apply_cnt = 0

    for error_pair in gen_error_ls:
        qid, student = error_pair
        error_i = execute_and_store_error_subprocess(qid, student)
        included = False
      
        for i in error_i[0]:
            error_ls.append(i)
            if 'cannot be applied to given types;' in i:
                incomp_ls.append(error_i[0])
               
                cannot_apply_cnt += 1
                if not included:
                    param_cnt += 1
                    included = True

        if error_i[1] == 0:
            compile_error_cnt += 1
        else:
            runtime_error_cnt += 1
        error_cnt += error_i[2]

    for inc in incomp_ls:
        print(set(inc))
        print('----')

    print('No. of Compile Error: ', compile_error_cnt)
    print('No. of Runtime Error: ', runtime_error_cnt)

    print("Error total:", error_cnt)
    print('Error total without mismatch:', error_cnt - cannot_apply_cnt)
    print('Average errors on generated error code:', error_cnt/len(gen_error_ls))
    print('Parameter mismatch error cnt:', param_cnt)
    print('Average errors on generated error code without mismatch:', (error_cnt - cannot_apply_cnt)/len(gen_error_ls))

    error_dict = Counter(error_ls)
    plt.figure(figsize=(10, 8))
    plt.bar(list(error_dict.keys()), error_dict.values())
    plt.xticks(rotation=-45)
    plt.tight_layout()
    # plt.savefig('gt_error_histogram.png')
    print(error_dict)
    return error_dict

def main():
    now = '20240921_043740'

    all_good_test, good_test_case = test_case_check()

    student_code_pair, desc_id_pair = validate_prompt_test_set(configs, all_good_test, now=now)

    student_code_pair = get_test_result_for_generated_code(good_test_case, student_code_pair, desc_id_pair, metric='granular', eval=True)
    with open(os.path.join(configs.model_save_dir, now, 'eval_logs.pkl'), 'wb') as f:
        pickle.dump(student_code_pair, f)


if __name__ == "__main__":
    main()