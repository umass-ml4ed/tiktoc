import pandas as pd
from tqdm import tqdm
import torch
from sklearn.model_selection import train_test_split
import pickle
from data_preprocessing import *
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from pdb import set_trace

def nested_dict():
    return defaultdict(list)    


def preprocess_valid_testcase_data(configs, pickle_file):
    val_data = pd.read_pickle(configs.data_path + pickle_file)

    gt_code_pairs = {}

    gt_code_pairs['ground_truth_codes'] = val_data['Code'].tolist()
    gt_code_pairs['prompts'] = val_data['prompt'].tolist()
    gt_code_pairs['students'] = val_data['SubjectID'].tolist()
    gt_code_pairs['scores'] = val_data['Score'].tolist()

    with open(os.path.join(configs.data_path, 'val_testcase_pairs.pkl'), 'wb') as f:
        pickle.dump(gt_code_pairs, f)

    return gt_code_pairs

# Construct the dataset with additional column of binary level label
# dataset_granular_1st.pkl is the dataset used in granularDKT, multitask model, and okt_testcase
def convert_bi_corr_dataset(configs, file_path, orig_df):
    with open(os.path.join(configs.data_path, file_path), 'rb') as f:
        data = pickle.load(f)
        final_ls = []
        for student, info in data.items():
            codes = info['ground_truth_codes']
            bi_scores = info['binary_correctness']
            # match_ls = info['score_match']

            # for code, bi_score, match in zip(codes, bi_scores, match_ls):
                # final_ls.append({'SubjectID': student, 'Code': code, 'binary_score': bi_score, 'score_match': match})
            for code, bi_score in zip(codes, bi_scores):
                final_ls.append({'SubjectID': student, 'Code': code, 'binary_score': bi_score})
            
        final_df = pd.DataFrame(final_ls)

    combined_df = pd.merge(orig_df, final_df, on=['SubjectID', 'Code'], how='left')
    combined_df.to_pickle(configs.data_path + '/dataset_granular_1st.pkl')

def read_data(configs, tokenizer, model, device):
    '''
    @param configs.label_type: whether to use binarized label, raw label, or ternery label
    @param configs.max_len: maximum allowed length for each student's answer sequence. longer
                    than this number will be truncated and set as new student(s)
    @param configs.seed: reproducibility
    '''
    ## load dataset

    if configs.okt_model == 'student' or configs.okt_model == 'funcom' or configs.okt_model == 'gpt-2':
        dataset = pd.read_pickle(configs.data_path + '/dataset_time.pkl')
    else:
        dataset, sat_questions = construct_dataset(configs, tokenizer, model, device)
    print('Dataset constructed')

    ## if only testing, subsample part of dataset
    if configs.testing:
        dataset = dataset.sample(n=120)
        # Sort sampled dataset for timestep columm creation logic to work below
        dataset = dataset.sort_values(by=["SubjectID", "AssignmentID", "ProblemID"])

    # choose label format
    if configs.label_type == 'binary':
        scores_y = []
        for item in dataset['Score_y']:
            if item >= 2:
                scores_y.append(1)
            else:
                scores_y.append(0)
        dataset['Score'] = scores_y
    elif configs.label_type == 'ternery':
        dataset['Score'] = dataset['Score_y']
    elif configs.label_type == 'raw':
        dataset['Score'] = dataset['Score_x']
    dataset = dataset.drop(columns=['Score_x','Score_y'])
    
    ## optionally keep only the first answer by the student
    if configs.first_ast_convertible:
        ('only using first ast-convertible code')
        dataset = dataset.drop_duplicates(
                        subset = ['SubjectID', 'ProblemID'],
                        keep = 'first').reset_index(drop = True)

    # # Filtered out question not in valid test case questions
    # # Run only once to save the pickle file used to label binary correctness
    # sat_questions_word = sat_questions.keys()
    # mask = dataset['prompt'].isin(sat_questions_word)
    # dataset = dataset[mask]

    ## split a student's record into multiples 
    ## if it exceeds configs.max_len, change the subject ID to next one
    prev_subject_id = 0
    subjectid_appendix = []
    timesteps = []
    for i in tqdm(range(len(dataset)), desc="splitting students' records ..."):
        if prev_subject_id != dataset.iloc[i].SubjectID:
            # when encountering a new student ID
            prev_subject_id = dataset.iloc[i].SubjectID
            accumulated = 0
            id_appendix = 1
        else:
            accumulated += 1
            if accumulated >= configs.max_len:
                id_appendix += 1
                accumulated = 0
        timesteps.append(accumulated)
        subjectid_appendix.append(id_appendix)
    dataset['timestep'] = timesteps
    dataset['SubjectID_appendix'] = subjectid_appendix
    dataset['SubjectID'] = [dataset.iloc[i].SubjectID + \
                '_{}'.format(dataset.iloc[i].SubjectID_appendix) for i in range(len(dataset))]


    # dataset.to_pickle(configs.data_path + '/dataset_testcase.pkl')
    
    # preprocess_valid_testcase_data(configs, '/dataset_testcase.pkl')
    # convert_bi_corr_dataset(configs, 'student_pair_1st.pkl', dataset)

    # preprocess_all_submission(configs, dataset)

    ## Each subject ID implies a student
    students = dataset['SubjectID'].unique()

    # Train, val, test split 
    if configs.split_by == 'student':
        train_dkt, test_dkt = train_test_split(students, test_size=configs.test_size, random_state=configs.seed)
        valid_dkt, test_dkt = train_test_split(test_dkt, test_size=0.5, random_state=configs.seed)
        return train_dkt, valid_dkt, test_dkt, dataset, students, None


    # Keep copy of dataset with timestep=0 for creating LSTM input dataset since we require (p_0, c_0) to compute h_0 used to predict c_1
    dropped_dataset = dataset.copy()
    # Drop entries with timestep=0 since we don't have student history (p_i, c_i) to compute student knowledge state to predict c_0 for p_0
    dropped_dataset = dropped_dataset.drop(dropped_dataset.index[dropped_dataset['timestep'] == 0]).reset_index(drop = True)


    # Split on entries instead of on students
    trainset, testset = train_test_split(dropped_dataset, test_size=configs.test_size, random_state=configs.seed)
    validset, testset = train_test_split(testset, test_size=0.5, random_state=configs.seed)

    # For OKT model
    if configs.exp_name == 'okt':
        return trainset, validset, testset, dataset, students
    # For codeDKT model
    else:
        if configs.okt_model != 'gpt-2':
            return trainset, validset, testset, dataset, students, sat_questions
        return trainset, validset, testset, dataset, students, None

# read_granular_data split the dataset contains granular correctness for each submission based on student, which
# follows the standard way to split the dataset
def read_granular_data(configs):
    if configs.first_ast_convertible:
        if configs.okt_model == 'codellama/CodeLlama-7b-Instruct-hf' or configs.okt_model == 'meta-llama/Meta-Llama-3-8B-Instruct' or configs.okt_model == 'Qwen/Qwen1.5-7B':
            dataset = pd.read_pickle(configs.data_path + '/dataset_granular_1st.pkl')
        else:
            dataset = pd.read_pickle(configs.data_path + '/dataset_testcase_1st_gpt2.pkl')
    else:
        dataset = pd.read_pickle(configs.data_path + '/dataset_granular_all.pkl')

    students = dataset['SubjectID'].unique()

    trainset, testset = train_test_split(students, test_size=configs.test_size, random_state=configs.seed)
    validset, testset = train_test_split(testset, test_size=0.5, random_state=configs.seed)

    return trainset, validset, testset, dataset, students

# Adding score to the dataset for DKT baseline
def make_pytorch_dataset(dataset_split, dataset_full, do_lstm_dataset=True, split_by_student=False, granular=False, okt_model=False):
    '''
    convert the pandas dataframe into dataset format that pytorch dataloader takes
    the resulting format is a list of dictionaries
    '''
    # dictionary, key=student id, value=list of lstm inputs at each time step
    if do_lstm_dataset:
        if split_by_student:
            students = dataset_split
            lstm_student = []
            for student in students:
                subset = dataset_full[dataset_full.SubjectID == student]
                subset.loc[:, 'prompt-embedding'] = subset['prompt-embedding'].apply(lambda x: torch.tensor(x))
                data_dict = {
                    'SubjectID': student,
                    'ProblemID_seq': subset.ProblemID.tolist(),
                    'Score': subset.Score.tolist(),
                    'prompt-embedding': subset['prompt-embedding'].tolist(),
                    'input': subset.input.tolist(),
                }
                if granular:
                    data_dict['granular_correctness'] = subset['binary_score'].tolist()
                
                if okt_model:
                    data_dict['next_prompt'] = subset.prompt.tolist()
                    data_dict['next_code'] = subset.Code.tolist()
                lstm_student.append(data_dict)
            return lstm_student

        else:
            students = dataset_full['SubjectID'].unique()
            lstm_dataset = {}
            for student in students:
                lstm_dataset[student]=dataset_full[dataset_full.SubjectID==student].input.tolist()
            del dataset_full


    okt_dataset = []
    students = dataset_split['SubjectID'].unique()

    for student in students:
        subset = dataset_split[dataset_split.SubjectID==student]
        for t in range(len(subset)):
            # Set step = timestep-1 for alignment with LSTM input dataset [(p_i, c_i)] to ensure h_t computed using [(p_0, c_0), ..., (p_t, c_t)] is used to predict c_{t+1}
            data_dict = {
                'SubjectID': student,
                'ProblemID': subset.iloc[t].ProblemID,
                'step': subset.iloc[t].timestep-1, 
                'next_Score': subset.iloc[t].Score,
                'next_prompt': subset.iloc[t].prompt,
                'next_code': subset.iloc[t].Code,
                'prompt_embedding': subset.iloc[t]['prompt-embedding']
            }
            okt_dataset.append(data_dict)
    del dataset_split
    
    if do_lstm_dataset:
        return okt_dataset, lstm_dataset
    else:
        return okt_dataset


# If split_by_student = true, make_dataloader is used for codeDKT, granularDKT, multitask model.
def make_dataloader(dataset_split, dataset_full, collate_fn, configs, n_workers=0, do_lstm_dataset=True, train=True, split_by_student=False, granular=False, okt_model=False):
    # Make two datasets: one with a list of dict (for GPT), and another a dict with student_id as key (for LSTM to compute knowledge states)
    shuffle = True if train else False
    if do_lstm_dataset:
        if split_by_student:
            lstm_student = make_pytorch_dataset(dataset_split, dataset_full, do_lstm_dataset, split_by_student, granular=granular, okt_model=okt_model)
            data_loader = torch.utils.data.DataLoader(lstm_student, collate_fn=collate_fn, shuffle=shuffle, batch_size=configs.batch_size, num_workers=n_workers)
            return lstm_student, data_loader

        else:
            okt_dataset, lstm_dataset = make_pytorch_dataset(dataset_split, dataset_full, do_lstm_dataset)
            data_loader = torch.utils.data.DataLoader(
                okt_dataset, collate_fn=collate_fn, shuffle=shuffle, batch_size=configs.batch_size, num_workers=n_workers)   
            return okt_dataset, data_loader, lstm_dataset
    else:
        okt_dataset = make_pytorch_dataset(dataset_split, dataset_full, do_lstm_dataset)
        data_loader = torch.utils.data.DataLoader(
            okt_dataset, collate_fn=collate_fn, shuffle=shuffle, batch_size=configs.batch_size, num_workers=n_workers)   
        return okt_dataset, data_loader

def get_lstm_inputs(configs, train_set, dataset, collate_fn):
    
    _, _, lstm_inputs = make_dataloader(train_set, dataset, 
                                                   collate_fn=collate_fn, 
                                                   configs=configs, do_lstm_dataset=True)
    
    return lstm_inputs


def construct_okt_dataset_from_granular(configs):
    if configs.first_ast_convertible:
        if configs.okt_model == 'codellama/CodeLlama-7b-Instruct-hf' or configs.okt_model == 'meta-llama/Meta-Llama-3-8B-Instruct' or configs.okt_model == 'Qwen/Qwen1.5-7B':
            dataset = pd.read_pickle(configs.data_path + '/dataset_granular_1st.pkl')
        else:
            dataset = pd.read_pickle(configs.data_path + '/dataset_testcase_1st_gpt2.pkl')
    else:
        dataset = []

    students = dataset['SubjectID'].unique()

    train_student_set, test_student_set = train_test_split(students, test_size=configs.test_size, random_state=configs.seed)
    valid_student_set, test_student_set = train_test_split(test_student_set, test_size=0.5, random_state=configs.seed)

    train_dataset = dataset[dataset['SubjectID'].isin(train_student_set)]
    dropped_train_set = train_dataset.drop(train_dataset.index[train_dataset['timestep'] == 0]).reset_index(drop=True)

    valid_dataset = dataset[dataset['SubjectID'].isin(valid_student_set)]
    dropped_valid_set = valid_dataset.drop(valid_dataset.index[valid_dataset['timestep'] == 0]).reset_index(drop=True)

    test_dataset = dataset[dataset['SubjectID'].isin(test_student_set)]
    dropped_test_set = test_dataset.drop(test_dataset.index[test_dataset['timestep'] == 0]).reset_index(drop=True)

    return dropped_train_set, dropped_valid_set, dropped_test_set, dataset

# Return a dict contains question id to corresponding test case solution
def get_test_case_solution():
    prompt_df = pd.read_csv(os.path.join('test-case-query-results/prompt_concept_summary.csv'), on_bad_lines='warn')
    all_good = prompt_df.loc[prompt_df['Test Case Status'] == 'All Good']

    test_info_df = pd.read_csv(os.path.join('test-case-query-results/test_cases-1-26-24.csv'), on_bad_lines='skip')
    coding_prompt_id = set(test_info_df['coding_prompt_id'].unique())
    ls = [29, 37, 106, 236, 239, 240]
    for i in ls:
        coding_prompt_id.add(i)
    coding_prompt_id = {int(item) for item in coding_prompt_id if not (isinstance(item, float) and np.isnan(item))}

    sat_questions = dict(zip(all_good['ProblemID'], all_good['Requirement']))
    sat_id = sat_questions.keys()
    good_test_case = test_info_df[test_info_df['coding_prompt_id'].isin(sat_id)]

    question_input_dict = uniq_test_construct(good_test_case)
    question_input_dict = handle_uniq_test_exception(question_input_dict)

    solutions = good_test_case.groupby('coding_prompt_id')
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

    return solution_dict, question_input_dict

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
    question_input_dict[37] = cleaned_37
    
    return question_input_dict


def map_test_case_to_dataset(solution_dict, question_input_dict, configs):
    if configs.first_ast_convertible:
        if configs.okt_model == 'codellama/CodeLlama-7b-Instruct-hf' or configs.okt_model == 'meta-llama/Meta-Llama-3-8B-Instruct' or configs.okt_model == 'Qwen/Qwen1.5-7B':
            dataset = pd.read_pickle(configs.data_path + '/dataset_granular_1st.pkl')
        else:
            dataset = pd.read_pickle(configs.data_path + '/dataset_testcase_1st_gpt2.pkl')
    else:
        dataset = []

    students = dataset['SubjectID'].unique()
    lstm_dataset = {}
    for student in students:
        lstm_dataset[student]=dataset[dataset.SubjectID==student].input.tolist()

    dataset['test_inputs'] = dataset['ProblemID'].map(question_input_dict)
    dataset['test_solutions'] = dataset['ProblemID'].map(solution_dict)


    dataset_testcase = dataset.explode(['binary_score', 'test_inputs', 'test_solutions'])
    dataset_testcase = dataset_testcase.drop(dataset_testcase.index[dataset_testcase['timestep'] == 0]).reset_index(drop = True)

    trainstudent, teststudent = train_test_split(students, test_size=configs.test_size, random_state=configs.seed)
    validstudent, teststudent = train_test_split(teststudent, test_size=0.5, random_state=configs.seed)

    trainset = dataset_testcase[dataset_testcase['SubjectID'].isin(trainstudent)]

    validset = dataset_testcase[dataset_testcase['SubjectID'].isin(validstudent)]
    testset = dataset_testcase[dataset_testcase['SubjectID'].isin(teststudent)]

    return trainset, validset, testset, dataset, lstm_dataset

def make_pytorch_testcase_dataset(dataset):
    subset = dataset[['SubjectID', 'ProblemID', 'Score', 'prompt', 'Code', 'timestep', 
                      'test_inputs', 'test_solutions', 'binary_score']]
    
    subset.rename(columns={'Score': 'next_Score', 'prompt':'next_prompt', 'Code': 'next_code', 'test_inputs': 'next_test_input',
                           'test_solutions': 'next_test_solution', 'binary_score': 'next_binary_score'}, inplace=True)
    
    okt_dataset = subset.to_dict(orient='records')
    
    return okt_dataset

def make_testcase_dataloader(dataset, collate_fn, configs, n_workers=0):
    okt_dataset = make_pytorch_testcase_dataset(dataset)

    data_loader = torch.utils.data.DataLoader(
        okt_dataset, collate_fn=collate_fn, batch_size=configs.batch_size, num_workers=n_workers)   
    return okt_dataset, data_loader

# GPT-2 tokenizer handle add_special_tokens differently. When using Llama-3, BOS_token will be added automatically, still need to add EOS_token manually
def build_input_with_special_tokens(prompt, code, tokenizer, configs, ins_ft=None):
    # Match GPT2 pretraining input style: https://github.com/huggingface/transformers/issues/3311
    # https://huggingface.co/patrickvonplaten/bert2gpt2-cnn_dailymail-fp16#training-script
    # Input format for GPT-2: <|endoftext|>question: <question> student written code: <code><|endoftext|>
    # Input format for CodeLlama: question: <question> student written code: <code></s>
    # Start completion (student code) with whitespace
    input = build_prompt_with_special_tokens(prompt, tokenizer, configs, ins_ft=ins_ft) + " " + code.strip() + tokenizer.eos_token

    return input


def build_prompt_with_special_tokens(prompt, tokenizer, configs, ins_ft=None):
    # Remove delimiter : in prompt since we use it to calculate prompt length
    if( ":" in prompt ):
        prompt = prompt.replace(":", "")
    # Phrase "student written code:" should serve as our separator between prompt and completion
    assert "student written code" not in prompt
    if configs.okt_model != 'codellama/CodeLlama-7b-Instruct-hf' and configs.okt_model != 'meta-llama/Meta-Llama-3-8B-Instruct' and configs.okt_model != 'Qwen/Qwen1.5-7B':
        prompt = tokenizer.bos_token + "Question: " + prompt + " Student written code:"
    else:
        prompt = "Question: " + prompt + " Student written code:"
    
    if ins_ft:
        prompt = build_instruction_prefix(ins_ft) + prompt

    return prompt


def build_prompt_for_bianry_res(prompt, test_input, test_output, ins_ft=None):
    if( ":" in prompt ):
        prompt = prompt.replace(":", "")
    # Phrase "student written code:" should serve as our separator between prompt and completion
    assert "student written code" not in prompt
    
    prompt = "Question: " + prompt + "\n" + "test case input: " + test_input + "\n" + "expected output: " + test_output + "\n" + "Student result:"
    if ins_ft:
        prompt = build_instruction_prefix(ins_ft) + prompt

    return prompt

def build_input_for_binary_res(prompt, test_input, test_output, res, tokenizer, ins_ft=None):
    result = 'pass' if res else 'fail'
    input = build_prompt_for_bianry_res(prompt, test_input, test_output, ins_ft=ins_ft) + " " + result + tokenizer.eos_token
   
    return input

def build_instruction_prefix(task):
    instruction = 'Test case result prediction task: ' if task == 'tc' else 'Code generation task: '
    common = 'You are simulating a student learning to program in Java. '
    task = 'Given an input problem and associated test case, predict whether the code written by the student will pass or fail the test case. ' if task == 'tc' else 'Given an input problem, predict the code written by the student. '

    final_input = instruction + common + task
    return final_input



# Use only when okt_model belongs 'llama-3'
def find_max_token_length(tokenizer, inputs_ids, attention_mask, prompt_id_lens):
    padding_ids = torch.where(inputs_ids == tokenizer.convert_tokens_to_ids(tokenizer.eos_token), 1, 0)
    masked_padding_ids = padding_ids.masked_fill((attention_mask == 0), 0)
    prompt_total_lens = torch.argmax(masked_padding_ids, dim=-1)

    label_length = prompt_total_lens - prompt_id_lens
    label_length = torch.add(label_length, 1)
    max_label_length = torch.max(label_length)
    return max_label_length.item()


class CollateForOKT(object):
    def __init__(self, tokenizer, configs, device):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "right"
        self.max_length_label = 0

        # Pad if required with <|endoftext|> tokens on the right of input since GPT2 uses absolute position embeddings
        # assert self.tokenizer.padding_side == "right"
        self.configs = configs
        self.device = device
        # Token id 25 corresponds to ":" in vocab https://huggingface.co/gpt2/raw/main/vocab.json
        # self.delimiter_token_id = 25
        self.delimiter_token_id = tokenizer.convert_tokens_to_ids(":")

    def __call__(self, batch):
        inputs_text = [build_input_with_special_tokens(b['next_prompt'], b['next_code'], self.tokenizer, self.configs) for b in batch]
        inputs = self.tokenizer(inputs_text, return_tensors='pt', padding=True, truncation=True)
        inputs_ids, attention_mask = inputs['input_ids'].to(self.device), inputs['attention_mask'].to(self.device)

        # Handle truncation: Replace last token id with tokenizer.eos_token_id to ensure generation ends with eos_token_id
        inputs_ids[:, -1] = self.tokenizer.eos_token_id

        # Find prompt length which is needed to linearly combine student knowledge state with prompt tokens only
        # To find prompt length we find the second occurence of delimiter ":" in <|endoftext|>question: <question> student written code: <code><|endoftext|>
        delimiter_indices = torch.where(inputs_ids == self.delimiter_token_id, 1, 0)
        # Ignore first occurence of delimiter at index 2 since our prompt always starts with bos_token when model is not Qwen1.5-7B:
        if self.configs.okt_model == 'Qwen/Qwen1.5-7B':
            delimiter_indices[:, 1] = 0
        else:
            delimiter_indices[:, 2] = 0
        # Argmax returns first occurence of maximum value. Here the first occurence of maximum value will be the second occurence of delimiter (we ignored the first occurence)
        prompt_id_lens = torch.argmax(delimiter_indices, dim=-1)
        # Add 1 since length = zero-based index + 1
        prompt_id_lens = torch.add(prompt_id_lens, 1)

        # Compute labels
        labels = inputs_ids.detach().clone()
        # Ignore padding
        labels = labels.masked_fill((attention_mask == 0), -100)
        # Use only code tokens, ignore prompt tokens
        range_tensor = torch.arange(inputs_ids.size(1), device=self.device).unsqueeze(0)
        range_tensor = range_tensor.repeat(prompt_id_lens.size(0), 1)
        mask_tensor = (range_tensor < prompt_id_lens.unsqueeze(-1)) 
        labels[mask_tensor] = -100

        self.max_length_label = find_max_token_length(self.tokenizer, inputs_ids, attention_mask, prompt_id_lens)
        
        
        students = [b['SubjectID'] for b in batch]
        timesteps = [b['step'] for b in batch]

        """
        # Print sample batch
        print("Sample batch:")
        for ids in inputs_ids:
            print(self.tokenizer.decode(ids))
        print("Input ids:", inputs_ids)
        print("Attention mask:", attention_mask)
        print("Labels:", labels)
        """

        return inputs_ids, attention_mask, labels, prompt_id_lens, students, timesteps

# collate for okt and split by student
class CollateForOKTstudent(object):
    def __init__(self, tokenizer, configs, device, eval=False, question_test_dict=None, question_no_map=None):
        self.tokenizer = tokenizer
        self.tokenizer.padding_side = "left" if eval else "right"
        self.max_length_label = 0

        # assert self.tokenizer.padding_side == "right"
        self.configs = configs
        self.device = device
        self.delimiter_token_id = tokenizer.convert_tokens_to_ids(":")
        self.eval = eval

        if configs.multitask_label == 'granular':
            self.test_case_dict = question_test_dict
            self.question_no_map = question_no_map
            self.T_max = max([len(i) for i in self.test_case_dict.values()])
    
    def __call__(self, batch):
        if self.configs.multitask_label != 'granular':
            scores = [b['Score'] for b in batch]
            max_len = max([len(i) for i in scores])
            padded_scores = [i + [-100] * (max_len - len(i)) for i in scores]
            padded_scores = torch.Tensor(padded_scores).t().to(self.device) #shape: (T, B)
            
        else:
            scores = [b['granular_correctness'] for b in batch]
            max_len = max([len(i) for i in scores])
            padded_scores = [[test_case + [-100]* (self.T_max - len(test_case)) for test_case in ite] for ite in scores]
            padded_scores = [i + [[-100]*self.T_max for _ in range(max_len - len(i))] for i in padded_scores]
            padded_scores = torch.Tensor(padded_scores).float() # padded_granular_scores shape: (B, T, T_max)
            padded_scores = torch.transpose(padded_scores, 0, 1) # transposed padded_granular_cor shape: (T, B, T_max)

        question_seqs = [b['ProblemID_seq'] for b in batch]
        question_seqs = [[self.question_no_map[i] for i in seqs] for seqs in question_seqs]
        padded_question_seqs = [i + [0]*(max_len - len(i)) for i in question_seqs]
        padded_question_seqs = torch.tensor(padded_question_seqs).t()  #shape: (T, B)

        inputs = [b['input'] for b in batch]
        padded_inputs = [i + [torch.zeros(i[0].shape[0])]*(max_len - len(i)) for i in inputs]
        padded_inputs = torch.stack([torch.stack(x, dim=0) for x in padded_inputs], dim=1).float().to(self.device) #shape: (T, B, D)
        
        codes = [b['next_code'] for b in batch]

        students = []
        for i in range(len(batch)):
            stu_name = batch[i]['SubjectID']
            student_ls = [stu_name] * len(codes[i])
            students.append(student_ls)
        
        padded_students = [i + [''] * (max_len - len(i)) for i in students]
        stacked_students = list(map(list, zip(*padded_students)))

        padded_codes = [i + [''] * (max_len - len(i)) for i in codes]
        stacked_codes = list(map(list, zip(*padded_codes)))

        prompts = [b['next_prompt'] for b in batch]
        padded_prompts = [i + [''] * (max_len - len(i)) for i in prompts]
        stacked_prompts = list(map(list, zip(*padded_prompts)))

        if self.eval:
            # Remove the <|eot_id|> at end in each template prompt for inference
            if self.configs.use_template:
                input_texts = [[self.tokenizer.apply_chat_template([
                {"role": "system", "content": "You are simulating a student learning to program in Java. Given an input problem, predict the code written by the student."},
                {"role": "user", "content": build_prompt_with_special_tokens(prompt_i, self.tokenizer, self.configs)},
                {"role": "assistant", "content": ''}
                ], tokenize=False)[:-10] for prompt_i in entry['next_prompt']] for entry in batch]
            else:
                input_texts = [[build_prompt_with_special_tokens(prompt_i, self.tokenizer, self.configs) for prompt_i in entry['next_prompt']] for entry in batch]
        else:
            if self.configs.use_template:
                input_texts = [[self.tokenizer.apply_chat_template([
                {"role": "system", "content": "You are simulating a student learning to program in Java. Given an input problem, predict the code written by the student."},
                {"role": "user", "content": build_prompt_with_special_tokens(prompt_i, self.tokenizer, self.configs)},
                {"role": "assistant", "content": code_i}
                ], tokenize=False) for prompt_i, code_i in zip(entry['next_prompt'], entry['next_code'])] for entry in batch]
            else:
                input_texts = [[build_input_with_special_tokens(prompt_i, code_i, self.tokenizer, self.configs) for prompt_i, code_i in zip(entry['next_prompt'], entry['next_code'])] for entry in batch]

            
        inputs_ids_ls, attention_mask_ls, labels_ls, prompt_id_lens_ls = [], [], [], []

        for input_sub in input_texts:
            # Not using tokenizer.apply_chat_template
            if not self.configs.use_template:
                inputs = self.tokenizer(input_sub, return_tensors='pt', padding=True, truncation=True)
                inputs_ids, attention_mask = inputs['input_ids'].to(self.device), inputs['attention_mask'].to(self.device)

                if not self.eval:
                    inputs_ids[:, -1] = self.tokenizer.eos_token_id

                delimiter_indices = torch.where(inputs_ids == self.delimiter_token_id, 1, 0)
                if self.configs.okt_model == 'Qwen/Qwen1.5-7B':
                    delimiter_indices[:, 1] = 0
                else:
                    delimiter_indices[:, 2] = 0
                prompt_id_lens = torch.argmax(delimiter_indices, dim=-1)
                prompt_id_lens = torch.add(prompt_id_lens, 1)

            else:
                inputs = self.tokenizer(input_sub, return_tensors='pt', padding=True, truncation=True, add_special_tokens=False)
                inputs_ids, attention_mask = inputs['input_ids'].to(self.device), inputs['attention_mask'].to(self.device)
                delimiter_indices = torch.where(inputs_ids == self.tokenizer.convert_tokens_to_ids('assistant'), 1, 0)
                prompt_id_lens = torch.argmax(delimiter_indices, dim=-1)
                prompt_id_lens = torch.add(prompt_id_lens, 3)

            labels = inputs_ids.detach().clone()
            labels = labels.masked_fill((attention_mask == 0), -100)
            range_tensor = torch.arange(inputs_ids.size(1), device=self.device).unsqueeze(0)
            range_tensor = range_tensor.repeat(prompt_id_lens.size(0), 1)
            mask_tensor = (range_tensor < prompt_id_lens.unsqueeze(-1)) 
            labels[mask_tensor] = -100

            inputs_ids_ls.append(inputs_ids)
            attention_mask_ls.append(attention_mask)
            labels_ls.append(labels)
            prompt_id_lens_ls.append(prompt_id_lens)
        
        max_length = max([sub.shape[1] for sub in inputs_ids_ls])

        padded_input_ids_ls = [torch.nn.functional.pad(input_ids, (0, max_length - input_ids.shape[1]), value=self.tokenizer.eos_token_id) for input_ids in inputs_ids_ls]
        padded_input_ids_ls = pad_sequence(padded_input_ids_ls, batch_first=True, padding_value=self.tokenizer.eos_token_id)
        padded_input_ids_ls = torch.transpose(padded_input_ids_ls, 0, 1)  # shape: (T, B, max_length)

        padded_attention_mask_ls = [torch.nn.functional.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=0) for attention_mask in attention_mask_ls]
        padded_attention_mask_ls = pad_sequence(padded_attention_mask_ls, batch_first=True, padding_value=0)
        padded_attention_mask_ls = torch.transpose(padded_attention_mask_ls, 0, 1) # shape: (T, B, max_length)

        padded_labels_ls = [torch.nn.functional.pad(labels, (0, max_length - labels.shape[1]), value=-100) for labels in labels_ls]
        padded_labels_ls = pad_sequence(padded_labels_ls, batch_first=True, padding_value=-100)
        padded_labels_ls = torch.transpose(padded_labels_ls, 0, 1) # shape: (T, B, max_length)

        padded_prompt_id_lens_ls = [torch.cat((i, torch.zeros(max_len - i.size(0)).to(self.device)), 0) for i in prompt_id_lens_ls]
        padded_prompt_id_lens_ls = torch.stack(padded_prompt_id_lens_ls).t() # shape: (T, B)

        if self.eval:
            return padded_inputs, padded_input_ids_ls, padded_attention_mask_ls, stacked_codes, stacked_prompts, padded_scores, padded_question_seqs, stacked_students
        
        return padded_scores, padded_inputs, padded_input_ids_ls, padded_attention_mask_ls, padded_labels_ls, padded_prompt_id_lens_ls, padded_question_seqs


class CollateForGranularDKT(object):
    def __init__(self, configs, question_test_dict, question_no_map):
        self.configs = configs
        self.test_case_dict = question_test_dict
        self.question_no_map = question_no_map

        self.T_max = max([len(i) for i in self.test_case_dict.values()])

    def __call__(self, batch):
        scores = [b['Score'] for b in batch]
        max_len = max([len(i) for i in scores])

        inputs = [b['input'] for b in batch]
        padded_inputs = [i + [torch.zeros(i[0].shape[0])] * (max_len - len(i)) for i in inputs]
        padded_inputs = torch.stack([torch.stack(x, dim=0) for x in padded_inputs], dim=1).float()

        test_case_cors = [b['granular_correctness'] for b in batch]
        
        padded_granular_cor = [[test_case + [-100]* (self.T_max - len(test_case)) for test_case in ite] for ite in test_case_cors]
        padded_granular_cor = [i + [[-100]*self.T_max for _ in range(max_len - len(i))] for i in padded_granular_cor]
        padded_granular_cor = torch.Tensor(padded_granular_cor).float() # padded_granular_cor shape: (B, max_timestep_len, T_max)
        padded_granular_cor = torch.transpose(padded_granular_cor, 0, 1) # transposed padded_granular_cor shape: (max_timestep_len, B, T_max)

        embeddings = [b['prompt-embedding'] for b in batch]
        padded_embeddings = [i + [torch.zeros(i[0].shape[0])]*(max_len - len(i)) for i in embeddings]
        padded_embeddings = torch.stack([torch.stack(x, dim=0) for x in padded_embeddings], dim=1).float()

        question_seqs = [b['ProblemID_seq'] for b in batch]
        question_seqs = [[self.question_no_map[i] for i in seqs] for seqs in question_seqs]
        padded_question_seqs = [i + [0]*(max_len - len(i)) for i in question_seqs]
        padded_question_seqs = torch.tensor(padded_question_seqs).t()
        return padded_granular_cor, padded_inputs, padded_embeddings, padded_question_seqs
