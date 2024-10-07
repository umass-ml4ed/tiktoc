import pandas as pd
import numpy as np
import torch
import hydra
from model import *
from pdb import set_trace
import pickle


def nested_dict():
    return defaultdict(list)  

# dataset.pkl contains GPT-2 question representation embedding only
# Saved dataset_wo_emb.pkl to create embedding based on LLM used in future work.
def clean_dataset(configs):
    dataset = pd.read_pickle(configs.data_path + '/dataset.pkl')
    dataset = dataset.drop(columns=['prompt-embedding', 'input'])
    dataset.to_pickle(configs.data_path + '/dataset_wo_emb.pkl')


# There are 50 unique programming questions.
# Returns a dictionary (key: question prompt, value: question representation embedding)
def get_questions_prompt(configs, tokenizer, model, device):
    dataset = pd.read_pickle(configs.data_path + '/dataset.pkl')
    questions = list(dataset['prompt'].unique())
    
    question_embed_dict = {}

    for q_i in questions:
        prompt_embedding = tokenizer(q_i, return_tensors='pt')

        if configs.okt_model != 'codellama/CodeLlama-7b-Instruct-hf' and configs.okt_model != 'meta-llama/Meta-Llama-3-8B-Instruct' and configs.okt_model != 'Qwen/Qwen1.5-7B':
            token_embedding = model.transformer.wte(prompt_embedding['input_ids'].to(device))
        else:
            token_embedding = model.base_model.model.model.embed_tokens(prompt_embedding['input_ids'].to(device))
        
        avg_token_embedding = torch.mean(token_embedding[0], axis=0)
        question_embed_dict[q_i] = avg_token_embedding.cpu().detach().numpy()

    return question_embed_dict

def get_full_test_question_emb(question_embed_dict):
    prompt_df = pd.read_csv(os.path.join('test-case-query-results/prompt_concept_summary.csv'), on_bad_lines='warn')
    all_good = prompt_df.loc[prompt_df['Test Case Status'] == 'All Good']
    sat_questions = all_good['Requirement'].tolist()

    sat_questions_dict = {}
    for question in sat_questions:
        sat_questions_dict[question] = question_embed_dict[question].tolist()
    
    return sat_questions_dict


# Construct the input used in knowledge representation model
# Returns the dataset with input embedding, dimension should be 968(768+200) or 4296(4096+200)
def construct_dataset(configs, tokenizer, model, device):
    question_embed_dict = get_questions_prompt(configs, tokenizer, model, device)
    dataset = pd.read_pickle(configs.data_path + '/dataset_wo_emb_time.pkl')

    dataset['prompt-embedding'] = dataset['prompt'].map(question_embed_dict)
    dataset['input'] = dataset[['prompt-embedding', 'embedding']].apply(lambda x: np.concatenate((x[0], x[1])), axis=1)
    dataset['input'] = dataset['input'].apply(lambda x: torch.from_numpy(x))

    sat_questions = get_full_test_question_emb(question_embed_dict)

    return dataset, sat_questions

def handle_score_mismatch(configs):
    gt_code_ls = []
    student_ls = []
    prompt_ls = []
    score_ls = []
    bi_score_ls = []

    for i in range(7):
        cnt = 0
        filename = '/student_pair_all_' + str(i) + '.pkl'
        dataset = pd.read_pickle(configs.data_path + filename)

        for student, info in dataset.items():
            codes = info['ground_truth_codes']
            bi_scores = info['binary_correctness']
            match_ls = info['score_match']
            prompts = info['prompts']
            scores = info['scores']

            for code, bi_score, match, prompt, score in zip(codes, bi_scores, match_ls, prompts, scores):
                    if not match:
                        cnt += 1
                        gt_code_ls.append(code)
                        student_ls.append(student)
                        prompt_ls.append(prompt)
                        score_ls.append(score)
                        bi_score_ls.append(bi_score)
        
        print('mismatch count:', cnt)
    
    final_dict = {'ground_truth_codes': gt_code_ls, 'prompts': prompt_ls, 'students': student_ls, 'scores': score_ls, 'binary_correctness': bi_score_ls}

    with open(os.path.join(configs.data_path, 'mismatch_subset.pkl'), 'wb') as f:
            pickle.dump(final_dict, f)
    
    df = pd.DataFrame(final_dict)

    pa = configs.data_path+ '/mismatch_df.csv'
    df.to_csv(pa)


def preprocess_all_submission(configs, orig_df):
    gt_code_ls = []
    student_ls = []
    prompt_ls = []
    score_ls = []
    bi_score_ls = []

    for i in range(7):
        filename = '/student_pair_all_' + str(i) + '.pkl'
        dataset = pd.read_pickle(configs.data_path + filename)

        for student, info in dataset.items():
            codes = info['ground_truth_codes']
            bi_scores = info['binary_correctness']
            match_ls = info['score_match']
            prompts = info['prompts']
            scores = info['scores']

            for code, bi_score, match, prompt, score in zip(codes, bi_scores, match_ls, prompts, scores):
                    if match:
                        gt_code_ls.append(code)
                        student_ls.append(student)
                        prompt_ls.append(prompt)
                        score_ls.append(score)
                        bi_score_ls.append(bi_score)
        
        
    final_dict = {'Code': gt_code_ls, 'prompt': prompt_ls, 'SubjectID': student_ls, 'Score': score_ls, 'binary_correctness': bi_score_ls}

    df = pd.DataFrame(final_dict)
    # df.to_csv(configs.data_path + '/all_submission_preprocessed.csv')

    combined_df = pd.merge(orig_df, df, on=['SubjectID', 'Code', 'Score'], how='right')
    combined_df = combined_df.drop_duplicates(subset=['SubjectID', 'Code', 'timestep'], keep='first')

    combined_df.to_pickle(configs.data_path + '/dataset_granular_all.pkl')


def match_timestep(configs, tokenizer, model, device):
    dataset, sat_questions = construct_dataset(configs, tokenizer, model, device)

    maintable = pd.read_csv(configs.data_path + '/MainTable.csv')

    filtered = maintable[maintable['EventType'] == 'Run.Program']
    filtered = filtered.drop(columns=['Order', 'ToolInstances','ServerTimezone', 'CourseID','CourseSectionID', 'Score', 'Compile.Result', 'CompileMessageType', 'CompileMessageData', 'EventID', 'ParentEventID', 'SourceLocation', 'IsEventOrderingConsistent', 'EventType'])

    combined_df = pd.merge(dataset, filtered, on=['SubjectID', 'AssignmentID', 'ProblemID', 'CodeStateID'], how='left')

    combined_df = combined_df.sort_values(by=["SubjectID", "ServerTimestamp"])
    combined_df.to_pickle(configs.data_path + '/dataset_time.pickle')

    

@hydra.main(version_base=None, config_path=".", config_name="configs_okt")
def main(configs):
    preprocess_all_submission(configs, [])

if __name__ == "__main__":
    main()

