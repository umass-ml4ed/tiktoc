import torch
import pickle
import nltk
from nltk import ngrams
import os
from multiprocessing import Pool
import abc
from tqdm import tqdm
from pdb import set_trace
import hydra
import json
from transformers import GenerationConfig
from model import *
import wandb
import time
from sklearn.metrics import f1_score, roc_auc_score
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind



from model import create_tokenizer
from utils import set_random_seed
from trainer import *
from data_loader import *
from evaluator.CodeBLEU import calc_code_bleu
from huggingface_hub import login
from utils import aggregate_metrics
import warnings

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def test_case_check():
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

    return sat_questions, good_test_case

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


def evaluate(configs, now, test_set, lstm_inputs, tokenizer, device):
    # TODO p2: add batch generation for GPT2 and assert outputs are equal to single generation (https://github.com/huggingface/transformers/pull/7552)
    results = {}
    lstm = None
    
    if configs.save_model:
        # Load best models
        if configs.okt_model == 'codellama/CodeLlama-7b-Instruct-hf' or configs.okt_model == 'meta-llama/Meta-Llama-3-8B-Instruct' or configs.okt_model == 'Qwen/Qwen1.5-7B':
            model = okt_model_init(configs, device, now, False, load_in_8bit=True)
            linear = nn.Linear(configs.lstm_hid_dim, 4096).to(device)
        else:
            model = AutoModelForCausalLM.from_pretrained(os.path.join(configs.model_save_dir, now, 'model')).to(device)
            linear = nn.Linear(configs.lstm_hid_dim, 768).to(device)
        
        linear.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'linear')))
        
        # ## Used for 20 epoch trained model
        # linear = torch.load(os.path.join(configs.model_save_dir, now, 'linear'))

        if configs.use_lstm:
            # lstm = torch.load(os.path.join(configs.model_save_dir, now, 'lstm'))
            lstm = create_lstm_model(configs, device)
            lstm.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'lstm')))

    else:
        lstm, tokenizer, model, linear = create_okt_model(configs, device)


    tokenizer.padding_side = 'left'

    # Set model to eval mode
    model.eval()
    linear.eval()
    if configs.use_lstm:
        lstm.eval()

    test_set = make_pytorch_dataset(test_set, None, do_lstm_dataset=False)

    generated_codes = []
    ground_truth_codes = []
    prompts = []
    students = []

    # start = time.time()
    for idx in tqdm(range(len(test_set)), desc="inference"):
        generated_code, ground_truth_code, prompt, student = generate_code(test_set, lstm_inputs, tokenizer, 
                                                                idx, model, lstm, linear, configs, device)
        
        
        # generated_code, ground_truth_code = generate_without_knowledge_state(test_set, tokenizer, idx, model, configs, device)

        
        generated_codes.append(generated_code)
        # if idx < 5:
        #     print('Generated code:', generated_code)
        #     print('Ground_truth code:', ground_truth_code)
        ground_truth_codes.append(ground_truth_code)
        prompts.append(prompt)
        students.append(student)
        
    # end = time.time()
    # print('Individual inference takes:', end - start)
    
    ## compute codebleu
    codebleu_score, detailed_codebleu_score = compute_code_bleu(ground_truth_codes, generated_codes)
    results['codebleu'] = codebleu_score
    results['detailed_codebleu'] = detailed_codebleu_score
    
    ## compute diversity
    metrics = {'dist_1': Distinct_N(1), 
               'dist_2': Distinct_N(2), 
               'dist_3': Distinct_N(3),
    }
    for i, (name, metric) in enumerate(metrics.items()):
        metric_result = metric.compute_metric(generated_codes)
        results[name] = metric_result

    print(f"results: {results}")

    ## save results
    results['generated_codes'] = generated_codes
    results['ground_truth_codes'] = ground_truth_codes
    results['prompts'] = prompts
    results['students'] = students
    
    if configs.save_model:
        with open(os.path.join(configs.model_save_dir, now, 'eval_logs.pkl'), 'wb') as f:
            pickle.dump(results, f)
        with open(os.path.join(configs.model_save_dir, now, 'eval_logs.txt'), 'w') as f:
            json.dump(results, f, indent=2)

    # # # write results to wandb
    # if configs.log_wandb:
    #     for idx, (k, v) in enumerate(results.items()):
    #         wandb.log({'metrics/test/generation_{}'.format(k): str(v)})

    return results

# Added student to the return item, so that the generated code could match with their student for calculating test case score
def generate_code(test_set, lstm_inputs, tokenizer, idx, model, lstm, linear, configs, device):
    # Get student knowledge state
    student, step, prompt, code = test_set[idx]['SubjectID'], test_set[idx]['step'], test_set[idx]['next_prompt'], test_set[idx]['next_code']
    ks = get_knowledge_states_for_generator(lstm, lstm_inputs, [student], [step], configs, device, generation=True)
    
    # Get generator input
    inputs = tokenizer(build_prompt_with_special_tokens(prompt, tokenizer, configs), return_tensors='pt')
    if configs.okt_model != 'codellama/CodeLlama-7b-Instruct-hf' and configs.okt_model != 'meta-llama/Meta-Llama-3-8B-Instruct' and configs.okt_model != 'Qwen/Qwen1.5-7B':
        inputs_embeds = model.transformer.wte(inputs['input_ids'].to(device))
    else:
        inputs_embeds = model.base_model.model.model.embed_tokens(inputs['input_ids'].to(device))
    
    # Add linear transformation of student knowledge state with prompt tokens including delimiter ":" matching finetuning format
    inputs_embeds = torch.add(inputs_embeds, linear(ks[0]))
    
    # Generate student code by greedy decoding
    config = GenerationConfig(max_new_tokens=configs.max_new_tokens, do_sample=False)

    inputs_embeds = inputs_embeds.to(dtype=model.dtype, device=device)
    
    attention_mask = inputs['attention_mask'].to(device=device, dtype=model.dtype)

    # Set eos_token_id in generate() to tokenizer.eos_token_id manually for codeLlama and use terminaters for Llama-3
    if configs.okt_model == 'meta-llama/Meta-Llama-3-8B-Instruct':
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs = model.generate(inputs_embeds=inputs_embeds, max_new_tokens=configs.max_new_tokens, do_sample=False, generation_config=config, bos_token_id=tokenizer.bos_token_id, pad_token_id=tokenizer.eos_token_id, eos_token_id=terminators, attention_mask=attention_mask)
    else:
        outputs = model.generate(inputs_embeds=inputs_embeds, max_new_tokens=configs.max_new_tokens, do_sample=False, generation_config=config, bos_token_id=tokenizer.bos_token_id, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, attention_mask=attention_mask)
    
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_code.strip(), code.strip(), prompt, student



def eval_student(configs, now, test_set, dataset, tokenizer, device):
    results = {}
    lstm = None
    predictor = None
    
    if configs.save_model:
        if configs.okt_model == 'codellama/CodeLlama-7b-Instruct-hf' or configs.okt_model == 'meta-llama/Meta-Llama-3-8B-Instruct' or configs.okt_model == 'Qwen/Qwen1.5-7B':
            model = okt_model_init(configs, device, now, False, load_in_8bit=True)
            # linear = nn.Linear(configs.lstm_hid_dim, 4096).to(device)
            linear = nn.Sequential(
                nn.Linear(configs.lstm_hid_dim, 1600),
                nn.ReLU(),
                nn.Linear(1600, 4096)
            ).to(device)
            
        else:
            model = AutoModelForCausalLM.from_pretrained(os.path.join(configs.model_save_dir, now, 'model')).to(device)
            linear = nn.Linear(configs.lstm_hid_dim, 768).to(device)
        
        linear.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'linear')))

        if configs.use_lstm:
            lstm = create_lstm_model(configs, device)
            lstm.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'lstm')))

        if configs.multitask:
            if configs.multitask_label != 'granular':
                predictor = create_multitask_predictor(configs, device)
                predictor.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'predictor')))
            else:
                predictor = torch.load(os.path.join(configs.model_save_dir, now, 'predictor.pth'))

    else:
        lstm, tokenizer, model, linear = create_okt_model(configs, device)
        if configs.multitask:
            if configs.multitask_label != 'granular':
                predictor = create_multitask_predictor(configs, device)
            else:
                predictor = create_granular_model(configs, device)

    tokenizer.padding_side = 'left'

    # Set model to eval mode
    model.eval()
    linear.eval()
    if configs.use_lstm:
        lstm.eval()


    granular = False
    question_input_dict = None
    question_no_map = None
    if configs.multitask_label == 'granular':
        granular = True
        _, good_test_case = test_case_check()
        question_input_dict = uniq_test_construct(good_test_case)
        question_input_dict = handle_uniq_test_exception(question_input_dict)
        question_ids = [1, 3, 5, 12, 13, 17, 20, 21, 22, 24, 25, 34, 37, 39, 40, 46, 71]
        question_no_map = {question_ids[i]:i for i in range(len(question_ids))}
    
    collate_fn = CollateForOKTstudent(tokenizer=tokenizer, configs=configs, device=device, eval=True, question_test_dict=question_input_dict, question_no_map=question_no_map)

    _, test_loader  = make_dataloader(test_set, dataset, collate_fn=collate_fn, configs=configs, do_lstm_dataset=True, 
                                      train=False, split_by_student=True, granular=granular, okt_model=True)

    
    generated_code_total, gt_code_total, prompt_total, pred_score_total, gt_score_total, pred_label_total, student_total = [], [], [], [], [], [], []

    for idx, batch in enumerate(tqdm(test_loader, desc="inference", leave=False)):
        if configs.multitask:
            if configs.multitask_label == 'granular':
                generated_code_ls, gt_code_ls, prompt_ls, pred_score_ls, gt_score_ls, pred_label_ls, student_ls = generate_code_student(batch, tokenizer, model, lstm, linear, configs, device, predict_linear=predictor)
                pred_label_total.append(pred_label_ls)
            else:
                generated_code_ls, gt_code_ls, prompt_ls, pred_score_ls, gt_score_ls, student_ls = generate_code_student(batch, tokenizer, model, lstm, linear, configs, device, predict_linear=predictor)
            pred_score_total.append(pred_score_ls)
            gt_score_total.append(gt_score_ls)
        else:
            generated_code_ls, gt_code_ls, prompt_ls, student_ls = generate_code_student(batch, tokenizer, model, lstm, linear, configs, device, predict_linear=predictor)
        
        generated_code_total.append(generated_code_ls)
        gt_code_total.append(gt_code_ls)
        prompt_total.append(prompt_ls)
        student_total.append(student_ls)

    generated_codes = [gen_code_i for code_ls in generated_code_total for gen_code_i in code_ls]
    gt_codes = [gt_code_i for code_ls in gt_code_total for gt_code_i in code_ls]
    prompts = [prompt_i for prompt_ls in prompt_total for prompt_i in prompt_ls]
    students = [student_i for student_ls in student_total for student_i in student_ls]

    if configs.multitask:
        if configs.multitask_label != 'granular':
            pred_scores = [pred_score_i for pred_subset in pred_score_total for pred_score_i in pred_subset]
            gt_scores = [gt_score_i for gt_subset in gt_score_total for gt_score_i in gt_subset]
            
            mse = np.square(np.subtract(pred_scores, gt_scores)).mean()
            results['MSE'] = mse
        else:
            pred_scores, gt_scores, pred_labels = [], [], []

            overall_gt_ls, overall_pred_ls =[], []

            cherry_pick_pred_total, cherry_pick_label_total, cherry_pick_score_total = [], [], []

            for res_list_i in range(len(gt_score_total)):
                res_list_gt = gt_score_total[res_list_i]
                res_list_pred = pred_score_total[res_list_i]
                res_list_label = pred_label_total[res_list_i]

                for test_results_i in range(len(res_list_gt)):
                    labels_i_gt = res_list_gt[test_results_i]
                    labels_i_pred = res_list_pred[test_results_i]
                    labels_i = res_list_label[test_results_i]

                    cherry_pick_pred_total.append(labels_i)
                    cherry_pick_label_total.append(labels_i_gt)
                    cherry_pick_score_total.append(labels_i_pred)

                    valid_gt_ls = []
                    valid_pred_ls = []

                    for gran_i in range(len(labels_i_gt)):
                        if labels_i_gt[gran_i] != -100:
                            gt_scores.append(labels_i_gt[gran_i])
                            pred_scores.append(labels_i_pred[gran_i])
                            pred_labels.append(labels_i[gran_i])

                            valid_gt_ls.append(labels_i_gt[gran_i])
                            valid_pred_ls.append(labels_i[gran_i])

                    # Used for MSE calculation
                    gt_score_overall = np.mean(valid_gt_ls)
                    pred_score_overall = np.mean(valid_pred_ls)

                    overall_gt_ls.append(gt_score_overall)
                    overall_pred_ls.append(pred_score_overall)

            pred_res = sum([pred == label for pred, label in zip(pred_labels, gt_scores)])
            acc = pred_res / len(gt_scores)
            results['Acc'] = acc

            f1 = f1_score(gt_scores, pred_labels)
            results['F1'] = f1

            auc = roc_auc_score(gt_scores, pred_scores)
            results['AUC'] = auc

            mse = np.square(np.subtract(overall_gt_ls, overall_pred_ls)).mean()
            results['MSE'] = mse

        
    codebleu_score, detailed_codebleu_score = compute_code_bleu(gt_codes, generated_codes)
    results['codebleu'] = codebleu_score
    results['detailed_codebleu'] = detailed_codebleu_score
    
    ## compute diversity
    metrics = {'dist_1': Distinct_N(1), 
               'dist_2': Distinct_N(2), 
               'dist_3': Distinct_N(3),
    }
    for i, (name, metric) in enumerate(metrics.items()):
        metric_result = metric.compute_metric(generated_codes)
        results[name] = metric_result

    print(f"results: {results}")

    ## save results
    results['generated_codes'] = generated_codes
    results['ground_truth_codes'] = gt_codes
    results['prompts'] = prompts
    results['students'] = students
    if configs.multitask:
        results['prediction'] = cherry_pick_pred_total
        results['labels'] = cherry_pick_label_total
        results['prob'] = cherry_pick_score_total
    
    if configs.save_model:
        with open(os.path.join(configs.model_save_dir, now, 'eval_logs.pkl'), 'wb') as f:
            pickle.dump(results, f)
        with open(os.path.join(configs.model_save_dir, now, 'eval_logs.txt'), 'w') as f:
            json.dump(results, f, indent=2)

    return results

    
def generate_code_student(batch, tokenizer, model, lstm, linear, configs, device, predict_linear=None):
    gen_code_ls, gt_code_ls, prompt_ls, gt_score_ls, pred_score_ls, pred_label_ls, student_ls = [], [], [], [], [], [], []
    padded_inputs, padded_input_ids_ls, padded_attention_mask_ls ,padded_codes, padded_prompts, padded_scores, padded_question_seqs, padded_students = batch[0][:-1], batch[1][1:], batch[2][1:], batch[3][1:], batch[4][1:], batch[5][1:], batch[6][1:], batch[7][1:]

    if configs.okt_model != 'codellama/CodeLlama-7b-Instruct-hf' and configs.okt_model != 'meta-llama/Meta-Llama-3-8B-Instruct' and configs.okt_model != 'Qwen/Qwen1.5-7B':
        generator_input_wte = model.transformer.wte(padded_input_ids_ls).to(device)
    else:
        generator_input_wte = model.base_model.model.model.embed_tokens(padded_input_ids_ls).to(device)

    if configs.use_lstm:
        ks, hidden = lstm(padded_inputs)
        ks = linear(ks)

        if configs.multitask_inp == 'concat':
            avg_question_emb = torch.mean(generator_input_wte, dim=2)
            pred_input_emb = torch.cat((ks, avg_question_emb), dim=-1)


        ks = ks.unsqueeze(2).repeat(1, 1, padded_input_ids_ls.size(2), 1)
        generator_input_wte = torch.add(generator_input_wte, ks)

    generator_input_wte = generator_input_wte.to(dtype=model.dtype, device=device)
    padded_attention_mask_ls = padded_attention_mask_ls.to(device=device, dtype=model.dtype)

    T, B, max_length, D = generator_input_wte.shape
    generator_input_wte = generator_input_wte.view((T * B), max_length, D)
    padded_attention_mask_ls = padded_attention_mask_ls.reshape((T * B), -1)
    padded_scores = torch.unsqueeze(padded_scores, -1).reshape((T * B), -1)
    
    flattened_codes = [code_i for subcode in padded_codes for code_i in subcode]
    flattened_prompt = [prompt_i for subprompt in padded_prompts for prompt_i in subprompt]
    flattened_students = [student_i for substudent in padded_students for student_i in substudent]

    input_wte_subset = torch.split(generator_input_wte, 1)
    attention_mask_subset = torch.split(padded_attention_mask_ls, 1)
    
    if configs.multitask:
        if configs.use_lstm:
            ks = ks.reshape((T * B), max_length, D)
            ks_subset = torch.split(ks, 1)
        scores_subset = torch.split(padded_scores, 1)
        if configs.multitask_inp == 'concat':
            pred_input_emb = pred_input_emb.reshape((T * B), -1)
            pred_input_subset = torch.split(pred_input_emb, 1)
        
        if configs.multitask_label == 'granular': 
            padded_ques_seqs = torch.unsqueeze(padded_question_seqs, -1)
            padded_ques_seqs = padded_ques_seqs.reshape((T * B), -1)
            ques_seqs_groups = torch.split(padded_ques_seqs, 1)

    config = GenerationConfig(max_new_tokens=configs.max_new_tokens, do_sample=False)

    for i in range(len(input_wte_subset)):
        ground_truth_code = flattened_codes[i]
        prompt = flattened_prompt[i]
        student = flattened_students[i]
        if ground_truth_code:
            input_wte_i = input_wte_subset[i]
            attention_i = attention_mask_subset[i]

            if configs.okt_model == 'meta-llama/Meta-Llama-3-8B-Instruct':
                terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                outputs = model.generate(inputs_embeds=input_wte_i, max_new_tokens=configs.max_new_tokens, do_sample=False, generation_config=config, bos_token_id=tokenizer.bos_token_id, pad_token_id=tokenizer.eos_token_id, eos_token_id=terminators, attention_mask=attention_i)
            else:
                outputs = model.generate(inputs_embeds=input_wte_i, max_new_tokens=configs.max_new_tokens, do_sample=False, generation_config=config, bos_token_id=tokenizer.bos_token_id, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, attention_mask=attention_i)
            
            generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

            gen_code_ls.append(generated_code.strip())
            gt_code_ls.append(ground_truth_code.strip())
            prompt_ls.append(prompt)
            student_ls.append(student)

            if configs.multitask:
                granular = True if configs.multitask_label == 'granular' else False
                if configs.multitask_inp != 'concat':
                    if configs.multitask_label == 'granular':
                        prob_seq_sub = ques_seqs_groups[i].squeeze(-1)
                        predicted_label, predicted_score = predict_score_question_only(input_wte_i, attention_i, model, predict_linear, granular=granular, problem_seqs=prob_seq_sub)
                    else:
                        predicted_score = predict_score_question_only(input_wte_i, attention_i, model, predict_linear, granular=granular)
                else:
                    predicted_score = predict_score_concat(pred_input_subset[i], predict_linear)
                
                if granular:
                    pred_score_ls.append(predicted_score)
                    pred_label_ls.append(predicted_label)
                    gt_score_ls.append(scores_subset[i][0].tolist())
                
                else:
                    pred_score_ls.append(predicted_score.item())
                    gt_score_ls.append(scores_subset[i][0][0].cpu().item())

 
    if configs.multitask:
        if granular:
            return gen_code_ls, gt_code_ls, prompt_ls, pred_score_ls, gt_score_ls, pred_label_ls, student_ls
        
        return gen_code_ls, gt_code_ls, prompt_ls, pred_score_ls, gt_score_ls, student_ls
    
    return gen_code_ls, gt_code_ls, prompt_ls, student_ls



# Multitask Model predictor inference Version 2: take the question embedding only
def predict_score_question_only(generator_input_wte, attention_mask, model, predict_linear, granular=False, problem_seqs=None):
    eps = 1e-8

    with torch.no_grad():
        output = model(inputs_embeds=generator_input_wte, attention_mask=attention_mask, output_hidden_states=True, return_dict=True)
        hidden_states = output['hidden_states'][-1]

        attention_sub_expand = torch.unsqueeze(attention_mask, -1)
        hidden_states_valid = hidden_states * attention_sub_expand
        pooled_out = hidden_states_valid.sum(dim=1)
        valid_cnt = attention_sub_expand.sum(dim=1)
        pooled_out = pooled_out / (valid_cnt + eps)

        if granular:
            pooled_out = pooled_out.unsqueeze(1)
            model_weight = predict_linear[problem_seqs]
            logits = torch.matmul(pooled_out, model_weight)
            logits = logits.squeeze(1)
            
        else:
            logits = predict_linear(pooled_out)

        score = torch.sigmoid(logits)
    
    if granular:
        pred = (score > 0.5) * 1
        return pred[0].tolist(), score[0].tolist()

    return score[0][0].cpu()


def predict_score_concat(pred_inp, predictor):
    with torch.no_grad():
        logits = predictor(pred_inp)
        score = torch.sigmoid(logits)

    return score[0][0].cpu()


def compute_code_bleu(ground_truth_codes, generated_codes):
    params='0.25,0.25,0.25,0.25'
    lang='java'
    codebleu_score, detailed_codebleu_score = calc_code_bleu.get_codebleu(pre_references=[ground_truth_codes], hypothesis=generated_codes, lang=lang, params=params)
    
    return codebleu_score, detailed_codebleu_score


class Metric():
    """
    Defines a text quality metric.
    """
    def get_name(self):
        return self.name


    @abc.abstractmethod
    def compute_metric(self, texts):
        pass


class Distinct_N(Metric):

    def __init__(self, n):
        """
        Distinct n-grams metrics. This is a sequence-level diversity metric.
        See https://www.aclweb.org/anthology/N16-1014 for more details.

        Args:
            n (int): n-grams 
        """
        self.n = n
        self.name = f'Distinct_{n}'


    def compute_metric(self, texts):
        return self._distinct_ngrams(texts, self.n)


    def _distinct_ngrams(self, texts, n):
        total = 0.0
        for t in texts:
            try:
                tokens = nltk.tokenize.word_tokenize(t)
                n_distinct = len(set(ngrams(tokens, n)))
                total += n_distinct/ len(tokens)
            except Exception as e:
                print(f"Exception in computing Distinct_N metric: {e}")
                continue

        return total / len(texts)

def batch_generate(test_set, lstm_inputs, tokenizer, model, lstm, linear, configs, device):
    prompts = [test_set[i]['next_prompt'] for i in range(len(test_set))]
    ground_truch_codes =  [test_set[i]['next_code'].strip() for i in range(len(test_set))]
    students = [test_set[i]['SubjectID'] for i in range(len(test_set))]

    full_prompts = [build_prompt_with_special_tokens(prompt, tokenizer, configs) for prompt in prompts]

    inputs = tokenizer(full_prompts, return_tensors='pt', padding=True, truncation=True)

    config = GenerationConfig(max_new_tokens=400, do_sample=False)

    inputs_embeds = model.base_model.model.model.embed_tokens(inputs['input_ids'].to(device))
    attention_mask = inputs['attention_mask'].to(device=device, dtype=model.dtype)

    ks_batch = [get_knowledge_states_for_generator(lstm, lstm_inputs, [test_set[i]['SubjectID']], [test_set[i]['step']], configs, device, generation=True) for i in range(len(test_set))]

    ks_batch_ts = torch.stack(ks_batch)
    inputs_embeds = torch.add(inputs_embeds, linear(ks_batch_ts))

    inputs_embeds = inputs_embeds.to(dtype=model.dtype, device=device)
    outputs = model.generate(inputs_embeds=inputs_embeds, max_new_tokens=400, do_sample=False, generation_config=config, bos_token_id=tokenizer.bos_token_id, pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id, attention_mask=attention_mask)
    generated_code = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    generated_code = [i.strip() for i in generated_code]

    return generated_code, ground_truch_codes, prompts, students

def batch_inference(configs, now, tokenizer, test_set, lstm_inputs, device):
    results = {}
    lstm = None
    
    if configs.save_model:
        # Load best models
        if configs.okt_model == 'llama-3':
            model = okt_model_init(configs, device, now, False)
        else:
            model = torch.load(os.path.join(configs.model_save_dir, now, 'model'))
        linear = torch.load(os.path.join(configs.model_save_dir, now, 'linear'))    
        if configs.use_lstm:
            lstm = torch.load(os.path.join(configs.model_save_dir, now, 'lstm'))

    else:
        lstm, tokenizer, model, linear = create_okt_model(configs, device)

    tokenizer.padding_side = 'left'

    # Set model to eval mode
    model.eval()
    linear.eval()
    if configs.use_lstm:
        lstm.eval()

    test_set = make_pytorch_dataset(test_set, None, do_lstm_dataset=False)

    generated_codes = []
    ground_truth_codes = []
    prompts = []
    students = []

    for idx in tqdm(range(0, len(test_set), 20), desc="batch inference"):
        test_set_i = test_set[idx: idx+20]
        generated_code_i, ground_truth_code_i, prompts_i, students_i = batch_generate(test_set_i, lstm_inputs, tokenizer, model, lstm, linear, configs, device)
        generated_codes.extend(generated_code_i)
        ground_truth_codes.extend(ground_truth_code_i)
        prompts.extend(prompts_i)
        students.extend(students_i)
    

    ## compute codebleu
    codebleu_score, detailed_codebleu_score = compute_code_bleu(ground_truth_codes, generated_codes)
    results['codebleu'] = codebleu_score
    results['detailed_codebleu'] = detailed_codebleu_score
    
    ## compute diversity
    metrics = {'dist_1': Distinct_N(1), 
               'dist_2': Distinct_N(2), 
               'dist_3': Distinct_N(3),
    }
    for i, (name, metric) in enumerate(metrics.items()):
        metric_result = metric.compute_metric(generated_codes)
        results[name] = metric_result

    print(f"results: {results}")

    ## save results
    results['generated_codes'] = generated_codes
    results['ground_truth_codes'] = ground_truth_codes
    results['prompts'] = prompts
    results['students'] = students
    
    if configs.save_model:
        with open(os.path.join(configs.model_save_dir, now, 'eval_logs.pkl'), 'wb') as f:
            pickle.dump(results, f)
        with open(os.path.join(configs.model_save_dir, now, 'eval_logs.txt'), 'w') as f:
            json.dump(results, f, indent=2)

    # # write results to wandb
    if configs.log_wandb:
        for idx, (k, v) in enumerate(results.items()):
            wandb.log({'metrics/test/generation_{}'.format(k): str(v)})


def eval_granular(configs, test_loader, device, now, loss_function):
    if configs.save_model:
        lstm = create_lstm_model(configs, device)
        lstm.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'lstm')))
        if configs.use_transition_model:
            transition_model = create_transition_layer(configs, device)
            transition_model.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'transition')))
        else:
            transition_model = None
        granular_model = torch.load(os.path.join(configs.model_save_dir, now, 'granular_model.pth'))


    else:
        lstm = create_lstm_model(configs, device)
        granular_model = create_granular_model(configs, device)
        transition_model = None
        if configs.use_transition_model:
            transition_model = create_transition_layer(configs, device)
    
    lstm.eval()

    inf_logs = []

    pred_total = []
    label_total = []

    for idx, batch in enumerate(tqdm(test_loader, desc='inferece', leave=False)):
        test_log, pred_ls = predict_granular_step(idx, batch, lstm, granular_model, transition_model=transition_model, configs=configs, 
                                        loss_fn=loss_function, train=False, device=device, eval=True)
        inf_logs.append(test_log)

        logits_sub =  test_log['auc']['logits']
        pred_tensor = (torch.sigmoid(logits_sub) > 0.5) * 1
        pred_sub = pred_tensor.tolist()
        label_sub = test_log['auc']['scores'].tolist()

        pred_total.append(pred_sub)
        label_total.append(label_sub)

    preds = [pred for pred_ls in pred_total for pred in pred_ls]
    labels = [lab for lab_ls in label_total for lab in lab_ls]

    final_res = aggregate_metrics(inf_logs)
    f1 = f1_score(labels, preds)
    final_res['F1'] = f1

    with open('granularKT_res', 'wb') as fp:
        pickle.dump(pred_ls, fp)

    return final_res


def nested_dict():
    return defaultdict(list)  

def cal_p_value(a, b):
    stat = ttest_ind(a, b, equal_var=False)
    print(stat)


@hydra.main(version_base=None, config_path=".", config_name="configs_okt")
# @hydra.main(version_base=None, config_path=".", config_name="configs_okt_testcase")
def main(configs):
    warnings.filterwarnings("ignore")
    # Make reproducible
    set_random_seed(configs.seed)

    # now = configs.checkpoint
    now = '20240917_005126' #all-submission-TIKTOC
    print(now)
    
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    if configs.use_cuda: assert device.type == 'cuda', 'No GPU found'

    if configs.okt_model == 'meta-llama/Meta-Llama-3-8B-Instruct':
        login(token='')

    if configs.log_wandb:
        wandb.login(key=configs.wandb_key, verify=True)
        wandb.init(project=configs.wandb_project, id="22mhccqg", resume="must")

    tokenizer = create_tokenizer(configs)

    if configs.exp_name == 'okt':
        if configs.split_by == 'submission':
            if configs.save_model:
                # Load best models
                if configs.okt_model == 'codellama/CodeLlama-7b-Instruct-hf' or configs.okt_model == 'meta-llama/Meta-Llama-3-8B-Instruct' or configs.okt_model == 'Qwen/Qwen1.5-7B':
                    model = okt_model_init(configs, device, now, False, load_in_8bit=True)
            collate_fn = CollateForOKT(tokenizer=tokenizer, configs=configs, device=device)
            train_set, valid_set, test_set, dataset, students = read_data(configs, tokenizer, model, device)
            lstm_inputs = get_lstm_inputs(configs, train_set, dataset, collate_fn)

            print('start eval func:')
            res = evaluate(configs, now, test_set, lstm_inputs, tokenizer, device)
            # batch_inference(configs, now, tokenizer, test_set, lstm_inputs, device)
        else:
            train_set, valid_set, test_set, dataset, students = read_granular_data(configs)

            print('start eval func:')
            res = eval_student(configs, now, test_set, dataset, tokenizer, device)
    
        if configs.log_wandb:
            result = {'codeBLEU': res['codebleu']}
            result['Acc'] = res['Acc']
            result['AUC'] = res['AUC']
            result['F1'] = res['F1']
            result['MSE'] = res['MSE']

            wandb.log(result)

            wandb.finish()


if __name__ == "__main__":
    #torch.set_printoptions(profile="full")
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.cuda.empty_cache()

    main()
