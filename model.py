import torch
import torch.optim as optim
import os
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftConfig, PeftModel, LoraConfig, prepare_model_for_kbit_training, get_peft_model


def create_lstm_model(configs, device):
    if configs.okt_model == 'codellama/CodeLlama-7b-Instruct-hf' or configs.okt_model == 'meta-llama/Meta-Llama-3-8B-Instruct' or configs.okt_model == 'Qwen/Qwen1.5-7B':
        configs.lstm_inp_dim = 4296 # 4096+200 
    lstm = nn.LSTM(configs.lstm_inp_dim, configs.lstm_hid_dim, num_layers=configs.num_layers)
    lstm.to(device)
    
    return lstm

def create_tokenizer(configs):
    if configs.okt_model == 'codellama/CodeLlama-7b-Instruct-hf' or configs.okt_model == 'meta-llama/Meta-Llama-3-8B-Instruct' or configs.okt_model == 'Qwen/Qwen1.5-7B':
        tokenizer = AutoTokenizer.from_pretrained(configs.okt_model)
    else:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def create_okt_model(configs, device):
    # load the code generator model
    tokenizer = create_tokenizer(configs)
    
    if configs.okt_model == 'student':
        model = AutoModelForCausalLM.from_pretrained("model/gpt_code_v1_student")
    elif configs.okt_model == 'funcom':
        model = AutoModelForCausalLM.from_pretrained("model/gpt_code_v1")
    elif configs.okt_model == 'gpt-2':
        model = AutoModelForCausalLM.from_pretrained('gpt2')
    else:
        bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            configs.okt_model,
            quantization_config=bnb_config
        )

        lora_config = LoraConfig(
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            r=configs.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head", ],
            inference_mode=False
        )

        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
    
    model.to(device)

    if configs.okt_model != 'codellama/CodeLlama-7b-Instruct-hf' and configs.okt_model != 'meta-llama/Meta-Llama-3-8B-Instruct' and configs.okt_model != 'Qwen/Qwen1.5-7B':
        linear = nn.Linear(configs.lstm_hid_dim, 768).to(device)
    else: 
        # linear = nn.Linear(configs.lstm_hid_dim, 4096).to(device)

        # Add two hidden layers to transfer
        linear = nn.Sequential(
            nn.Linear(configs.lstm_hid_dim, 1600),
            nn.ReLU(),
            nn.Linear(1600, 4096)
        ).to(device)
    
    # Create LSTM to compute knowledge states of students over time
    lstm = None
    if configs.use_lstm:
        lstm = create_lstm_model(configs, device)
    
    return lstm, tokenizer, model, linear


def load_okt_model(configs, device, now, continue_train):
    tokenizer = create_tokenizer(configs)
    model = okt_model_init(configs, device, now, continue_train)

    if configs.okt_model != 'codellama/CodeLlama-7b-Instruct-hf' and configs.okt_model != 'meta-llama/Meta-Llama-3-8B-Instruct' and configs.okt_model != 'Qwen/Qwen1.5-7B':
        linear = nn.Linear(configs.lstm_hid_dim, 768).to(device)
    else: 
        linear = nn.Linear(configs.lstm_hid_dim, 4096).to(device)

    linear.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'linear')))
    lstm = create_lstm_model(configs, device)
    lstm.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'lstm')))

    return lstm, tokenizer, model, linear



def okt_model_init(configs, device, now, continue_train, load_in_8bit=True):
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=load_in_8bit,
        bnb_8bit_compute_dtype=torch.bfloat16 if continue_train else torch.float16
    )

    model_dir = os.path.join(configs.model_save_dir, now, 'model')
    peft_config = PeftConfig.from_pretrained(model_dir)
    _hf_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        quantization_config=bnb_config,
    )

    model = PeftModel.from_pretrained(_hf_model, model_dir, is_trainable=continue_train).to(device)

    for param in model.parameters():
        if param.dtype == torch.float16:
            param.data = param.data.float()
            if param.grad is not None:
                param.grad.data = param.grad.data.float()

    return model


def create_granular_model(configs, device):
    if configs.okt_model == 'codellama/CodeLlama-7b-Instruct-hf' or configs.okt_model == 'meta-llama/Meta-Llama-3-8B-Instruct' or configs.okt_model == 'Qwen/Qwen1.5-7B':
        prompt_dim = 4096
    else:
        prompt_dim = 768
    
    if configs.use_transition_model:
        model = nn.Parameter(torch.empty((configs.valid_question_no, configs.trans_hid_dim, configs.T_max)).to(device), requires_grad=True)
    else:
        if configs.multitask_label:
            model = nn.Parameter(torch.empty((configs.valid_question_no, prompt_dim, configs.T_max)).to(device), requires_grad=True)
        else:
            if configs.use_lstm:
                model = nn.Parameter(torch.empty((configs.valid_question_no, configs.lstm_hid_dim + prompt_dim, configs.T_max)).to(device), requires_grad=True)
            else:
                model = nn.Parameter(torch.empty((configs.valid_question_no, prompt_dim, configs.T_max)).to(device), requires_grad=True)

    # nn.init.normal_(model.data, mean=0.0, std=0.1)
    torch.nn.init.xavier_uniform_(model.data)

    return model

# Optionally created in granularDKT model based on configs.use_transition_model
def create_transition_layer(configs, device):
    if configs.okt_model == 'codellama/CodeLlama-7b-Instruct-hf' or configs.okt_model == 'meta-llama/Meta-Llama-3-8B-Instruct' or configs.okt_model == 'Qwen/Qwen1.5-7B':
        prompt_dim = 4096
    else:
        prompt_dim = 768

    transition_layer = nn.Sequential(
        nn.Linear(configs.lstm_hid_dim+prompt_dim, configs.trans_hid_dim),
        nn.ReLU()
    )
    transition_layer = transition_layer.to(device)

    return transition_layer

def create_multitask_predictor(configs, device):
    if configs.okt_model == 'codellama/CodeLlama-7b-Instruct-hf' or configs.okt_model == 'meta-llama/Meta-Llama-3-8B-Instruct' or configs.okt_model == 'Qwen/Qwen1.5-7B':
        hid_dim = 4096
    else:
        hid_dim = 768
    
    if configs.multitask_inp == 'hid':
        predictor = nn.Linear(hid_dim, 1).to(device)
    else:
        predictor = nn.Linear(hid_dim + 4096, 1).to(device)

    torch.nn.init.xavier_uniform_(predictor.weight)
    return predictor


def create_multi_linear_with_emd(device, tokenizer, model, question_dict, solution, question_in=False, question_prompt_dict=None):
    prompt_dim = 4096
    weight_ls = []

    for key in question_dict.keys():
        test_inputs = question_dict[key]
        sol = solution[key]
        if question_in:
            question = question_prompt_dict[key]
        testcase_pair = []
        for i in range(len(test_inputs)):
            pair = 'Test case input: <' + test_inputs[i] + '> Test case output: <' +sol[i] + '>'
            if question_in:
                pair = question + ' ' + pair

            testcase_pair.append(pair)

        test_embedding = tokenizer(test_inputs, return_tensors='pt', padding=True)
        token_embedding = model.base_model.model.model.embed_tokens(test_embedding['input_ids'].to(device))
        avg_emb = torch.mean(token_embedding, dim=1).t()

        dummy_weight = torch.empty(prompt_dim, 26 - avg_emb.shape[-1]).to(device)
        torch.nn.init.xavier_uniform_(dummy_weight.data)
        combined_weight = torch.cat((avg_emb, dummy_weight), dim=1)
        weight_ls.append(combined_weight)
    
    model_weight = torch.stack(weight_ls, dim=0)
    model_weight = torch.nn.Parameter(model_weight)
    return model_weight


    
