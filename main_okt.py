import os
from datetime import datetime
import torch.optim as optim
import transformers
import hydra
from omegaconf import OmegaConf
import wandb

from data_loader import *
from model import *
from trainer import *
from utils import *
from eval import *
from huggingface_hub import login
from pdb import set_trace
from gen_code_compiler import test_case_check, uniq_test_construct, handle_uniq_test_exception, get_test_case_solution


def sanitize_configs(configs):
    assert ( (configs.use_lstm == False and configs.lstm_hid_dim == 0) or (configs.use_lstm == True and configs.lstm_hid_dim > 0) ), "Invalid combination of configs use_lstm and lstm_hid_dim"


@hydra.main(version_base=None, config_path=".", config_name="configs_okt")
def main(configs):
    torch.cuda.empty_cache()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    # now = '20240818_205248'
    print(now)
    # Make reproducible
    set_random_seed(configs.seed)
    # Sanity checks on config
    # sanitize_configs(configs)

    print(configs.okt_model)
    if configs.okt_model == 'meta-llama/Meta-Llama-3-8B-Instruct':
        login(token='')
    
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if configs.use_cuda: 
        if torch.cuda.is_available():
            device = torch.device('cuda')
        assert device.type == 'cuda', 'No GPU found'
    # Apple metal acceleration: don't enable for now since some operations are not implemented in MPS and torch.gather has an issue (https://github.com/pytorch/pytorch/issues/94765)
    #elif( torch.backends.mps.is_available() ):
    #    device = torch.device("mps")
    else:
        device = torch.device('cpu')    

    # Test on smaller fraction of dataset
    if configs.testing:
        configs.epochs = 2
        configs.log_wandb = False
        configs.save_model = False
        configs.batch_size = 4

    # # Use wandb to track experiment
    if configs.log_wandb:
        wandb.login(key=configs.wandb_key, verify=True)
        if configs.continue_training:
            # need to pass the run id
            wandb.init(project=configs.wandb_project, id="", resume="must")
        else:
            wandb.init(project=configs.wandb_project)
            print('Run id:', wandb.run.id)
            wandb.config.update(OmegaConf.to_container(configs, resolve=True))

    
    if configs.save_model:
        os.makedirs(os.path.join(configs.model_save_dir, now), exist_ok=True)
    
    ## load model
    if configs.continue_training:
        lstm, tokenizer, model, linear = load_okt_model(configs, device, now, True)
        
    else:    
        lstm, tokenizer, model, linear = create_okt_model(configs, device)   
        predictor = None
        if configs.multitask:
            if configs.multitask_label == 'raw':
                predictor = create_multitask_predictor(configs, device)
                

    ## load the init dataset
    if configs.split_by == 'submission':
        # train_set, valid_set, test_set, dataset, students = read_data(configs, tokenizer, model, device)

        # Create new dataset for okt by submission on valid question dataset
        train_set, valid_set, test_set, dataset = construct_okt_dataset_from_granular(configs)

        ## load data
        collate_fn = CollateForOKT(tokenizer=tokenizer, configs=configs, device=device)
        # _, train_loader, lstm_inputs = make_dataloader(train_set, dataset, 
        #                                                collate_fn=collate_fn, 
        #                                                configs=configs, do_lstm_dataset=True)

        # When running OKT from granular dataset, avoid shuffle for train, 
        # (the extracting timestep method)
        _, train_loader, lstm_inputs = make_dataloader(train_set, dataset, 
                                                    collate_fn=collate_fn, configs=configs, 
                                                    do_lstm_dataset=True, train=False)

        _, valid_loader = make_dataloader(valid_set, None, 
                                        collate_fn=collate_fn, configs=configs, 
                                        do_lstm_dataset=False, train=False)
        _, test_loader  = make_dataloader(test_set , None, 
                                        collate_fn=collate_fn, configs=configs, 
                                        do_lstm_dataset=False, train=False)

    else:
        train_set, valid_set, test_set, dataset, students = read_granular_data(configs)

        # _, good_test_case = test_case_check()
        # question_input_dict = uniq_test_construct(good_test_case)
        # question_input_dict = handle_uniq_test_exception(question_input_dict)
        # tcs = question_input_dict.values()

        # set_trace()
        # solution = get_test_case_solution(good_test_case)
        # sol = solution.values()
        # tc = [tc_i for tc_ls in tcs for tc_i in tc_ls]
        # sols = [s_i for s_ls in sol for s_i in s_ls]
        # tc_tok = ['(' + x + '): '+ y for x,y in zip(tc, sols)]
        # tokenized = tokenizer(tc_tok, padding=False, truncation=False, add_special_tokens=False)
        # num_tokens = [len(token_list) for token_list in tokenized["input_ids"]]

        
        granular = False
        question_input_dict = None
        question_no_map = None
        if configs.multitask_label == 'granular':
            granular = True
            _, good_test_case = test_case_check()
            question_input_dict = uniq_test_construct(good_test_case)
            question_input_dict = handle_uniq_test_exception(question_input_dict)
            question_ids = [1, 3, 5, 12, 13, 17, 20, 21, 22, 24, 25, 34, 37, 39, 40, 46, 71]
            quest_prompt_dict = None
            if configs.multitask_init_combine:
                question_prompt = dataset.sort_values(by='ProblemID').prompt.unique().tolist()
                quest_prompt_dict = {question_ids[i]: question_prompt[i] for i in range(len(question_prompt))}
            question_no_map = {question_ids[i]:i for i in range(len(question_ids))}
            
            if configs.multitask_init != 'emb':
                predictor = create_granular_model(configs, device)
            else:
                solution = get_test_case_solution(good_test_case)
                predictor = create_multi_linear_with_emd(device, tokenizer, model, question_input_dict, solution, question_in=configs.multitask_init_combine, question_prompt_dict=quest_prompt_dict)
        
        collate_fn = CollateForOKTstudent(tokenizer=tokenizer, configs=configs, device=device, eval=False, question_test_dict=question_input_dict, question_no_map=question_no_map)
 

        # same data loader for multitask model: with both raw score prediction and binary test case prediction
        _, train_loader = make_dataloader(train_set, dataset, 
                                        collate_fn=collate_fn, configs=configs, 
                                        do_lstm_dataset=True, train=True, split_by_student=True, granular=granular, okt_model=True)

        _, valid_loader = make_dataloader(valid_set, dataset, 
                                        collate_fn=collate_fn, configs=configs, 
                                        do_lstm_dataset=True, train=False, split_by_student=True, granular=granular, okt_model=True)
        _, test_loader  = make_dataloader(test_set , dataset, 
                                        collate_fn=collate_fn, configs=configs, 
                                        do_lstm_dataset=True, train=False, split_by_student=True, granular=granular, okt_model=True)

    ## optimizers and loss function
    optimizers_generator = []
    if configs.continue_training:
        optimizer_lm = optim.AdamW(model.parameters(), lr=configs.lr)
        optimizer_lm.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'optimizer_lm.pth')))
        optimizers_generator.append(optimizer_lm)

        optimizer_linear = optim.AdamW(linear.parameters(), lr=configs.lr_linear)
        optimizer_linear.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'optimizer_linear.pth')))
        optimizers_generator.append(optimizer_linear)

        optimizer_lstm = optim.RMSprop(lstm.parameters(), lr=configs.lstm_lr, momentum=0.9)
        optimizer_lstm.load_state_dict(torch.load(os.path.join(configs.model_save_dir, now, 'optimizer_lstm.pth')))
        optimizers_lstm = []
        optimizers_lstm.append(optimizer_lstm)
    
    else:
        optimizer_lm = optim.AdamW(model.parameters(), lr=configs.lr)
        # optimizer_lm = optim.SGD(model.parameters(), lr=configs.lr, momentum=0.9)
        optimizers_generator.append(optimizer_lm)
        optimizer_linear = optim.AdamW(linear.parameters(), lr=configs.lr_linear)
        # optimizer_linear = optim.SGD(linear.parameters(), lr=configs.lr_linear, momentum=0.9)
        optimizers_generator.append(optimizer_linear)

        ## optimizer for lstm
        optimizers_lstm = None
        if configs.train_lstm and configs.use_lstm:
            optimizers_lstm = []
            optimizer_lstm = optim.RMSprop(lstm.parameters(), lr=configs.lstm_lr, momentum=0.9)
            optimizers_lstm.append(optimizer_lstm)
        
        optimizers_predictor = None
        multitask_loss_fn = None
        if configs.multitask:
            optimizers_predictor = []
            if configs.multitask_label == 'raw':
                optimizer_predictor = optim.AdamW(predictor.parameters(), lr=configs.multitask_pred_linear)
            else:
                optimizer_predictor = optim.AdamW([predictor], lr=configs.multitask_pred_linear)
            optimizers_predictor.append(optimizer_predictor)
            if configs.loss_fn == 'MSE':
                multitask_loss_fn = nn.MSELoss(reduction='none')
            else:
                multitask_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            
        
    # LR scheduler
    num_training_steps = len(train_loader) * configs.epochs
    num_warmup_steps = configs.warmup_ratio * num_training_steps
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer_lm, num_warmup_steps, num_training_steps)
    if configs.continue_training:
        scheduler_dir = os.path.join(configs.model_save_dir, now, 'scheduler.pth')
        scheduler.load_state_dict(torch.load(scheduler_dir))


    ## start training
    best_valid_metrics =  {'loss': float('inf')} 
    best_test_metrics =  {'loss': float('inf')} 
    best_metrics_with_valid =  {'loss': float('inf')} 
    max_len_for_gen = []
    train_dl_len = len(train_loader)

    for ep in tqdm(range(configs.start_epoch, configs.epochs), desc="epochs"):
        train_logs, test_logs, valid_logs = [], [], []
        
        ## training
        for idx, batch in enumerate(tqdm(train_loader, desc="training", leave=False)):
            # Original generator_step
            if configs.split_by == 'submission':
                train_log = generator_step(idx, batch, lstm_inputs, model, lstm, linear, optimizers_generator, optimizers_lstm,
                                          configs, train_dl_len=train_dl_len, train=True, scheduler=scheduler, device=device)

            # New generator_step, split by student
            else:
                train_log = generator_student_step(idx, batch, model, lstm, linear, optimizers_generator, optimizers_lstm,
                                                    configs, train_dl_len=train_dl_len, train=True, scheduler=scheduler, device=device, 
                                                    multitask=configs.multitask, predictor=predictor, pred_loss_fn=multitask_loss_fn, optimizers_multitask=optimizers_predictor)
            train_logs.append(train_log)

            # Find the max length of labels in training set only once as a reference for generate() max_length
            if ep == 0 and configs.okt_model == 'meta-llama/Meta-Llama-3-8B-Instruct':
                max_len_for_gen.append(collate_fn.max_length_label)
            
            
            ## save results to wandb
            if configs.log_train_every_itr and configs.log_wandb:
                if (idx+1) % configs.log_train_every_itr == 0:
                    itr_train_logs = aggregate_metrics(train_logs)
                    for key in itr_train_logs:
                        wandb.log({"metrics/train_every_{}_itr/{}".format(configs.log_train_every_itr,key): itr_train_logs[key]})
        
        if ep == 0 and configs.okt_model == 'meta-llama/Meta-Llama-3-8B-Instruct':
            max_label_length = max(max_len_for_gen)

        ## validation
        for idx, batch in enumerate(tqdm(valid_loader, desc="validation", leave=False)):
            if configs.split_by == 'submission':
                valid_log = generator_step(idx, batch, lstm_inputs, model, lstm, linear, configs=configs, train=False, device=device)
            else:
                valid_log = generator_student_step(idx, batch, model, lstm, linear, configs=configs, train=False, device=device, multitask=configs.multitask, 
                                                   predictor=predictor, pred_loss_fn=multitask_loss_fn, optimizers_multitask=optimizers_predictor)

            valid_logs.append(valid_log)
            
        ## testing
        for idx, batch in enumerate(tqdm(test_loader, desc="testing", leave=False)):
            if configs.split_by == 'submission':
                test_log = generator_step(idx, batch, lstm_inputs, model, lstm, linear, configs=configs, train=False, device=device)
            else:
                test_log = generator_student_step(idx, batch, model, lstm, linear, configs=configs, train=False, device=device, multitask=configs.multitask, 
                                                  predictor=predictor, pred_loss_fn=multitask_loss_fn, optimizers_multitask=optimizers_predictor)
            test_logs.append(test_log)
        
        ## logging
        train_logs = aggregate_metrics(train_logs)
        valid_logs = aggregate_metrics(valid_logs)
        test_logs  = aggregate_metrics(test_logs )


        ## log the results and save models
        for key in valid_logs:
            ## only one key (loss) available for OKT
            if key == 'loss':
                if( float(valid_logs[key]) < best_valid_metrics[key] ):
                    best_valid_metrics[key] = float(valid_logs[key])
                    for key_best_metric in best_metrics_with_valid:
                        best_metrics_with_valid[key_best_metric] = float(test_logs[key_best_metric])
                    ## Save the model with lowest validation loss
                    print('Saved at Epoch:', ep)
                    print('Best model stats:', test_logs)
                    if configs.save_model:
                        if configs.log_wandb:
                            wandb.log({"best_model_at_epoch": ep, "best_valid_loss": best_valid_metrics[key]})

                        # torch.save(model, os.path.join(configs.model_save_dir, now, 'model'))
                        ## Save the adapter model for Lora instead of the whole model when using Llama
                        model_dir = os.path.join(configs.model_save_dir, now, 'model')
                        model.save_pretrained(model_dir)
                        torch.save(linear.state_dict(), os.path.join(configs.model_save_dir, now, 'linear'))
                        # torch.save(linear, os.path.join(configs.model_save_dir, now, 'linear'))
                        if configs.use_lstm:
                            # torch.save(lstm, os.path.join(configs.model_save_dir, now, 'lstm'))
                            torch.save(lstm.state_dict(), os.path.join(configs.model_save_dir, now, 'lstm'))
                            optimizer_lstm_dir = os.path.join(configs.model_save_dir, now, 'optimizer_lstm.pth')
                            torch.save(optimizer_lstm.state_dict(), optimizer_lstm_dir)
                        
                        if configs.multitask:
                              if configs.multitask_label == 'granular':
                                  torch.save(predictor, os.path.join(configs.model_save_dir, now, 'predictor.pth'))
                              else:
                                  torch.save(predictor.state_dict(), os.path.join(configs.model_save_dir, now, 'predictor'))
                                  optimizer_predictor_dir = os.path.join(configs.model_save_dir, now, 'optimizer_predictor.pth')
                                  torch.save(optimizer_predictor.state_dict(), optimizer_predictor_dir)

                        scheduler_dir = os.path.join(configs.model_save_dir, now, 'scheduler.pth')
                        torch.save(scheduler.state_dict(), scheduler_dir)

                        optimizer_lm_dir = os.path.join(configs.model_save_dir, now, 'optimizer_lm.pth')
                        torch.save(optimizer_lm.state_dict(), optimizer_lm_dir)

                        optimizer_linear_dir = os.path.join(configs.model_save_dir, now, 'optimizer_linear.pth')
                        torch.save(optimizer_linear.state_dict(), optimizer_linear_dir)
                            


        for key in test_logs:
            if key == 'loss':
                if float(test_logs[key])<best_test_metrics[key]:
                    best_test_metrics[key] = float(test_logs[key])

        ## save results to wandb:
        if configs.log_wandb:
            saved_stats = {}
            for key in train_logs:
                saved_stats["metrics/train/"+key] = train_logs[key]
            for key in valid_logs:
                saved_stats["metrics/valid/"+key] = valid_logs[key]
            for key in test_logs:
                saved_stats["metrics/test/"+key] = test_logs[key]
            for key in best_test_metrics:
                saved_stats["metrics/test/best_"+key] = best_test_metrics[key]
            for key in best_metrics_with_valid:
                saved_stats["metrics/test/best_"+key+"_with_valid"] = best_metrics_with_valid[key]
            saved_stats["epoch"] = ep

            wandb.log(saved_stats)
    
    # # Evaluation post training for code generation on test set and CodeBleu
    # if configs.change_generation_length and len(max_len_for_gen) > 0:
    #     configs.max_new_tokens = max_label_length
    if configs.split_by == 'submission':
        res = evaluate(configs, now, test_set, lstm_inputs, tokenizer, device)
    else:
        res = eval_student(configs, now, test_set, dataset, tokenizer, device)
    
    if configs.log_wandb:
        result = {'codeBLEU': res['codebleu']}
        if configs.multitask:
            if configs.multitask_label != 'granular':
                result['MSE'] = res['MSE']
            else:
                result['Acc'] = res['Acc']
                result['F1'] = res['F1']

        wandb.log(result)

        wandb.finish()


if __name__ == "__main__":
    #torch.set_printoptions(profile="full")
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()
