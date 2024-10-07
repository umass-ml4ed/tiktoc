import os
from datetime import datetime

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import transformers
import hydra
from omegaconf import OmegaConf

from data_loader import *
from model import *
from trainer import *
from utils import *
from eval import *
from pdb import set_trace
from test_case_check_update import test_case_check, uniq_test_construct, handle_uniq_test_exception

@hydra.main(version_base=None, config_path=".", config_name="configs_granulardkt")
def main(configs):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(now)

    set_random_seed(configs.seed)

    # Test on smaller fraction of dataset
    if configs.testing:
        configs.epochs = 2
        configs.log_wandb = False
        # configs.save_model = False
        configs.batch_size = 4
    
    # Use wandb to track experiment
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

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # question_no_map is used to construct multiple linear head
    all_good_test, good_test_case = test_case_check()
    question_input_dict = uniq_test_construct(good_test_case)
    question_input_dict = handle_uniq_test_exception(question_input_dict)
    question_ids = [1, 3, 5, 12, 13, 17, 20, 21, 22, 24, 25, 34, 37, 39, 40, 46, 71]
    question_no_map = {question_ids[i]:i for i in range(len(question_ids))}


    train_set, valid_set, test_set, dataset, students = read_granular_data(configs)

    collate_fn = CollateForGranularDKT(configs, question_input_dict, question_no_map)
    _, train_loader = make_dataloader(train_set, dataset, collate_fn, configs, split_by_student=True, granular=True)
    _, valid_loader = make_dataloader(valid_set, dataset, 
                                        collate_fn=collate_fn, configs=configs, 
                                        do_lstm_dataset=True, train=False, split_by_student=True, granular=True)
    _, test_loader  = make_dataloader(test_set , dataset, 
                                        collate_fn=collate_fn, configs=configs, 
                                        do_lstm_dataset=True, train=False, split_by_student=True, granular=True)


    # Create model
    lstm = create_lstm_model(configs, device)
    transition_model = None
    granular_model = create_granular_model(configs, device)


    # Create optimizer and scheduler
    optimizer_lstm = None
    if configs.use_lstm:
        optimizer_lstm = optim.AdamW(lstm.parameters(), lr=configs.lstm_lr)
        # optimizer_lstm = optim.RMSprop(lstm.parameters(), lr=configs.lstm_lr, momentum=0.9)
    optimizer_trans = None
    optimizer_granular = optim.AdamW([granular_model], lr=configs.granular_lr)

    if configs.use_transition_model:
        transition_model = create_transition_layer(configs, device)
        optimizer_trans = optim.AdamW(transition_model.parameters(), lr=configs.trans_lr)

    schedulers = []
    if configs.use_scheduler: 
        if configs.use_lstm:
            if configs.use_scheduler_lstm:
                scheduler1 = ReduceLROnPlateau(optimizer_lstm, 'min', factor=configs.scheduler_lstm_factor, patience=5)
                schedulers.append(scheduler1)
        if configs.use_scheduler_classifier:
            scheduler2 = ReduceLROnPlateau(optimizer_granular, 'min', factor=configs.scheduler_granular, patience=5)
            schedulers.append(scheduler2)
        if configs.use_transition_model:
            scheduler3 = ReduceLROnPlateau(optimizer_trans, 'min', factor=configs.scheduler_trans, patience=5)
            schedulers.append(scheduler3)

    # Define loss function
    loss_function = nn.BCEWithLogitsLoss(reduction='none')

    # Start training
    best_valid_metrics = {'loss': float('inf')}
    best_test_metrics = {'loss': float('inf')}
    best_metrics_with_valid = {'loss': float('inf')}

    for ep in tqdm(range(configs.epochs), desc='epochs'):
        train_logs, valid_logs, test_logs = [], [], []

        for idx, batch in enumerate(tqdm(train_loader, desc="training", leave=False)):
            train_log = predict_granular_step(idx, batch, lstm, granular_model, transition_model=transition_model, optimizers_granular=optimizer_granular, optimizers_lstm=optimizer_lstm, 
                optimizers_trans=optimizer_trans, configs=configs, loss_fn=loss_function, train_dl_len=None, train=True, scheduler=schedulers, device=device, eval=False)
            
        
            train_logs.append(train_log)

            if configs.log_train_every_itr and configs.log_wandb:
                if (idx+1) % configs.log_train_every_itr == 0:
                    itr_train_logs = aggregate_metrics(train_logs)
                    # print('train logs:', itr_train_logs)
                    for key in itr_train_logs:
                        wandb.log({"metrics/train_every_{}_itr/{}".format(configs.log_train_every_itr,key): itr_train_logs[key]})


        # validation
        for idx, batch in enumerate(tqdm(valid_loader, desc='validation', leave=False)):
            valid_log = predict_granular_step(idx, batch, lstm, granular_model, transition_model=transition_model, optimizers_trans=optimizer_trans,
                            configs=configs, loss_fn=loss_function, train_dl_len=None, train=False, scheduler=None, device=device, eval=False)
        
            valid_logs.append(valid_log)

        
        # testing
        for idx, batch in enumerate(tqdm(test_loader, desc='testing', leave=False)):
            test_log = predict_granular_step(idx, batch, lstm, granular_model, transition_model=transition_model, optimizers_trans=optimizer_trans, 
                            configs=configs, loss_fn=loss_function, train_dl_len=None, train=False, scheduler=None, device=device, eval=False)
        
            test_logs.append(test_log)

        train_logs = aggregate_metrics(train_logs)
        valid_logs = aggregate_metrics(valid_logs)
        test_logs  = aggregate_metrics(test_logs)

        # log results and save model
        for key in valid_logs:
            if key == 'loss':
                if float(valid_logs[key]) < best_valid_metrics[key]:
                    best_valid_metrics[key] = float(valid_logs[key])
                    for key_best_metric in best_metrics_with_valid:
                        best_metrics_with_valid[key_best_metric] = float(test_logs[key_best_metric])

                    print('Saved at Epoch:', ep)
                    if configs.save_model:
                        if configs.log_wandb:
                            wandb.log({"best_model_at_epoch": ep, "best_valid_loss": best_valid_metrics[key]})
                        
                        torch.save(lstm.state_dict(), os.path.join(configs.model_save_dir, now, 'lstm'))
                        torch.save(granular_model, os.path.join(configs.model_save_dir, now, 'granular_model.pth'))
                        if transition_model:
                            torch.save(transition_model.state_dict(), os.path.join(configs.model_save_dir, now, 'transition'))

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
    
    # Evaluate the final metrics
    res = eval_granular(configs, test_loader, device, now, loss_function)
    print('accuracy:', res['acc'])
    print('auc:', res['auc'])
    print('MSE:', res['MSE'])
    print('F1:', res['F1'])

    if configs.log_wandb:
        wandb.log({'Accuracy': res['acc'], 'AUC': res['auc'], 'MSE': res['MSE'], 'F1': res['F1']})
        wandb.finish()


if __name__ == "__main__":
    main()