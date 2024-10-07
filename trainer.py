import torch
from pdb import set_trace
import numpy as np
from torch.utils.checkpoint import checkpoint

def generator_step(idx, batch, lstm_inputs, model, lstm, linear, optimizers=None, optimizers_lstm=None,
                   configs=None, train_dl_len=None, train=True, scheduler=None, device=None):    

    if train:
        if configs.train_okt:
            assert(optimizers != None)
            model.train()
            linear.train()
        if configs.train_lstm and configs.use_lstm:
            assert(optimizers_lstm != None)
            lstm.train()
    else:
        model.eval()
        linear.eval()
        if configs.use_lstm:
            lstm.eval()
        
    # assemble generator input
    generator_inputs_ids, attention_mask, labels, prompt_id_lens, students, timesteps = batch[0], batch[1], batch[2], batch[3], batch[4], batch[5]
    
    generator_inputs_wte, ks = assemble_generator_input(model, lstm, linear, configs, generator_inputs_ids, prompt_id_lens, 
                                                        lstm_inputs, students, timesteps, device, generation=False)

    # forward generator
    if train:
        outputs = model(inputs_embeds=generator_inputs_wte, attention_mask=attention_mask, labels=labels, output_hidden_states=True, return_dict=True)
    else:
        with torch.no_grad():
            outputs = model(inputs_embeds=generator_inputs_wte, attention_mask=attention_mask, labels=labels, output_hidden_states=True, return_dict=True)
    
    # compute stats
    loss = outputs["loss"]

    log = {'loss': loss.cpu().detach()}
        
    # Adding gradient accumulation for training
    if train:
        loss /= configs.accum_iter
        loss.backward()
    
    # optimization
    if train:
        if (idx+1) % configs.accum_iter == 0 or idx == train_dl_len - 1:
            # Training the LM and linear layer for ks alignment with problem token embeddings
            for optimizer in optimizers:
                optimizer.step()
            if configs.use_scheduler:
                scheduler.step()
            for optimizer in optimizers:
                optimizer.zero_grad()
            
            # training the lstm
            if configs.train_lstm and configs.use_lstm:
                assert(optimizers_lstm != None)
                for optimizer in optimizers_lstm:
                    optimizer.step()
                for optimizer in optimizers_lstm:
                    optimizer.zero_grad()

    return log


def generator_student_step(idx, batch, model, lstm, linear, optimizers=None, optimizers_lstm=None,
                           configs=None, train_dl_len=None, train=True, scheduler=None, device=None, 
                           group_size=2, multitask=False, predictor=None, pred_loss_fn=None, optimizers_multitask=None):    
    eps = 1e-8
    if train:
        if configs.train_okt:
            assert(optimizers != None)
            model.train()
            linear.train()
        if configs.train_lstm and configs.use_lstm:
            assert(optimizers_lstm != None)
            lstm.train()
    else:
        model.eval()
        linear.eval()
        if configs.use_lstm:
            lstm.eval()
        
    # assemble generator input
    padded_scores, padded_inputs, padded_input_ids_ls, padded_attention_mask_ls, padded_labels_ls, padded_prompt_id_lens_ls, padded_question_seqs = batch[0][1:].to(device), batch[1][:-1], batch[2][1:], batch[3][1:], batch[4][1:], batch[5][1:], batch[6][1:].to(device)
    
    if configs.okt_model != 'codellama/CodeLlama-7b-Instruct-hf' and configs.okt_model != 'meta-llama/Meta-Llama-3-8B-Instruct' and configs.okt_model != 'Qwen/Qwen1.5-7B':
        generator_input_wte = model.transformer.wte(padded_input_ids_ls) # Shape = [B, T, generator_inputs_ids.shape(1), 768]
    else:
        generator_input_wte = model.base_model.model.model.embed_tokens(padded_input_ids_ls)

    def custom_forward(inputs_embeds, attention_mask, labels):
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        return outputs
    
    range_tensor = torch.arange(padded_input_ids_ls.size(2), device=device).unsqueeze(0).unsqueeze(0) # Shape = [1, 1, max_length]
    range_tensor = range_tensor.repeat(padded_input_ids_ls.size(0), padded_input_ids_ls.size(1), 1) # Shape = [T, B, max_length]
    mask_tensor = (range_tensor >= padded_prompt_id_lens_ls.unsqueeze(-1)) # Shape = [T, B, max_length]

    if configs.use_lstm:
        ks, hidden = lstm(padded_inputs)
        ks = linear(ks)

        ks = ks.unsqueeze(2).repeat(1, 1, padded_input_ids_ls.size(2), 1) # (T, B, max_length, hidden_dim)
        ks[mask_tensor] = torch.zeros(ks.size(-1), device=device) # Shape = [T, B, max_length, 4096]
        
        generator_input_wte = torch.add(generator_input_wte, ks) # Shape = [T, B, max_length, 4096]

    T, B, max_length, D = generator_input_wte.shape
    generator_input_wte = generator_input_wte.view((T * B), max_length, D)
    padded_attention_mask = padded_attention_mask_ls.reshape((T * B), -1)
    padded_label = padded_labels_ls.reshape((T * B), max_length)

    input_wte_groups = torch.split(generator_input_wte, group_size)
    attention_mask_groups = torch.split(padded_attention_mask, group_size)
    label_groups = torch.split(padded_label, group_size)

    if multitask:
        if configs.multitask_label == 'raw':
            padded_scores = torch.unsqueeze(padded_scores, -1)
        else:
            padded_ques_seqs = torch.unsqueeze(padded_question_seqs, -1)
            padded_ques_seqs = padded_ques_seqs.reshape((T * B), -1)
            ques_seqs_groups = torch.split(padded_ques_seqs, group_size)


        padded_scores = padded_scores.reshape((T * B), -1)
        score_groups = torch.split(padded_scores, group_size)

        # Mask hidden states for label embeddings
        padded_mask = mask_tensor.reshape((T * B), -1)
        mask_groups = torch.split(padded_mask, group_size)


    cum_loss = 0.0
    cum_cnt = 0
    pred_cum_loss = 0.0
    pred_cnt = 0

    pred_total = torch.tensor([]).to(device)
    gt_total = torch.tensor([]).to(device)
    logits_total = torch.tensor([]).to(device)

    for i in range(len(input_wte_groups)):
        input_wte_sub = input_wte_groups[i]
        attention_mask_sub = attention_mask_groups[i]
        label_sub = label_groups[i]
        if multitask:
            mask_sub = mask_groups[i]

        # forward generator
        if train:
            if (configs.okt_model == 'codellama/CodeLlama-7b-Instruct-hf' or configs.okt_model == 'meta-llama/Meta-Llama-3-8B-Instruct' or configs.okt_model == 'Qwen/Qwen1.5-7B') and configs.first_ast_convertible:
                outputs = model(inputs_embeds=input_wte_sub, attention_mask=attention_mask_sub, labels=label_sub, output_hidden_states=True, return_dict=True)
            else:
                outputs = checkpoint(custom_forward, input_wte_sub, attention_mask_sub, label_sub, use_reentrant=False)
        
            if multitask:
                    hidden_states = outputs['hidden_states'][-1]

                    # Question emebedding 
                    mask_expand = torch.unsqueeze(mask_sub, -1)
                    hidden_states_question = hidden_states * ~mask_expand
                    pooled_out = hidden_states_question.sum(dim=1)
                    ques_cnt = torch.sum(~mask_expand, dim=1)
                    pooled_out = pooled_out / (ques_cnt + eps)

                    if configs.multitask_label == 'raw':
                        logits = predictor(pooled_out)
                    else:
                        ques_seq_sub = ques_seqs_groups[i].squeeze(-1)
                        pooled_out = pooled_out.unsqueeze(1)
                        model_weight = predictor[ques_seq_sub]
                        logits = torch.matmul(pooled_out, model_weight)
                        logits = logits.squeeze(1)

        else:
            with torch.no_grad():
                outputs = model(inputs_embeds=input_wte_sub, attention_mask=attention_mask_sub, labels=label_sub, output_hidden_states=True, return_dict=True)
                if multitask:
                    hidden_states = outputs['hidden_states'][-1]

                    # Question emebedding
                    mask_expand = torch.unsqueeze(mask_sub, -1)
                    hidden_states_question = hidden_states * ~mask_expand
                    pooled_out = hidden_states_question.sum(dim=1)
                    ques_cnt = torch.sum(~mask_expand, dim=1)
                    pooled_out = pooled_out / (ques_cnt + eps)
                    
                    if configs.multitask_label == 'raw':
                        logits = predictor(pooled_out)
                    else:
                        ques_seq_sub = ques_seqs_groups[i].squeeze(-1)
                        pooled_out = pooled_out.unsqueeze(1)
                        model_weight = predictor[ques_seq_sub]
                        logits = torch.matmul(pooled_out, model_weight)
                        logits = logits.squeeze(1)
                    

        # compute stats
        loss = outputs["loss"]
        valid_token_cnt = attention_mask_sub.sum()
        cum_loss += loss * valid_token_cnt
        cum_cnt += valid_token_cnt

        if multitask:
            score_sub = score_groups[i]
            if configs.loss_fn == 'MSE':
                logits = torch.sigmoid(logits)
            
            if configs.multitask_label == 'granular':
                gt_total = torch.cat((gt_total, score_sub), 0)
                pred = (torch.sigmoid(logits) > 0.5) * 1
                pred_total = torch.cat((pred_total, pred), 0)
                logits_total = torch.cat((logits_total, logits), 0)
                
                non_padding_mask = score_sub.ne(-100)
                pred_loss_sub = pred_loss_fn(logits[non_padding_mask], score_sub[non_padding_mask]).sum()
                pred_cum_loss += pred_loss_sub
                pred_cnt += non_padding_mask.sum()

            else:
                pred_loss_sub = pred_loss_fn(logits[score_sub != -100], score_sub[score_sub != -100]).sum()
                pred_cum_loss += pred_loss_sub
                pred_cnt += logits[score_sub != -100].shape[-1]

    if multitask:
        pred_cum_loss = pred_cum_loss / pred_cnt
    
    cum_loss = cum_loss / cum_cnt

    total_loss = cum_loss + pred_cum_loss
    if multitask: 
        back_loss = cum_loss + pred_cum_loss
    else:
        back_loss = total_loss
        
    # Adding gradient accumulation for training
    if train:
        back_loss /= configs.accum_iter
        back_loss.backward()
    

    # optimization
    if train:
        if (idx+1) % configs.accum_iter == 0 or idx == train_dl_len - 1:
            # Training the LM and linear layer for ks alignment with problem token embeddings
            for optimizer in optimizers:
                optimizer.step()
            if configs.use_scheduler:
                scheduler.step()
            for optimizer in optimizers:
                optimizer.zero_grad()
            
            # training the lstm
            if configs.train_lstm and configs.use_lstm:
                assert(optimizers_lstm != None)
                for optimizer in optimizers_lstm:
                    optimizer.step()
                for optimizer in optimizers_lstm:
                    optimizer.zero_grad()
            
            if multitask:
                for optimizer in optimizers_multitask:
                    optimizer.step()
                for optimizer in optimizers_multitask:
                    optimizer.zero_grad()

    log = {'loss': total_loss.cpu().detach(), 'weighted_loss': back_loss.cpu().detach()}
    if configs.multitask:
        log['generator_loss'] = cum_loss.cpu().detach()
        log['predictor_loss'] = pred_cum_loss.cpu().detach()
        if configs.multitask_label == 'granular':
            pred_res = pred_total[gt_total != -100].detach().cpu() == gt_total[gt_total != -100].detach().cpu()
            log['acc'] = pred_res
            log['auc'] = {'logits': logits_total[gt_total != -100].detach().cpu(), 'scores': gt_total[gt_total != -100].detach().cpu()}
        
    return log



def get_knowledge_states_for_generator(lstm, lstm_inputs, students, timesteps, configs, device, generation=False):
    '''
    used during ***inference (generation) time*** to get a student's knowledge state
    '''
    ks = None

    if configs.use_lstm:
        # get lstm inputs
        lstm_ins = [lstm_inputs[s] for s in students]
        
        # TODO p2: vectorize
        max_len = max(len(i) for i in lstm_ins)
        padded_lstm_ins = [i + [torch.zeros(i[0].shape[0])]*(max_len - len(i)) for i in lstm_ins]
        padded_lstm_ins = torch.stack([torch.stack(x, dim=0) for x in padded_lstm_ins], dim=1).float() # Shape = [T, B, D_bar]
        # Get student knowledge states
        if( configs.train_lstm and not generation ):
            out, hidden = lstm(padded_lstm_ins.to(device)) # Shape = [T, B, D_bar], D_bar = lstm_hid_dim
        else:
            with torch.no_grad():
                out, hidden = lstm(padded_lstm_ins.to(device)) # Shape = [T, B, D_bar], D_bar = lstm_hid_dim
        ks = out[timesteps, list(range(out.shape[1])), :] # Extract the hidden states -> shape = [B, D_bar]
    
    return ks


def assemble_generator_input(model, lstm, linear, configs, generator_inputs_ids, prompt_id_lens, 
                                lstm_inputs, students, timesteps, device, generation=False):
    '''
    linear: linear transform the knowledge state before adding in with the generator input
    '''
    # compute generator embeddings for the batch
    if configs.okt_model != 'codellama/CodeLlama-7b-Instruct-hf' and configs.okt_model != 'meta-llama/Meta-Llama-3-8B-Instruct' and configs.okt_model != 'Qwen/Qwen1.5-7B':
        generator_input_wte = model.transformer.wte(generator_inputs_ids) # Shape = [B, generator_inputs_ids.shape(1), 768]
    else:
        generator_input_wte = model.base_model.model.model.embed_tokens(generator_inputs_ids) # Shape = [B, generator_inputs_ids.shape(1), 4096] 
    
    # get knowledge states
    ks = get_knowledge_states_for_generator(lstm, lstm_inputs, students, timesteps, configs, device, generation)
    # Add linear transformation of student knowledge state with only prompt tokens

    ks = linear(ks) # Shape = [B, 768] if gpt-2 else [B, 4096]
    ks = ks.unsqueeze(1).repeat(1, generator_input_wte.size(1), 1) # Shape = [B, T, 768], T refers to max_input_length
    range_tensor = torch.arange(generator_inputs_ids.size(1), device=device).unsqueeze(0) # Shape = [1, T]
    range_tensor = range_tensor.repeat(prompt_id_lens.size(0), 1) # Shape = [B, T]
    mask_tensor = (range_tensor >= prompt_id_lens.unsqueeze(-1)) # Shape = [B, T]
    ks[mask_tensor] = torch.zeros(ks.size(-1), device=device) # Shape = [B, T, 768]
    generator_input_wte = torch.add(generator_input_wte, ks) # Shape = [B, T, 768]

    return generator_input_wte, ks

def predict_granular_step(idx, batch, lstm, granular_model, transition_model=None, optimizers_granular=None, optimizers_lstm=None, optimizers_trans=None,
                   configs=None, loss_fn=None, train_dl_len=None, train=True, scheduler=None, device=None, eval=False):
    if train:
        lstm.train()
    else:
        lstm.eval()
        
    padded_granular_cor, padded_inputs, padded_embeddings, padded_question_seqs = batch[0][1:].to(device), batch[1][:-1].to(device), batch[2][1:].to(device), batch[3][1:].to(device)
    
    if configs.use_lstm:
        ks, hidden = lstm(padded_inputs)
        combined_ks = torch.cat((ks, padded_embeddings), dim=-1)
    else:
        combined_ks = padded_embeddings

    if transition_model:
        combined_ks = transition_model(combined_ks)

    batch_combined_ks = torch.transpose(combined_ks, 0, 1)
    batch_combined_ks = torch.unsqueeze(batch_combined_ks, 2)

    batch_question_seqs = torch.transpose(padded_question_seqs, 0, 1)
    model_weight = granular_model[batch_question_seqs]

    if train:
        logits = torch.matmul(batch_combined_ks, model_weight)
    else:
        with torch.no_grad():
            logits = torch.matmul(batch_combined_ks, model_weight)

    logits = torch.squeeze(logits, 2)
    label = torch.transpose(padded_granular_cor, 0, 1)
    loss = loss_fn(logits[label != -100], label[label != -100]).sum()

    if train: 
        (loss / len(label[label != -100])).backward()

        optimizers_granular.step()
        if configs.use_lstm:
            optimizers_lstm.step()
        if transition_model:
            optimizers_trans.step()
            optimizers_trans.zero_grad()

        optimizers_granular.zero_grad()
        if configs.use_lstm:
            optimizers_lstm.zero_grad()

        if configs.use_scheduler:
            for scheduler_i in scheduler:
                scheduler_i.step(loss / len(label[label != -100]))

    pred = (torch.sigmoid(logits) > 0.5) * 1

    # # Majority baseline:
    # logits = torch.ones(label.shape).to(device)
    # pred = torch.ones(label.shape).to(device)

    # # Random baseline:
    # logits = torch.randn((label.shape)).to(device)
    # pred = (torch.sigmoid(logits) > 0.5) * 1

    pred_res = pred[label != -100].detach().cpu() == label[label != -100].detach().cpu()

    log = {'loss': loss.detach().cpu().true_divide(len(label[label != -100])), 'acc': pred_res, 'auc': {'logits': logits[label != -100].detach().cpu(), 'scores': label[label != -100].detach().cpu()}}


    if eval:
        non_padded_mask = (label != -100)
        test_case_cnt = non_padded_mask.sum(-1)
        converted_label = label * non_padded_mask
        converted_score = converted_label.sum(-1)
        label_scores = converted_score / test_case_cnt

        converted_pred = pred * non_padded_mask
        converted_pred_score = converted_pred.sum(-1)
        pred_scores = converted_pred_score / test_case_cnt

        label_scores = label_scores[~label_scores.isnan()]
        pred_scores = pred_scores[~pred_scores.isnan()]

        mse = torch.square(torch.subtract(label_scores, pred_scores)).mean().detach().cpu()
        log['MSE'] = mse

        match_pred = torch.where(label == -100, torch.tensor(-100), pred)

        split_groups = torch.split(match_pred, 2)
        split_trans = [torch.transpose(i, 0, 1) for i in split_groups]

        reorg_res = []
        for subbatch in split_trans:
            T, B, D = subbatch.shape
            combine_pred = subbatch.reshape((T*B), -1)
            reorg_res.append(combine_pred.tolist())

        filt_res = []
        cnt = 0 
        for pairs in reorg_res:
            for sample in pairs:
                filt = [i for i in sample if i != -100]
                if len(filt) > 0:
                    filt_res.append(filt)
                    cnt += 1

        return log, filt_res

    return log
