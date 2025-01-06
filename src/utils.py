import torch


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_perplexity_and_embedding_whole_text(tokenizer, model, text, max_length, window_size=-1, stride=4):
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, add_special_tokens=False, max_length=max_length).to(device)

    loss = torch.tensor([0]).to(device)

    if window_size == -1:
        with torch.no_grad(): 
            outputs = model(input_ids, labels=input_ids.contiguous())
        loss = outputs.loss
    else:
        for i in range(0, input_ids.size(-1), stride):
            windowed_input_ids = input_ids[:, i:i+window_size]
            with torch.no_grad(): 
                outputs = model(windowed_input_ids, labels=windowed_input_ids.contiguous())
            if not torch.isnan(outputs.loss):
                loss = max(outputs.loss, loss)
    
    perplexity = torch.exp(loss).to("cpu").item()
    loss = loss.to("cpu").item()

    return perplexity, loss


def get_perplexity_and_embedding_part_text(tokenizer, model, text, target_span, max_length):
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, add_special_tokens=False, max_length=max_length).to(device)
    
    target_ids = tokenizer.encode(target_span, return_tensors="pt", truncation=True, add_special_tokens=False, max_length=max_length).to(device)

    end_token = input_ids.shape[1] - 1
    
    for i in range(input_ids.shape[1] - 1, -1, -1):
        if target_ids[0].tolist() == input_ids[0, i - target_ids.shape[1] + 1:i + 1].tolist():
            end_token = i
            break
        
    start_token = end_token - target_ids.shape[1] + 1
    
    assert tokenizer.decode(input_ids[0, start_token:end_token + 1]) == target_span

    labels = input_ids.clone()
    labels[0, :start_token] = -100
    labels[0, end_token + 1:] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)

    loss = outputs.loss.to("cpu").item()
    perplexity = torch.exp(loss).to("cpu").item()

    return perplexity, loss

