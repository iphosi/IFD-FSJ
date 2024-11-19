import torch


device = "cuda" if torch.cuda.is_available() else "cpu"


def get_perplexity_and_embedding_whole_text(tokenizer, model, text, max_length):
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    with torch.no_grad(): 
        outputs = model(input_ids, labels=input_ids.contiguous())
    loss = outputs.loss
    perplexity = torch.exp(loss)

    return perplexity.to('cpu').item(), loss.to('cpu').item()


def get_perplexity_and_embedding_part_text(tokenizer, model, text, target_span, max_length):
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    start_index = text.rfind(target_span)
    start_token = len(tokenizer.encode(text[:start_index]))
    end_token = input_ids.shape[1]

    labels = input_ids.clone()
    labels[0, :start_token] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=labels)

    loss = outputs.loss
    perplexity = torch.exp(loss)

    return perplexity.to('cpu').item(), loss.to('cpu').item()
