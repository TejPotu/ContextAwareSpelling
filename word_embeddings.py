# importing libraries
import random
import torch
from transformers import BertTokenizer, BertModel
from transformers import GPT2Tokenizer, GPT2Model
from sklearn.metrics.pairwise import cosine_similarity

def get_mispelled_words(file_pth):
    misspelled_words = []
    correct_spellings = []
    with open(file_pth, 'r') as file:
        for line in file:
            correct, misspelled = line.strip().split('\t')
            misspelled_words.append(misspelled)
            correct_spellings.append(correct)
    return misspelled_words, correct_spellings

def get_gpt_embeddings(text, model, tokenizer):
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)

    input_ids = tokenizer(text)['input_ids']
    print(input_ids)
    input_ids_tensor = torch.tensor([input_ids])
    with torch.no_grad():
        model_output = model(input_ids_tensor)
        last_hidden_states = model_output.last_hidden_state
    return last_hidden_states
    

def get_bert_embeddings(text, model, tokenizer):
    # Tokenize the text
    input = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    # Get the BERT model embeddings
    with torch.no_grad():
        model_output = model(**input)
        embeddings = model_output.last_hidden_state
    return embeddings

def get_word_embedding(text, word, model, tokenizer):
    # Returns word embedding for a given word in a text
    inputs = tokenizer(text, return_tensors="pt")
    word_tokens = tokenizer.tokenize(word)
    word_ids = tokenizer.convert_tokens_to_ids(word_tokens)
    token_ids = inputs['input_ids'][0].tolist()  # Convert to list to handle more easily
    # Finding the first complete match of the word tokens in the input IDs
    for i in range(len(token_ids)):
        if token_ids[i:i+len(word_tokens)] == word_ids:
            word_index = i
            break

    # Get the embeddings from BERT
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state

    # Extract the embeddings for the word (average subword embeddings if needed)
    word_embeddings = embeddings[0, word_index:word_index+len(word_tokens)].mean(dim=0)
    
    return word_embeddings

def get_cosine_similarity(embeddings_1, embeddings_2):
    similarity = cosine_similarity(embeddings_1, embeddings_2)
    return similarity

def main():
	# Set a random seed
    random_seed = 42
    random.seed(random_seed)

    # Set a random seed for PyTorch (for GPU as well)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Load the pre-trained BERT model and tokenizer
    model = BertModel.from_pretrained("bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load the pre-trained GPT-2 model and tokenizer
    tokenizer_gpt = GPT2Tokenizer.from_pretrained('gpt2')
    model_gpt = GPT2Model.from_pretrained('gpt2')
    model_gpt1 = GPT2Model.from_pretrained('gpt2')

    # Encode the text
    misspelled_words, correct_spellings = get_mispelled_words("misspelled_words.txt")
    text1 = "The hotel staff went out of their way to accomodate the guests."
    text2 = "The hotel staff went out of their way to accommodate the guests."

    # Get the BERT embeddings
    embeddings1 = get_bert_embeddings(text1, model, tokenizer)
    embeddings2 = get_bert_embeddings(text2, model, tokenizer)

    print(embeddings1.shape) # torch.Size([1, 102, 768])

    # compute the average of the embeddings
    embeddings1 = torch.mean(embeddings1, dim=1)
    embeddings2 = torch.mean(embeddings2, dim=1)

    # Compute the cosine similarity of each pair of embeddings
    similarity = get_cosine_similarity(embeddings1, embeddings2)
    print(similarity)
    
    # Get the embeddings for the misspelled words and correct spellings
    #embeddings_misspelled = []
    #embeddings_correct = []
    # similarity = []
    # for misspelled, correct in zip(misspelled_words, correct_spellings):
    #     embeddings_misspelled = get_bert_embeddings(misspelled, model, tokenizer)
    #     embeddings_correct = get_bert_embeddings(correct, model, tokenizer)
    #     # Compute the average of the embeddings
    #     embeddings_misspelled = torch.mean(embeddings_misspelled, dim=1)
    #     embeddings_correct = torch.mean(embeddings_correct, dim=1)

    #     # Compute the cosine similarity of each pair of embeddings
    #     similarity.append(get_cosine_similarity(embeddings_misspelled, embeddings_correct))
        
    
    #print(similarity)

    # Get the GPT-2 embeddings
    embeddings_gpt1 = get_gpt_embeddings("accomodate", model_gpt, tokenizer_gpt)
    embeddings_gpt2 = get_gpt_embeddings("accommodate", model_gpt1, tokenizer_gpt)

    # Compute the average of the embeddings
    embeddings_gpt1 = torch.mean(embeddings_gpt1, dim=1)
    embeddings_gpt2 = torch.mean(embeddings_gpt2, dim=1)

    print(embeddings_gpt1.shape) # torch.Size([1, 768])

    similarity_gpt = get_cosine_similarity(embeddings_gpt1, embeddings_gpt2)
    print("gpt 2 embeddings similarity: ", similarity_gpt)

    text1 = "She walked slowly down the grocery store aisle, comparing prices and ingredients of different cereal brands."
    text2 = "She walked slowly down the grocery store isle, comparing prices and ingredients of different cereal brands."

    # Get embeddings for the specific words
    embedding1 = get_word_embedding(text1, "aisle", model, tokenizer)
    embedding2 = get_word_embedding(text2, "isle", model, tokenizer)

    # Reshape embeddings to match expected input for cosine_similarity ([1, -1] for single vector)
    embedding1 = embedding1.unsqueeze(0)
    embedding2 = embedding2.unsqueeze(0)

    # Compute the cosine similarity
    similarity = cosine_similarity(embedding1, embedding2)

    # Print the similarity
    print("Cosine similarity between 'accomodate' and 'accommodate':", similarity)





if __name__ == "__main__":
    main()