
#! pip install tokenizers
#%% Import Statements

from pathlib import Path
from transformers import RobertaTokenizer

from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE 
from tokenizers.pre_tokenizers import Whitespace

# from tokenizers import ByteLevelBPETokenizer
# from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
import os.path as osp

#%% Train Tokenizer
if (not osp.exists('models/BPEtokenizer.json')):
    paths = [str(x) for x in Path("./eo_data/").glob("**/*.txt")]

    # Initialize a tokenizer
    # tokenizer = ByteLevelBPETokenizer()

    # # Customize training
    # tokenizer.train(files=paths, vocab_size=52000, min_frequency=3, special_tokens=[
    #     "<s>",
    #     "<pad>",
    #     "</s>",
    #     "<unk>",
    #     "<mask>"
    # ])

    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(vocab_size=52000,min_frequency=3, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(trainer, paths)
    
    # Save files to disk    
    tokenizer.save('models/BPEtokenizer.json')

#%% Tokenize
tokenizer = Tokenizer.from_file('models/BPEtokenizer.json')
# tokenizer._tokenizer.post_processor = BertProcessing(
#     ("</s>", tokenizer.token_to_id("</s>")),
#     ("<s>", tokenizer.token_to_id("<s>")),
# )
# tokenizer.enable_truncation(max_length=512)
output = tokenizer.encode("Mi estas Julien.😁")
print(output.tokens)

print(output.ids)
# Encoding(num_tokens=7, ...)
# tokens: ['<s>', 'Mi', 'Ġestas', 'ĠJuli', 'en', '.', '</s>']
