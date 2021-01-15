import torch
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from pathlib import Path
from transformers import DataCollatorForLanguageModeling
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
# Tutorial from https://colab.research.google.com/github/huggingface/blog/blob/master/notebooks/01_how_to_train.ipynb#scrollTo=BzMqR-dzF4Ro 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# tokenizer = RobertaTokenizerFast.from_pretrained("./EsperBERTo", max_len=512)
# tokenizer = ByteLevelBPETokenizer("models/esperberto-vocab.json","models/esperberto-merges.txt") # ? This actually doesn't work. You will get an error saying tokenizer is not callable. 

tokenizer = PreTrainedTokenizerFast(tokenizer_file="models/byte-level.tokenizer.json")
mlm=False

config = RobertaConfig(
    vocab_size=52_000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

# Training from scratch 
model = RobertaForMaskedLM(config=config)
model.num_parameters()

paths = [str(x) for x in Path("eo_data/").glob("**/*.txt")]
# Build the dataset 

dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path="eo_data/shuff-orig/eo/eo.txt",block_size=128)

# mlm = mask modeling language 
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=mlm, mlm_probability=0.15
    )



from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="model/EsperBERTo-small",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    prediction_loss_only=True,
)

trainer.train()