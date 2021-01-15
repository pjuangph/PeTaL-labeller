from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="./EsperBERTo",
    tokenizer="./EsperBERTo"
)

# The sun <mask>.
# =>

fill_mask("La suno <mask>.")

fill_mask("Jen la komenco de bela <mask>.")
