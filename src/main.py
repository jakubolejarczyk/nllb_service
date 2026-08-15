from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "facebook/nllb-200-distilled-600M"

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


class TranslationRequest(BaseModel):
    message: str
    source_language: str
    target_language: str


@app.post("/translate")
def translate(request: TranslationRequest):
    tokenizer.src_lang = request.source_language

    inputs = tokenizer(
        request.message,
        return_tensors="pt"
    )

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(
            request.target_language
        )
    )

    translation = tokenizer.batch_decode(
        translated_tokens,
        skip_special_tokens=True
    )[0]

    return {
        "translation": translation
    }
