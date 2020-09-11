import logging

import pandas as pd
from simpletransformers.seq2seq import Seq2SeqModel

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

model_args = {
    "reprocess_input_data": True,
    "max_seq_length": 256,
    "use_multiprocessing": True,
    "max_length": 256,
    "use_cuda": True,
}

model = Seq2SeqModel(
    encoder_decoder_type="marian",
    encoder_decoder_name="Helsinki-NLP/opus-mt-de-en",
    args=model_args,
)

dev_data = pd.read_csv('dev.csv')

src = dev_data.text

predictions = model.predict(src)

for en, de in zip(src, predictions):
    print("-------------")
    print(en)
    print(de)
    print()