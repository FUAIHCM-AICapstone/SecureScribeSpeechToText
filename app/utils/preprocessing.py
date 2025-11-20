# Other
import glob
import os

# SentencePiece
import sentencepiece as spm
import torch
from tqdm import tqdm


def collate_fn_pad(batch):
    # Regular Mode
    if len(batch[0]) == 2:
        # Sorting sequences by lengths
        sorted_batch = sorted(batch, key=lambda x: x[0].shape[1], reverse=True)

        # Pad data sequences
        data = [item[0].squeeze() for item in sorted_batch]
        data_lengths = torch.tensor([len(d) for d in data], dtype=torch.long)
        data = torch.nn.utils.rnn.pad_sequence(data, batch_first=True, padding_value=0)

        # Pad labels
        target = [item[1] for item in sorted_batch]
        target_lengths = torch.tensor([t.size(0) for t in target], dtype=torch.long)
        target = torch.nn.utils.rnn.pad_sequence(target, batch_first=True, padding_value=0)

        return data, target, data_lengths, target_lengths

    # LM Mode
    elif len(batch[0]) == 1:
        # Sort Batch
        sorted_batch = sorted(batch, key=lambda x: x[0].size(0), reverse=True)
        sorted_batch = [item[0] for item in sorted_batch]

        # Create Labels
        x = torch.nn.utils.rnn.pad_sequence(sorted_batch, batch_first=True, padding_value=0)
        x_len = torch.tensor([t.size(0) for t in sorted_batch], dtype=torch.long)
        y = [torch.cat([item, item.new_zeros(1)]) for item in sorted_batch]
        y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=-1)

        return x, x_len, y

    else:
        raise Exception("Batch Format Error")


def create_tokenizer(training_params, tokenizer_params):
    txt_files = glob.glob(training_params["training_dataset_path"] + "/*/" + "val/*.txt") + glob.glob(training_params["training_dataset_path"] + "/*/" + "train/*.txt")

    # LibriSpeech Dataset
    if training_params["training_dataset"] == "LibriSpeech":
        # Corpus File Path
        corpus_path = training_params["training_dataset_path"] + training_params["training_dataset"] + "_corpus.txt"

        # Create Corpus File
        if not os.path.isfile(corpus_path):
            print("Create Corpus File")
            text_totals = []
            for file_path in tqdm(txt_files):
                for line in open(file_path, encoding="utf8").readlines():
                    text = " ".join(line.split())
                    if len(text.split()) > 1:
                        text_totals.append(text)
            text_totals = list(set(text_totals))
            with open(corpus_path, "w", encoding="utf8") as corpus_file:
                for text in text_totals:
                    corpus_file.write(text.lower() + "\n")

        # Train Tokenizer
        print("Training Tokenizer")
        spm.SentencePieceTrainer.train(input=training_params["training_dataset_path"] + training_params["training_dataset"] + "_corpus.txt", model_prefix=tokenizer_params["tokenizer_path"].split(".model")[0], vocab_size=tokenizer_params["vocab_size"], character_coverage=1.0, model_type=tokenizer_params["vocab_type"], bos_id=-1, eos_id=-1, unk_surface="")
        print("Training Done")
