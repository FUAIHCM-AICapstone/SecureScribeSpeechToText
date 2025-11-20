import glob

import torch
import torchaudio
from tqdm import tqdm


class VietnameseDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, training_params, tokenizer_params, split, args):
        self.names = glob.glob(dataset_path + "/*/" + split + "/*.wav")
        self.vocab_type = tokenizer_params["vocab_type"]
        self.vocab_size = str(tokenizer_params["vocab_size"])
        self.lm_mode = training_params.get("lm_mode", False)
        self.split = split

        if split == "train":
            self.names = self.filter_lengths(training_params["train_audio_max_length"], training_params["train_label_max_length"], args.rank)
        else:
            self.names = self.filter_lengths(training_params["eval_audio_max_length"], training_params["eval_audio_max_length"], args.rank)

    def __getitem__(self, i):
        if self.lm_mode:
            return [torch.load(self.names[i].split(".wav")[0] + "." + self.vocab_type + "_" + self.vocab_size)]
        else:
            return [torchaudio.load(self.names[i])[0], torch.load(self.names[i].split(".wav")[0] + "." + self.vocab_type + "_" + self.vocab_size)]

    def __len__(self):
        return len(self.names)

    def filter_lengths(self, audio_max_length, label_max_length, rank=0):
        if audio_max_length is None or label_max_length is None:
            return self.names

        if rank == 0:
            print("LibriSpeech dataset filtering")
            print(f"Audio maximum length : {audio_max_length} / Label sequence maximum length : {label_max_length}")
            self.names = tqdm(self.names)

        if self.split == "train":
            with open("data/train_wav_names.txt", encoding="utf8") as fr:
                names1 = fr.readlines()
            with open("data/train_agument_wav_names.txt", encoding="utf8") as fr:
                names2 = fr.readlines()
            names = names1 + names2
            print("Total " + str(self.split) + " examples: " + str(len(names)))
            return [name.replace("\n", "") for name in names]
        else:
            with open("data/val_wav_names.txt", encoding="utf8") as fr:
                names = fr.readlines()
            print("Total " + str(self.split) + " examples: " + str(len(names)))
            return [name.replace("\n", "") for name in names]
