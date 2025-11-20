import torch
import torch.nn as nn

# CTC Decode Beam Search
from pyctcdecode import build_ctcdecoder

# Encoders
from app.models.encoders import ConformerEncoder, ConformerEncoderInterCTC

# Losses
from app.models.losses import LossCTC, LossInterCTC

# Base Model
from app.models.model import Model


class ModelCTC(Model):
    def __init__(self, encoder_params, tokenizer_params, training_params, decoding_params, name):
        super(ModelCTC, self).__init__(tokenizer_params, training_params, decoding_params, name)

        # Encoder
        if encoder_params["arch"] == "Conformer":
            self.encoder = ConformerEncoder(encoder_params)
        else:
            raise Exception("Unknown encoder architecture:", encoder_params["arch"])

        # FC Layer
        self.fc = nn.Linear(encoder_params["dim_model"][-1] if isinstance(encoder_params["dim_model"], list) else encoder_params["dim_model"], tokenizer_params["vocab_size"])

        # Criterion
        self.criterion = LossCTC()

        # Compile
        self.compile(training_params)

    def forward(self, batch):
        # Unpack Batch
        x, _, x_len, _ = batch

        # Forward Encoder (B, Taud) -> (B, T, Denc)
        logits, logits_len, attentions = self.encoder(x, x_len)

        # FC Layer (B, T, Denc) -> (B, T, V)
        logits = self.fc(logits)

        return logits, logits_len, attentions

    def distribute_strategy(self, rank):
        super(ModelCTC, self).distribute_strategy(rank)

        self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
        self.encoder = torch.nn.parallel.DistributedDataParallel(self.encoder, device_ids=[self.rank])
        self.fc = torch.nn.parallel.DistributedDataParallel(self.fc, device_ids=[self.rank])

    def load_encoder(self, path):
        # Load Encoder Params
        checkpoint = torch.load(path, map_location=next(self.parameters()).device, weights_only=False)
        if checkpoint["is_distributed"] and not self.is_distributed:
            self.encoder.load_state_dict({key.replace(".module.", ".").replace("encoder.", ""): value for key, value in checkpoint["model_state_dict"].items() if key[: len("encoder")] == "encoder"})
        else:
            self.encoder.load_state_dict({key.replace("encoder.", ""): value for key, value in checkpoint["model_state_dict"].items() if key[: len("encoder")] == "encoder"})

        # Print Encoder state
        if self.rank == 0:
            print("Model encoder loaded at step {} from {}".format(checkpoint["model_step"], path))

    def gready_search_decoding(self, x, x_len):
        # Forward Encoder (B, Taud) -> (B, T, Denc)
        logits, logits_len = self.encoder(x, x_len)[:2]

        # FC Layer (B, T, Denc) -> (B, T, V)
        logits = self.fc(logits)

        # Softmax -> Log > Argmax -> (B, T)
        preds = logits.log_softmax(dim=-1).argmax(dim=-1)

        # Batch Pred List
        batch_pred_list = []

        # Batch loop
        for b in range(logits.size(0)):
            # Blank
            blank = False

            # Pred List
            pred_list = []

            # Decoding Loop
            for t in range(logits_len[b]):
                # Blank Prediction
                if preds[b, t] == 0:
                    blank = True
                    continue

                # First Prediction
                if len(pred_list) == 0:
                    pred_list.append(preds[b, t].item())

                # New Prediction
                elif pred_list[-1] != preds[b, t] or blank:
                    pred_list.append(preds[b, t].item())

                # Update Blank
                blank = False

            # Append Sequence
            batch_pred_list.append(pred_list)

        # Decode Sequences
        return self.tokenizer.decode(batch_pred_list)

    def beam_search_decoding(self, x, x_len, beam_size=None):
        # Overwrite beam size
        if beam_size is None:
            beam_size = self.beam_size

        # Forward Encoder (B, Taud) -> (B, T, Denc)
        logits, logits_len = self.encoder(x, x_len)[:2]

        # FC Layer (B, T, Denc) -> (B, T, V)
        logits = self.fc(logits)

        # Apply Temperature
        logits = logits / self.tmp

        # Softmax -> Log
        logP = logits.softmax(dim=-1).log()

        # Build labels list: blank token (empty string) + vocab tokens
        # CTC blank is at index 0, vocabulary tokens start from index 1
        labels = [""] + [chr(idx + self.ngram_offset) for idx in range(1, self.tokenizer.vocab_size())]

        # Beam Search Decoder
        decoder = build_ctcdecoder(labels=labels, kenlm_model_path=self.ngram_path, alpha=self.ngram_alpha, beta=self.ngram_beta)

        # Batch Pred List
        batch_pred_list = []

        # Batch loop - decode each sample separately
        for b in range(logits.size(0)):
            # Extract single sample: (T, V)
            single_logP = logP[b, : logits_len[b], :].cpu().numpy()

            # Beam Search Decoding for single sample
            beam_result = decoder.decode(single_logP, beam_width=beam_size)

            # Convert characters back to token IDs
            batch_pred_list.append([ord(c) - self.ngram_offset for c in beam_result if c != ""])

        # Decode Sequences
        return self.tokenizer.decode(batch_pred_list)

    def hf_lm_beam_search_decoding(self, x, x_len, hf_decoder):
        logits, logits_len = self.encoder(x, x_len)[:2]
        logits = self.fc(logits)
        logits = logits / self.tmp
        batch_pred_list = []
        for b in range(logits.size(0)):
            single_logits = logits[b, : logits_len[b], :]
            vocab_to_hf_map = {i: i for i in range(1, self.tokenizer.vocab_size())}
            decoded_indices = hf_decoder.decode(single_logits, vocab_to_hf_map)
            batch_pred_list.append(decoded_indices)
        return self.tokenizer.decode(batch_pred_list)


class InterCTC(ModelCTC):
    def __init__(self, encoder_params, tokenizer_params, training_params, decoding_params, name):
        super(ModelCTC, self).__init__(tokenizer_params, training_params, name)

        # Update Encoder Params
        encoder_params["vocab_size"] = tokenizer_params["vocab_size"]

        # Encoder
        if encoder_params["arch"] == "Conformer":
            self.encoder = ConformerEncoderInterCTC(encoder_params)

        # FC Layer
        self.fc = nn.Linear(encoder_params["dim_model"][-1] if isinstance(encoder_params["dim_model"], list) else encoder_params["dim_model"], tokenizer_params["vocab_size"])

        # Criterion
        self.criterion = LossInterCTC(training_params["interctc_lambda"])

        # Compile
        self.compile(training_params)

    def forward(self, batch):
        # Unpack Batch
        x, _, x_len, _ = batch

        # Forward Encoder (B, Taud) -> (B, T, Denc)
        logits, logits_len, attentions, interctc_logits = self.encoder(x, x_len)

        # FC Layer (B, T, Denc) -> (B, T, V)
        logits = self.fc(logits)

        return logits, logits_len, attentions, interctc_logits
