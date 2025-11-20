import heapq
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class BeamEntry:
    sequence: List[int]
    ctc_score: float
    lm_score: float
    combined_score: float
    kv_cache: Optional[Tuple] = None


class HFLMBeamDecoder:
    def __init__(self, model_id: str, alpha: float, beta: float, beam_width: int, device: str = "cpu", hf_token: Optional[str] = None):
        print(f"\033[94m[HF_LM] Initializing {model_id} (alpha={alpha}, beta={beta}, beam={beam_width})\033[0m")
        self.alpha = alpha
        self.beta = beta
        self.beam_width = beam_width
        self.device = device
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, token=hf_token, trust_remote_code=True).to(device)
            self.model.eval()
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print("\033[92m[HF_LM] Model loaded successfully\033[0m")
        except Exception as e:
            print(f"\033[91m[HF_LM] ERROR: Failed to load model: {e}\033[0m")
            raise

    def compute_lm_score_incremental(self, prev_cache: Optional[Tuple], new_token: int) -> Tuple[float, Tuple]:
        with torch.no_grad():
            input_ids = torch.tensor([[new_token]], device=self.device)
            outputs = self.model(input_ids=input_ids, past_key_values=prev_cache, use_cache=True)
            logits = outputs.logits[0, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)
            token_logprob = log_probs[new_token].item()
            return token_logprob, outputs.past_key_values

    def decode(self, logits: torch.Tensor, vocab_to_hf_map: dict) -> List[int]:
        T, V = logits.shape
        print(f"\033[94m[HF_LM] Starting beam search: T={T}, V={V}, beam_width={self.beam_width}\033[0m")
        beams = [BeamEntry(sequence=[], ctc_score=0.0, lm_score=0.0, combined_score=0.0, kv_cache=None)]
        blank_idx = 0
        for t in range(T):
            if t % 20 == 0:
                print(f"\033[94m[HF_LM] Time step {t}/{T}: {len(beams)} beams active\033[0m")
            new_beams = []
            log_probs = F.log_softmax(logits[t], dim=-1)
            top_k_values, top_k_indices = torch.topk(log_probs, min(self.beam_width * 2, V))
            print(f"\033[94m[HF_LM] Time step {t}: Processing {len(beams)} beams, top-k tokens: {len(top_k_indices)}\033[0m")
            for beam in beams:
                print(f"\033[94m[HF_LM] Time step {t}: Expanding beam with sequence length {len(beam.sequence)}\033[0m")
                for ctc_logprob, token_idx in zip(top_k_values.tolist(), top_k_indices.tolist(), strict=False):
                    if token_idx == blank_idx:
                        new_beam = BeamEntry(sequence=beam.sequence[:], ctc_score=beam.ctc_score + ctc_logprob, lm_score=beam.lm_score, combined_score=beam.ctc_score + ctc_logprob + self.alpha * beam.lm_score + self.beta * len(beam.sequence), kv_cache=beam.kv_cache)
                        new_beams.append(new_beam)
                        print(f"\033[94m[HF_LM] Time step {t}: Added blank token beam, new beams count: {len(new_beams)}\033[0m")
                    else:
                        new_seq = beam.sequence + [token_idx]
                        new_ctc_score = beam.ctc_score + ctc_logprob
                        if token_idx in vocab_to_hf_map:
                            hf_token_id = vocab_to_hf_map[token_idx]
                            try:
                                lm_token_logprob, new_cache = self.compute_lm_score_incremental(beam.kv_cache, hf_token_id)
                                new_lm_score = beam.lm_score + lm_token_logprob
                                print(f"\033[94m[HF_LM] Time step {t}: LM scoring successful for token {token_idx}, lm_score increment: {lm_token_logprob:.4f}\033[0m")
                            except Exception as e:
                                print(f"\033[93m[HF_LM] WARNING: LM scoring failed at t={t}, falling back\033[0m")
                                print(f"\033[93m[HF_LM] Exception: {e}\033[0m")
                                new_lm_score = beam.lm_score
                                new_cache = beam.kv_cache
                        else:
                            new_lm_score = beam.lm_score
                            new_cache = beam.kv_cache
                            print(f"\033[93m[HF_LM] Time step {t}: Token {token_idx} not in vocab map, skipping LM score\033[0m")
                        combined = new_ctc_score + self.alpha * new_lm_score + self.beta * len(new_seq)
                        new_beam = BeamEntry(sequence=new_seq, ctc_score=new_ctc_score, lm_score=new_lm_score, combined_score=combined, kv_cache=new_cache)
                        new_beams.append(new_beam)
                        print(f"\033[94m[HF_LM] Time step {t}: Added non-blank token beam, new beams count: {len(new_beams)}\033[0m")
            beams = heapq.nlargest(self.beam_width, new_beams, key=lambda b: b.combined_score)
            print(f"\033[94m[HF_LM] Time step {t}: Selected top {len(beams)} beams from {len(new_beams)} candidates\033[0m")
        best_beam = max(beams, key=lambda b: b.combined_score)
        print(f"\033[92m[HF_LM] Beam search completed: Best sequence length {len(best_beam.sequence)}, ctc={best_beam.ctc_score:.2f}, lm={best_beam.lm_score:.2f}, combined={best_beam.combined_score:.2f}\033[0m")
        return best_beam.sequence
