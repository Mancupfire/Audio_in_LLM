import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, BertModel, BertTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoModel
import transformers
from peft import LoraConfig, TaskType, get_peft_model, IA3Config
import logging
import pytorch_lightning as pl
from torchmetrics import AUROC
import os
from Utils import get_dataloader, EarlyStopper, test
from Support_Utils import initialize_pretrained_model
from tqdm import tqdm


os.environ["TOKENIZERS_PARALLELISM"] = "false"
transformers.logging.set_verbosity_error()


token = "redacted"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

target_module_dict = {
    "operaCT": ["qkv", "proj"]
}


class FlattenHead(nn.Module):
    def __init__(self, nf, out_dim, head_dropout=0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, out_dim)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x, no_fc=False):
        x = self.flatten(x)
        if no_fc:
            return x
        x = self.linear(x)
        x = self.dropout(x)
        return x
    
class RespLLM(nn.Module):
    def __init__(self, configs):
        super(RespLLM, self).__init__()

        # Thiết lập các tham số
        self.n_cls = configs.n_cls
        self.d_ff = configs.d_ff
        self.d_llm = configs.llm_dim
        self.audio_peft = configs.audio_peft
        self.d_audio = configs.enc_dim
        self.patch_nums = configs.patch_nums
        self.head_nf = self.d_ff * self.patch_nums

        self.llm_peft = configs.llm_peft
        self.llm_lora_rank = configs.llm_lora_rank
        self.llm_lora_alpha = configs.llm_lora_alpha
        self.llm_lora_dropout = configs.llm_lora_dropout
        self.use_audio = configs.use_audio

        # --- LLM model chỉ hỗ trợ “llama” trong phiên bản gọn này ---
        if configs.llm_model == 'llama':
            self.llama_config = LlamaConfig.from_pretrained('huggyllama/llama-7b')
            try:
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                )
            except EnvironmentError:
                print("Local model files not found. Downloading ...")
                self.llm_model = LlamaModel.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:
                print("Local tokenizer files not found. Downloading ...")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    'huggyllama/llama-7b',
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise NotImplementedError("Chỉ hỗ trợ LLM model 'llama' trong phiên bản này.")

        # Thiết lập pad_token cho tokenizer
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = "[PAD]"
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # --- LLM Fine-tuning với PEFT (chỉ hỗ trợ chế độ LoRA hoặc frozen) ---
        if self.llm_peft == "lora":
            peft_config = LoraConfig(
                r=self.llm_lora_rank,
                lora_alpha=self.llm_lora_alpha,
                lora_dropout=self.llm_lora_dropout,
                target_modules=["q_proj", "v_proj"]
            )
            self.llm_model = get_peft_model(self.llm_model, peft_config)
        elif self.llm_peft == "frozen":
            for param in self.llm_model.parameters():
                param.requires_grad = False
        else:
            raise NotImplementedError("Chế độ fine-tuning LLM không được hỗ trợ.")

        if configs.audio_encoder == "operaCT":
            self.audio_encoder = initialize_pretrained_model("operaCT").encoder
        else:
            raise NotImplementedError("Chỉ hỗ trợ audio encoder 'operaCT'.")

        # --- Cấu hình fine-tuning cho audio encoder ---
        if self.audio_peft == "frozen":
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
        elif self.audio_peft in ["lora", "IA3"]:
            if self.audio_peft == "lora":
                peft_config = LoraConfig(
                    r=configs.audio_lora_rank,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    target_modules=target_module_dict[configs.audio_encoder]
                )
            elif self.audio_peft == "IA3":
                peft_config = IA3Config(
                    target_modules=target_module_dict[configs.audio_encoder],
                    feedforward_modules=['proj']
                )
            self.audio_encoder = get_peft_model(self.audio_encoder, peft_config)
        elif self.audio_peft == "full":
            for param in self.audio_encoder.parameters():
                param.requires_grad = True
        else:
            raise NotImplementedError("Chế độ fine-tuning audio không được hỗ trợ.")

        # --- Aligner module: chuyển đổi từ audio embedding sang không gian LLM ---
        if configs.aligner == "projection":
            self.aligner = nn.Linear(self.d_audio, self.d_llm)
        else:
            raise NotImplementedError("Aligner module undefined.")

        self.head_dropout = configs.head_dropout
        self.output_projection = FlattenHead(self.head_nf, self.n_cls, head_dropout=self.head_dropout)

    def print_trainable(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("Total trainable parameters:", trainable_params)

    def forward(self, x_spectrogram, x_prompt, x_context, no_fc=False):
        # Xử lý audio: Nếu patch_nums == 1 thì dùng encoder trực tiếp, nếu 64 dùng forward_window (nếu hỗ trợ)
        if self.patch_nums == 1:
            x_enc = self.audio_encoder(x_spectrogram)
            enc_out = self.aligner(x_enc)
            enc_out = enc_out.unsqueeze(dim=1)
        elif self.patch_nums == 64:
            x_enc = self.audio_encoder.forward_window(x_spectrogram)
            enc_out = self.aligner(x_enc)
        else:
            raise NotImplementedError("patch_nums không được hỗ trợ")

        # Tokenize prompt và context
        prompt = self.tokenizer(x_prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids.to(x_enc.device)
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt)
        context = self.tokenizer(x_context, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids.to(x_enc.device)
        context_embeddings = self.llm_model.get_input_embeddings()(context)

        # Kết hợp embeddings
        if self.use_audio:
            llama_enc_out = torch.cat([prompt_embeddings, context_embeddings, enc_out], dim=1)
        else:
            llama_enc_out = torch.cat([prompt_embeddings, context_embeddings], dim=1)

        # Cho LLM xử lý và lấy hidden state cuối
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        dec_out = self.output_projection(dec_out[:, :, -self.patch_nums:], no_fc=no_fc)
        return dec_out
