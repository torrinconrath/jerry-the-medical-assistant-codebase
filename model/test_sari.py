# -*- coding: utf-8 -*-
"""
SARI evaluation (class-based, logger.info)
------------------------------------------
Usage example:
    python test_sari.py \
        --data-path /content/data/intro2AI_sft/val.parquet \
        --model-path SFT_merged_model \
        --backend hf \
        --limit 50 \
        --out eval_sari.jsonl
"""

import os
import re
import json
import argparse
import logging
from typing import List, Dict, Any, Optional

import pandas as pd
from tqdm.auto import tqdm

import torch
from evaluate import load

# Optional imports (handled lazily)
# transformers, vllm, jieba are imported inside class as needed.

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("SARIAnalyzer")


class SARIAnalyzer:
    def __init__(
        self,
        data_path: str,
        model_path: str,
        backend: str = "hf",              # "hf" or "vllm"
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        limit: Optional[int] = None,
        out_path: str = "eval_sari.jsonl",
        chinese_seg: bool = False,        # if True, use jieba to segment
        device: Optional[str] = None,     # "cuda" / "cpu" / None(auto)
    ):
        self.data_path = data_path
        self.model_path = model_path
        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.limit = limit
        self.out_path = out_path
        self.chinese_seg = chinese_seg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.df: Optional[pd.DataFrame] = None
        self.sources: List[str] = []
        self.predictions: List[str] = []
        self.references: List[List[str]] = []

        # Lazy-loaded members:
        self.tok = None
        self.model = None            # for HF
        self.llm = None              # for vLLM
        self.sampler = None          # for vLLM
        self.jieba = None            # optional

    # -------------------- Loading --------------------
    def load_data(self):
        logger.info(f"Loading dataset: {self.data_path}")
        self.df = pd.read_parquet(self.data_path)
        logger.info(f"Dataset size = {len(self.df)} rows")
        if self.limit:
            self.df = self.df.head(self.limit)
            logger.info(f"Using the first {len(self.df)} rows due to --limit")

    def load_tokenizer(self):
        from transformers import AutoTokenizer
        logger.info(f"Loading tokenizer from {self.model_path}")
        self.tok = AutoTokenizer.from_pretrained(self.model_path, use_fast=True)
        if getattr(self.tok, "pad_token", None) is None and getattr(self.tok, "eos_token", None):
            self.tok.pad_token = self.tok.eos_token

    def load_model_hf(self):
        from transformers import AutoModelForCausalLM
        logger.info(f"[HF] Loading model from {self.model_path} on {self.device}")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            attn_implementation="flash_attention_2" if self.device == "cuda" else None,
        ).eval().to(self.device)

    def load_model_vllm(self):
        logger.info(f"[vLLM] Loading engine from {self.model_path}")
        from vllm import LLM, SamplingParams
        # vLLM still needs tokenizer to build chat template, so we reuse self.tok
        self.llm = LLM(model=self.model_path, trust_remote_code=True)
        self.sampler = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_new_tokens,
        )

    def maybe_load_jieba(self):
        if self.chinese_seg:
            try:
                import jieba  # type: ignore
                self.jieba = jieba
                logger.info("Chinese segmentation enabled (jieba).")
            except Exception as e:
                logger.info(f"Failed to import jieba, disabling zh segmentation: {e}")
                self.chinese_seg = False

    # -------------------- Prompt & Generation --------------------
    def build_prompt(self, question: str) -> str:
        """
        Prefer tokenizer's chat template if available; otherwise fallback.
        """
        messages = [{"role": "user", "content": question}]
        if self.tok is not None and hasattr(self.tok, "apply_chat_template"):
            return self.tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return f"User:\n{question}\nAssistant:"

    @torch.inference_mode()
    def generate_hf(self, prompt: str) -> str:
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=self.tok.eos_token_id,
            pad_token_id=self.tok.eos_token_id,
            return_dict_in_generate=True,
        )
        gen_ids = out.sequences[:, inputs["input_ids"].shape[1]:]
        text = self.tok.decode(gen_ids[0], skip_special_tokens=True).strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        return text

    def generate_vllm(self, prompt: str) -> str:
        # vLLM expects a list of prompts
        outs = self.llm.generate([prompt], self.sampler)
        text = outs[0].outputs[0].text.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        return text

    def generate(self, question: str) -> str:
        prompt = self.build_prompt(question)
        if self.backend == "vllm":
            return self.generate_vllm(prompt)
        return self.generate_hf(prompt)

    # -------------------- Dataset Fields --------------------
    def _extract_question_and_reference(self, row: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """
        Expect:
            row["extra_info"]["question"]
            row["reward_model"]["ground_truth"]
        """
        try:
            question = row["extra_info"]["question"]
            reference = row["reward_model"]["ground_truth"]
            if isinstance(question, str) and isinstance(reference, str):
                return {"question": question, "reference": reference}
        except Exception:
            return None
        return None

    # -------------------- SARI --------------------
    def segment_zh(self, s: str) -> str:
        if not self.chinese_seg or self.jieba is None:
            return s
        return " ".join(self.jieba.lcut(s))

    def compute_sari(self):
        logger.info("Computing SARI...")
        sari = load("sari")
        srcs = self.sources
        preds = self.predictions
        refs = self.references

        if self.chinese_seg and self.jieba is not None:
            srcs = [self.segment_zh(s) for s in srcs]
            preds = [self.segment_zh(p) for p in preds]
            refs = [[self.segment_zh(r) for r in rlist] for rlist in refs]

        result = sari.compute(sources=srcs, predictions=preds, references=refs)
        # result typically contains {"sari": x, "keep": ..., "add": ..., "delete": ...} depending on version
        logger.info(f"SARI = {result.get('sari', 0):.4f}")
        for k, v in result.items():
            if k != "sari":
                try:
                    logger.info(f"{k} = {float(v):.4f}")
                except Exception:
                    logger.info(f"{k} = {v}")
        return result

    # -------------------- Orchestration --------------------
    def run(self):
        # 1) load
        self.load_data()
        self.load_tokenizer()
        if self.backend == "vllm":
            self.load_model_vllm()
        else:
            self.load_model_hf()
        self.maybe_load_jieba()

        # 清空缓存，避免多次调用时串样本
        self.sources, self.predictions, self.references = [], [], []

        # 2) iterate rows: build sources/preds/refs
        logger.info("Generating model outputs...")
        assert self.df is not None
        out_rows = []

        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="[gen]"):
            sample = self._extract_question_and_reference(row)
            if not sample:
                continue
            q = sample["question"]
            ref = sample["reference"]
            try:
                pred = self.generate(q)
            except Exception as e:
                logger.info(f"[gen-err] {type(e).__name__}: {e}")
                pred = ""

            self.sources.append(q)
            self.predictions.append(pred)
            self.references.append([ref])  # list-of-list

            out_rows.append({
                "question": q,
                "prediction": pred,
                "reference": ref
            })

        # 3) compute sari
        result = self.compute_sari()

        # 4) dump details（逐样本）
        if self.out_path:
            with open(self.out_path, "w", encoding="utf-8") as f:
                for r in out_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            logger.info(f"Saved per-sample outputs to: {os.path.abspath(self.out_path)}")

        # 5) 汇总写入 summary_metrics.json（追加/去重）
        if result:
            summary_path = "summary_metrics.json"
            entry = {
                "temperature": float(self.temperature),
                "out_path": os.path.abspath(self.out_path) if self.out_path else None,
                **{k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
            }

            try:
                if os.path.exists(summary_path):
                    with open(summary_path, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                    if not isinstance(existing, list):
                        existing = [existing]
                else:
                    existing = []
            except Exception:
                existing = []

            # 若同一 temperature 已存在，则以这次结果覆盖
            existing = [e for e in existing if e.get("temperature") != entry["temperature"]]
            existing.append(entry)

            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
            logger.info(f"Appended overall SARI metrics to {summary_path}")

        logger.info("Done.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", type=str, required=True,
                    help="Parquet file with columns: extra_info.question, reward_model.ground_truth")
    ap.add_argument("--model-path", type=str, required=True,
                    help="HF or vLLM model path (e.g., SFT_merged_model)")
    ap.add_argument("--backend", type=str, default="hf", choices=["hf", "vllm"],
                    help="Inference backend: hf (transformers) or vllm")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--limit", type=int, default=None, help="Use first N samples")
    ap.add_argument("--out", type=str, default="eval_sari.jsonl", help="Write per-sample results here")
    ap.add_argument("--zh-seg", action="store_true", help="Enable Chinese word segmentation (jieba)")
    ap.add_argument("--device", type=str, default=None, help="Force device: cuda / cpu")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    base_t = float(args.temperature)
    offsets = [0.0]
    temps = []
    for off in offsets:
        t = round(base_t + off, 2)
        if 0.0 <= t <= 1.0:
            temps.append(t)
    temps = sorted(set(temps))

    for t in temps:
        root, ext = os.path.splitext(args.out)
        out_path = f"{root}.t{t}{ext or '.jsonl'}"

        logger.info(f"\n====== Running temperature = {t} ======")
        analyzer = SARIAnalyzer(
            data_path=args.data_path,
            model_path=args.model_path,
            backend=args.backend,
            max_new_tokens=args.max_new_tokens,
            temperature=t,
            top_p=args.top_p,
            limit=args.limit,
            out_path=out_path,
            chinese_seg=args.zh_seg,
            device=args.device,
        )
        analyzer.run()  # run() 内部会自己把该温度的指标追加到 summary_metrics.json