# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Pretrain Lorentz GPT model with hyperbolic geometry.

"""
Pretrain Lorentz GPT Model.

Standalone training script for hyperbolic GPT with Qwen3-like architecture.
Uses PyTorch DDP for distributed training.

Usage:
    torchrun --nproc_per_node=2 pretrain_lorentz_gpt.py --config qwen3_0.6b

Features:
    - Lorentz (hyperbolic) geometry in all transformer components
    - GQA (Grouped Query Attention) support
    - SwiGLU MLP
    - RMSNorm
    - Supports dummy data or real tokenized data
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Tuple, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# Add megatron to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from megatron.core.manifolds import Lorentz, project_space_to_lorentz
from megatron.core.transformer.lorentz_norm import LorentzRMSNorm
from megatron.core.transformer.lorentz_residual import LorentzResidual
from megatron.core.transformer.lorentz_attention import LorentzDotProductAttention
from megatron.core.transformer.lorentz_mlp import LorentzMLP
from megatron.core.models.gpt.lorentz_gpt_model import (
    LorentzEmbedding,
    LorentzOutputLayer,
)


# =============================================================================
# Model Configuration
# =============================================================================

@dataclass
class LorentzGPTConfig:
    """Configuration for Lorentz GPT model."""

    # Model name
    name: str = "lorentz-gpt"

    # Architecture
    hidden_size: int = 1024
    num_layers: int = 28
    num_attention_heads: int = 16
    num_kv_heads: int = 8  # For GQA
    ffn_hidden_size: int = 3072
    vocab_size: int = 151936
    max_seq_length: int = 2048

    # Normalization
    norm_epsilon: float = 1e-6

    # Dropout (Qwen3 uses 0)
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0

    # Hyperbolic settings
    curvature: float = 1.0
    learnable_curvature: bool = False

    # Derived properties
    @property
    def kv_channels(self) -> int:
        return self.hidden_size // self.num_attention_heads

    def to_dict(self):
        return asdict(self)


# Preset configurations
CONFIGS = {
    # Test config (small, for debugging)
    "test": LorentzGPTConfig(
        name="lorentz-gpt-test",
        hidden_size=256,
        num_layers=4,
        num_attention_heads=4,
        num_kv_heads=2,
        ffn_hidden_size=768,
        vocab_size=1024,
        max_seq_length=512,
    ),
    # Qwen3-0.6B-like config
    "qwen3_0.6b": LorentzGPTConfig(
        name="lorentz-gpt-qwen3-0.6b",
        hidden_size=1024,
        num_layers=28,
        num_attention_heads=16,
        num_kv_heads=8,
        ffn_hidden_size=3072,
        vocab_size=151936,
        max_seq_length=2048,
    ),
    # Smaller variant for faster iteration
    "qwen3_0.6b_small": LorentzGPTConfig(
        name="lorentz-gpt-qwen3-0.6b-small",
        hidden_size=512,
        num_layers=12,
        num_attention_heads=8,
        num_kv_heads=4,
        ffn_hidden_size=1536,
        vocab_size=32000,
        max_seq_length=1024,
    ),
}


# =============================================================================
# Model Components
# =============================================================================

class LorentzSelfAttention(nn.Module):
    """Lorentz self-attention with GQA support."""

    def __init__(self, manifold: Lorentz, config: LorentzGPTConfig, layer_idx: int = 0):
        super().__init__()
        self.manifold = manifold
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.kv_channels

        # QKV projections
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        # Core attention
        self.core_attention = LorentzDotProductAttention(
            manifold=manifold,
            num_attention_heads=self.num_heads,
            hidden_size_per_head=self.head_dim,
            attention_dropout=config.attention_dropout,
            layer_number=layer_idx,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape

        # Extract space-like dimensions
        x_space = hidden_states[..., 1:]

        # Project to Q, K, V
        q = self.q_proj(x_space).view(batch_size, seq_length, self.num_heads, self.head_dim)
        k = self.k_proj(x_space).view(batch_size, seq_length, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x_space).view(batch_size, seq_length, self.num_kv_heads, self.head_dim)

        # GQA: expand K, V
        if self.num_kv_heads < self.num_heads:
            n_rep = self.num_heads // self.num_kv_heads
            k = k.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(
                batch_size, seq_length, self.num_heads, self.head_dim
            )
            v = v.unsqueeze(3).expand(-1, -1, -1, n_rep, -1).reshape(
                batch_size, seq_length, self.num_heads, self.head_dim
            )

        # Transpose to (batch, heads, seq, dim)
        q, k, v = [x.transpose(1, 2) for x in (q, k, v)]

        # Project to Lorentz manifold
        q = project_space_to_lorentz(q, self.manifold.c)
        k = project_space_to_lorentz(k, self.manifold.c)
        v = project_space_to_lorentz(v, self.manifold.c)

        # Core attention
        attn_output, _ = self.core_attention(q, k, v, attention_mask)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output_space = attn_output[..., 1:].contiguous()
        attn_output_space = attn_output_space.view(batch_size, seq_length, -1)
        output_space = self.o_proj(attn_output_space)

        return project_space_to_lorentz(output_space, self.manifold.c)


class LorentzTransformerLayer(nn.Module):
    """Single Lorentz transformer layer."""

    def __init__(self, manifold: Lorentz, config: LorentzGPTConfig, layer_idx: int = 0):
        super().__init__()
        self.manifold = manifold

        self.input_layernorm = LorentzRMSNorm(manifold, config.hidden_size, config.norm_epsilon)
        self.self_attention = LorentzSelfAttention(manifold, config, layer_idx)
        self.attn_residual = LorentzResidual(manifold)
        self.pre_mlp_layernorm = LorentzRMSNorm(manifold, config.hidden_size, config.norm_epsilon)
        self.mlp = LorentzMLP(manifold, config.hidden_size, config.ffn_hidden_size, bias=False)
        self.mlp_residual = LorentzResidual(manifold)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attention(hidden_states, attention_mask)
        hidden_states = self.attn_residual(residual, hidden_states)

        # MLP block
        residual = hidden_states
        hidden_states = self.pre_mlp_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_residual(residual, hidden_states)

        return hidden_states


class LorentzGPT(nn.Module):
    """Full Lorentz GPT model."""

    def __init__(self, config: LorentzGPTConfig):
        super().__init__()
        self.config = config

        self.manifold = Lorentz(c=config.curvature, learnable=config.learnable_curvature)
        self.embedding = LorentzEmbedding(self.manifold, config.vocab_size, config.hidden_size)

        self.layers = nn.ModuleList([
            LorentzTransformerLayer(self.manifold, config, i)
            for i in range(config.num_layers)
        ])

        self.final_norm = LorentzRMSNorm(self.manifold, config.hidden_size, config.norm_epsilon)
        self.output_layer = LorentzOutputLayer(self.manifold, config.hidden_size, config.vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = self.embedding(input_ids)

        # Create causal mask
        if attention_mask is None:
            seq_length = input_ids.shape[1]
            attention_mask = torch.triu(
                torch.ones(seq_length, seq_length, dtype=torch.bool, device=input_ids.device),
                diagonal=1
            )

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        hidden_states = self.final_norm(hidden_states)
        logits = self.output_layer(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Dataset
# =============================================================================

class DummyDataset(Dataset):
    """Dummy dataset with random tokens."""

    def __init__(self, vocab_size: int, seq_length: int, num_samples: int):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (self.seq_length,))
        return {"input_ids": tokens, "labels": tokens.clone()}


class TokenizedDataset(Dataset):
    """Dataset from tokenized .bin file (Megatron format)."""

    def __init__(self, data_path: str, seq_length: int):
        import numpy as np

        self.seq_length = seq_length

        # Load .bin file
        bin_path = f"{data_path}.bin"
        idx_path = f"{data_path}.idx"

        if os.path.exists(bin_path):
            self.data = np.memmap(bin_path, dtype=np.int32, mode='r')
            self.num_samples = len(self.data) // seq_length
        else:
            raise FileNotFoundError(f"Data file not found: {bin_path}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start = idx * self.seq_length
        end = start + self.seq_length
        tokens = torch.from_numpy(self.data[start:end].astype(np.int64))
        return {"input_ids": tokens, "labels": tokens.clone()}


# =============================================================================
# Training
# =============================================================================

@dataclass
class TrainingArgs:
    """Training arguments."""

    # Data
    data_path: Optional[str] = None
    num_samples: int = 10000

    # Training
    batch_size: int = 4
    micro_batch_size: int = 1
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    max_steps: int = 1000
    warmup_steps: int = 100
    grad_clip: float = 1.0

    # Precision
    bf16: bool = True

    # Logging
    log_interval: int = 10
    eval_interval: int = 100
    save_interval: int = 500

    # Checkpointing
    checkpoint_dir: str = "./checkpoints"
    resume_from: Optional[str] = None

    # Distributed
    local_rank: int = -1


def setup_distributed():
    """Initialize distributed training."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_lr(step: int, args: TrainingArgs) -> float:
    """Cosine learning rate schedule with warmup."""
    if step < args.warmup_steps:
        return args.lr * step / args.warmup_steps

    progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
    return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * progress))


def train(args: TrainingArgs, config: LorentzGPTConfig):
    """Main training function."""

    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    is_main = rank == 0

    if is_main:
        print("=" * 60)
        print("Lorentz GPT Training")
        print("=" * 60)

    if is_main:
        print(f"Config: {config.name}")
        print(f"  hidden_size: {config.hidden_size}")
        print(f"  num_layers: {config.num_layers}")
        print(f"  num_heads: {config.num_attention_heads}")
        print(f"  num_kv_heads: {config.num_kv_heads}")
        print(f"  ffn_hidden_size: {config.ffn_hidden_size}")
        print(f"  vocab_size: {config.vocab_size}")
        print(f"  curvature: {config.curvature}")

    # Create model
    model = LorentzGPT(config).to(device)

    if is_main:
        num_params = model.get_num_params()
        print(f"Parameters: {num_params:,} ({num_params/1e6:.2f}M)")

    # Wrap with DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    # Create dataset
    if args.data_path:
        dataset = TokenizedDataset(args.data_path, config.max_seq_length)
    else:
        dataset = DummyDataset(config.vocab_size, config.max_seq_length, args.num_samples)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
    )

    if is_main:
        print(f"Dataset: {len(dataset)} samples")
        print(f"Batch size: {args.batch_size} (micro: {args.micro_batch_size})")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print("=" * 60)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    # Mixed precision
    scaler = torch.amp.GradScaler() if args.bf16 else None
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float32

    # Gradient accumulation
    grad_accum_steps = args.batch_size // (args.micro_batch_size * world_size)

    # Training loop
    model.train()
    data_iter = iter(dataloader)

    total_tokens = 0
    start_time = time.time()

    for step in range(1, args.max_steps + 1):
        # Update learning rate
        lr = get_lr(step, args)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Accumulate gradients
        total_loss = 0.0
        optimizer.zero_grad()

        for micro_step in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                sampler.set_epoch(step)
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype, enabled=args.bf16):
                _, loss = model(input_ids, labels=labels)
                loss = loss / grad_accum_steps

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item()

        # Gradient clipping and optimizer step
        if scaler:
            scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Update token count
        tokens_per_step = args.batch_size * config.max_seq_length
        total_tokens += tokens_per_step

        # Logging
        if step % args.log_interval == 0 and is_main:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / elapsed

            print(f"Step {step:5d} | Loss: {total_loss:.4f} | LR: {lr:.2e} | "
                  f"Grad: {grad_norm:.2f} | Tok/s: {tokens_per_sec:.0f}")

        # Save checkpoint
        if step % args.save_interval == 0 and is_main:
            save_checkpoint(model, optimizer, step, args, config)

    # Final save
    if is_main:
        save_checkpoint(model, optimizer, args.max_steps, args, config)
        print("=" * 60)
        print("Training complete!")
        print("=" * 60)

    cleanup_distributed()


def save_checkpoint(model, optimizer, step, args, config):
    """Save model checkpoint."""
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Get raw model (unwrap DDP if needed)
    raw_model = model.module if hasattr(model, 'module') else model

    checkpoint = {
        "step": step,
        "model_state_dict": raw_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config.to_dict(),
    }

    path = os.path.join(args.checkpoint_dir, f"checkpoint_step{step}.pt")
    torch.save(checkpoint, path)
    print(f"Saved checkpoint: {path}")

    # Save config
    config_path = os.path.join(args.checkpoint_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Pretrain Lorentz GPT")

    # Model Architecture
    parser.add_argument("--hidden-size", type=int, default=1024,
                        help="Hidden size")
    parser.add_argument("--num-layers", type=int, default=28,
                        help="Number of transformer layers")
    parser.add_argument("--num-attention-heads", type=int, default=16,
                        help="Number of attention heads")
    parser.add_argument("--num-kv-heads", type=int, default=8,
                        help="Number of KV heads for GQA")
    parser.add_argument("--ffn-hidden-size", type=int, default=3072,
                        help="FFN hidden size")
    parser.add_argument("--vocab-size", type=int, default=151936,
                        help="Vocabulary size")
    parser.add_argument("--seq-length", type=int, default=2048,
                        help="Sequence length")

    # Hyperbolic Configuration
    parser.add_argument("--curvature", type=float, default=1.0,
                        help="Hyperbolic curvature")

    # Data
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to tokenized data (without .bin/.idx extension)")
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Number of samples for dummy data")

    # Training
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Global batch size")
    parser.add_argument("--micro-batch-size", type=int, default=4,
                        help="Micro batch size per GPU")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate")
    parser.add_argument("--min-lr", type=float, default=3e-5,
                        help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.1,
                        help="Weight decay")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="Maximum training steps")
    parser.add_argument("--warmup-steps", type=int, default=100,
                        help="Warmup steps")
    parser.add_argument("--grad-clip", type=float, default=1.0,
                        help="Gradient clipping")

    # Precision
    parser.add_argument("--bf16", action="store_true", default=True,
                        help="Use BF16 mixed precision")
    parser.add_argument("--no-bf16", action="store_false", dest="bf16",
                        help="Disable BF16")

    # Logging
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--save-interval", type=int, default=500,
                        help="Save checkpoint every N steps")

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints",
                        help="Checkpoint directory")

    args = parser.parse_args()

    # Create model config from command-line args
    config = LorentzGPTConfig(
        name="lorentz-gpt",
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_attention_heads=args.num_attention_heads,
        num_kv_heads=args.num_kv_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        vocab_size=args.vocab_size,
        max_seq_length=args.seq_length,
        curvature=args.curvature,
    )

    # Convert to TrainingArgs
    training_args = TrainingArgs(
        data_path=args.data_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        bf16=args.bf16,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        checkpoint_dir=args.checkpoint_dir,
    )

    train(training_args, config)


if __name__ == "__main__":
    main()
