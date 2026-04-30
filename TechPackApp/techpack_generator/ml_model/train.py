import os
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader
import time
import json

from .config import ModelConfig, V2Config, V3Config, V4Config
from .model import Transformer
from .tokenizer import GarmentTokenizer
from .dataset import TechPackDataset, create_dataLoaders


class CombinedTrainConfig(ModelConfig):
    # 10x lower LR so we adapt to bottoms without losing what the model
    # already knows about tops
    learning_rate = 1e-5
    num_epochs    = 25
    save_every    = 5
    train_file    = os.path.join(ModelConfig.data_dir, 'combined_train.json')
    val_file      = os.path.join(ModelConfig.data_dir, 'combined_val.json')

def train_epoch(model, train_loader, criterion, optimiser, device, epoch):
    #trian for an epoch, return average loss
    model.train()
    total_loss = 0

    start_time = time.time()

    for batch_idx, (src, tgt) in enumerate(train_loader):
        #move to device
        src, tgt = src.to(device), tgt.to(device)

        #forward pass 
        # input all but last token 
        # target all but first token
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        output_logits = model(src, tgt_input)

        #reshape for loss calculation
        output_logits = output_logits.reshape(-1, output_logits.size(-1))
        tgt_output = tgt_output.reshape(-1)

        #calc loss
        loss = criterion(output_logits, tgt_output)

        #backward pass
        optimiser.zero_grad()
        loss.backward()

        #grad clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimiser.step()

        total_loss += loss.item()

        #print progress
        if (batch_idx + 1) % 50 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch} Batch {batch_idx+1}/{len(train_loader)} Loss: {loss.item():.4f} Elapsed: {elapsed:.2f}s")

    return total_loss / len(train_loader)
    
def validate(model, val_loader, criterion, device):
    #validate for an epoch, return average loss
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output_logits = model(src, tgt_input)

            output_logits = output_logits.reshape(-1, output_logits.size(-1))
            tgt_output = tgt_output.reshape(-1)

            loss = criterion(output_logits, tgt_output)
            total_loss += loss.item()

    return total_loss / len(val_loader)        
    
def train():

    #config 
    config = ModelConfig()

    #device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    #dataloaders
    train_loader, val_loader, tokenizer = create_dataLoaders(config)
    
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    #create model

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=config.d_model,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout_rate
    ).to(device)

    #loss and optimiser
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token2id["<PAD>"])
    optimiser = optim.Adam(model.parameters(), lr=config.learning_rate)

    #checkpoint directory
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)

    #training loop
    best_val_loss = float('inf')
    training_history = []

    for epoch in range(1, config.num_epochs + 1):
        epoch_start = time.time()

        #train 
        train_loss = train_epoch(model, train_loader, criterion, optimiser, device, epoch)

        #validate
        val_loss = validate(model, val_loader, criterion, device)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} completed in {epoch_time:.2f}s - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        #save history 
        training_history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'epoch_time': epoch_time
        })
        #save checkpoint
        if epoch % config.save_every == 0:
            checkpoint_path = f"{config.checkpoint_dir}/epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config.__dict__
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}\n")

        #save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = f"{config.model_dir}/best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config.__dict__
            }, best_model_path)
            print(f"New best model saved! Val Loss: {val_loss:.4f}\n")

    #save training history
    history_path = f"{config.model_dir}/training_history.json"
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=4)
    print(f"Training history saved: {history_path}")
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved at: {best_model_path}")

def _extend_vocab_weights(model, old_state: dict):
    # copies matching weights from the checkpoint; for vocab-size layers,
    # old rows are preserved and new token rows keep their Xavier init
    new_state = model.state_dict()
    for key, old_w in old_state.items():
        if key not in new_state:
            continue
        if new_state[key].shape == old_w.shape:
            new_state[key] = old_w
        elif new_state[key].dim() >= 1 and new_state[key].shape[0] > old_w.shape[0]:
            new_state[key][:old_w.shape[0]] = old_w
    model.load_state_dict(new_state)


def train_combined():
    config = CombinedTrainConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tokenizer = GarmentTokenizer()
    tokenizer.load(config.tokenizer_file)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    train_dataset = TechPackDataset(config.train_file, tokenizer, config.max_seq_length)
    val_dataset   = TechPackDataset(config.val_file,   tokenizer, config.max_seq_length)
    train_loader  = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=config.batch_size)
    print(f"Train: {len(train_dataset):,}  Val: {len(val_dataset):,}  Batch: {config.batch_size}")

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=config.d_model,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout_rate,
    ).to(device)

    combined_path  = Path(config.model_dir) / 'best_model_combined.pth'
    old_model_path = combined_path if combined_path.exists() else Path(config.model_dir) / 'best_model.pth'
    print(f"\nLoading: {old_model_path.name}")
    checkpoint = torch.load(old_model_path, map_location=device)
    _extend_vocab_weights(model, checkpoint['model_state_dict'])

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token2id['<PAD>'])
    optimiser = optim.Adam(model.parameters(), lr=config.learning_rate)

    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    history_path  = Path(config.model_dir) / 'training_history_combined.json'
    training_history = json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else []
    epoch_offset = training_history[-1]['epoch'] if training_history else 0
    best_path = Path(config.model_dir) / 'best_model_combined.pth'

    print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>10}  {'Time':>7}  Best?")
    print("-" * 52)

    for epoch in range(1, config.num_epochs + 1):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimiser, device, epoch)
        val_loss   = validate(model, val_loader, criterion, device)
        elapsed    = time.time() - t0
        is_best    = val_loss < best_val_loss

        abs_epoch = epoch_offset + epoch
        print(f"{abs_epoch:>6}  {train_loss:>11.4f}  {val_loss:>10.4f}  {elapsed:>6.1f}s  {'*' if is_best else ''}")

        training_history.append({
            'epoch': abs_epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'epoch_time': elapsed,
        })

        if epoch % config.save_every == 0:
            ckpt_path = Path(config.checkpoint_dir) / f'combined_epoch_{abs_epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'vocab_size': vocab_size,
                    'd_model': config.d_model,
                    'num_heads': config.num_heads,
                    'num_encoder_layers': config.num_encoder_layers,
                    'num_decoder_layers': config.num_decoder_layers,
                    'd_ff': config.d_ff,
                    'dropout_rate': config.dropout_rate,
                    'max_seq_length': config.max_seq_length,
                },
            }, ckpt_path)
            print(f"  checkpoint saved: {ckpt_path.name}")

        if is_best:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'vocab_size': vocab_size,
                    'd_model': config.d_model,
                    'num_heads': config.num_heads,
                    'num_encoder_layers': config.num_encoder_layers,
                    'num_decoder_layers': config.num_decoder_layers,
                    'd_ff': config.d_ff,
                    'dropout_rate': config.dropout_rate,
                    'max_seq_length': config.max_seq_length,
                },
            }, best_path)

    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=4)

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Model saved: {best_path}")


def _train_from_scratch(config, variant: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training {variant}  device={device}")

    tokenizer = GarmentTokenizer()
    tokenizer.load(config.tokenizer_file)
    vocab_size = tokenizer.vocab_size
    print(f"Vocab size: {vocab_size}")

    train_dataset = TechPackDataset(config.train_file, tokenizer, config.max_seq_length)
    val_dataset   = TechPackDataset(config.val_file,   tokenizer, config.max_seq_length)
    train_loader  = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader    = DataLoader(val_dataset,   batch_size=config.batch_size)
    print(f"Train: {len(train_dataset):,}  Val: {len(val_dataset):,}  Batch: {config.batch_size}")

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=config.d_model,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        max_seq_length=config.max_seq_length,
        dropout=config.dropout_rate,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token2id['<PAD>'])
    optimiser = optim.Adam(model.parameters(), lr=config.learning_rate)

    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.model_dir).mkdir(parents=True, exist_ok=True)

    history_path     = Path(config.model_dir) / f'training_history_{variant}.json'
    training_history = json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else []
    epoch_offset     = training_history[-1]['epoch'] if training_history else 0
    best_path        = Path(config.model_dir) / f'best_model_{variant}.pth'
    best_val_loss    = float('inf')

    print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>10}  {'Time':>7}  Best?")
    print("-" * 52)

    for epoch in range(1, config.num_epochs + 1):
        t0         = time.time()
        train_loss = train_epoch(model, train_loader, criterion, optimiser, device, epoch)
        val_loss   = validate(model, val_loader, criterion, device)
        elapsed    = time.time() - t0
        is_best    = val_loss < best_val_loss
        abs_epoch  = epoch_offset + epoch

        print(f"{abs_epoch:>6}  {train_loss:>11.4f}  {val_loss:>10.4f}  {elapsed:>6.1f}s  {'*' if is_best else ''}")

        training_history.append({
            'epoch': abs_epoch, 'train_loss': train_loss,
            'val_loss': val_loss, 'epoch_time': elapsed,
        })

        if epoch % config.save_every == 0:
            ckpt_path = Path(config.checkpoint_dir) / f'{variant}_epoch_{abs_epoch}.pth'
            torch.save({
                'epoch': abs_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'vocab_size': vocab_size,
                    'd_model': config.d_model,
                    'num_heads': config.num_heads,
                    'num_encoder_layers': config.num_encoder_layers,
                    'num_decoder_layers': config.num_decoder_layers,
                    'd_ff': config.d_ff,
                    'dropout_rate': config.dropout_rate,
                    'max_seq_length': config.max_seq_length,
                },
            }, ckpt_path)
            print(f"  checkpoint saved: {ckpt_path.name}")

        if is_best:
            best_val_loss = val_loss
            torch.save({
                'epoch': abs_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimiser.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'vocab_size': vocab_size,
                    'd_model': config.d_model,
                    'num_heads': config.num_heads,
                    'num_encoder_layers': config.num_encoder_layers,
                    'num_decoder_layers': config.num_decoder_layers,
                    'd_ff': config.d_ff,
                    'dropout_rate': config.dropout_rate,
                    'max_seq_length': config.max_seq_length,
                },
            }, best_path)

    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(training_history, f, indent=4)

    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Model saved:   {best_path}")


def train_v2(): _train_from_scratch(V2Config(), 'v2')
def train_v3(): _train_from_scratch(V3Config(), 'v3')
def train_v4(): _train_from_scratch(V4Config(), 'v4')


if __name__ == "__main__":
    import sys
    dispatch = {
        'combined': train_combined,
        'v2':       train_v2,
        'v3':       train_v3,
        'v4':       train_v4,
    }
    dispatch.get(sys.argv[1] if len(sys.argv) > 1 else '', train)()

