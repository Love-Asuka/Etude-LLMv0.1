import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import tiktoken
from model_train import Etude, GPTConfig  

class SFTDataset(Dataset):
    def __init__(self, path, block_size=512):
        self.enc = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        self.eos_token = self.enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        self.bos_token = self.enc.encode("<|startoftext|>", allowed_special={"<|startoftext|>"})[0]
        self.sep_token = self.enc.encode("\n", allowed_special={"\n"})[0] 
        self.encoded_data = []
        self.max_lines = 500
        raw_conversations = []
        
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= self.max_lines:
                    break
                try:
                    data = json.loads(line.strip())
                    conversations = data.get("conversations", [])
                    if conversations:
                        raw_conversations.append(conversations)
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    continue
        

        for conv in raw_conversations:

            full_encoded = [self.bos_token]
            for turn in conv:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role and content:

                    role_encoded = self.enc.encode(role + ": ")
                    content_encoded = self.enc.encode(content)
                    full_encoded.extend(role_encoded + content_encoded + [self.sep_token])
            
            full_encoded.append(self.eos_token)

            for i in range(0, len(full_encoded), self.block_size):
                chunk = full_encoded[i:i+self.block_size+1]
                if len(chunk) < self.block_size + 1:
                    chunk = chunk + [self.eos_token] * (self.block_size + 1 - len(chunk))
                self.encoded_data.append(chunk)
    
    def __len__(self):
        return len(self.encoded_data)
    
    def __getitem__(self, idx):
        chunk = self.encoded_data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

def train_sft(model, optimizer, scheduler, train_loader, val_loader, device, epoch, total_epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"SFT训练第 {epoch+1}/{total_epochs} 轮", unit="batch")
    
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        

        logits, loss = model(x, targets=y)
        

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"损失": f"{loss.item():.4f}"})
    
    return total_loss / len(train_loader)

def evaluate_sft(model, val_loader, device):
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            val_loss += loss.item()
    
    return val_loss / len(val_loader)

def main2():

    config = GPTConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    

    model = Etude(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    

    total_params = sum(p.numel() for p in model.parameters())
    print(f"总参数量: {total_params / 1e6:.2f} M")

    # 检查点恢复逻辑（与原文件一致）
    checkpoint_path = "weight/sft_model_latest.pt"
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"从检查点恢复训练，从第 {start_epoch} 轮开始")

    # 多文件处理逻辑（与原文件一致）
    folder_path = "jsonl_sft"  # SFT对话数据目录
    jsonl_files = [f for f in os.listdir(folder_path) if f.endswith('.jsonl')]
    num_jsonls_to_train = 200  # 指定训练的JSONL文件数量
    epochs_per_file = 1  # 每个JSONL文件训练的轮数
    total_epochs = num_jsonls_to_train * epochs_per_file  
    
    save_interval = 100  # 保存间隔

    for epoch in range(start_epoch, total_epochs):
        file_idx = (epoch // epochs_per_file) % num_jsonls_to_train
        file_path = os.path.join(folder_path, jsonl_files[file_idx % len(jsonl_files)])
        

        train_dataset = SFTDataset(file_path)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)


        train_loss = train_sft(model, optimizer, scheduler, train_loader, val_loader, device, epoch, total_epochs)
        val_loss = evaluate_sft(model, val_loader, device)
        print(f'轮次: {epoch+1}, 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}')


        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_loss,
        }

        if (epoch + 1) % save_interval == 0:
            torch.save(checkpoint, f'weight/sft_model_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)

if __name__ == "__main__":
    main2()