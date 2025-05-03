import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from multiprocessing import freeze_support
import torchaudio

# Return raw waveform segments
torch.manual_seed(42)
class AudioWaveformDataset(Dataset):
    def __init__(self, file_list, labels, sample_rate=16000, segment_length=5.0,):
        self.file_list = file_list # Danh sách đường dẫn tới các file
        self.labels = labels # Các nhãn tương ứng
        self.sample_rate = sample_rate # Tần số mẫu (16000 Hz)
        self.segment_samples = int(sample_rate * segment_length) 

    def __len__(self):
        return len(self.file_list) # Số lượng gile trong danh sách 

    def __getitem__(self, idx):
        path = self.file_list[idx] # Đường dẫn tới file âm thanh
        label = self.labels[idx]
        waveform, sr = torchaudio.load(path) # Đọc file âm thanh (chanels, samples)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(dim=0)  # mono: (samples,)

        # split into fixed-length waveform segments
        segments = []
        total = waveform.size(0)
        for start in range(0, total - self.segment_samples + 1, self.segment_samples):
            segment = waveform[start:start + self.segment_samples]
            segments.append(segment)
        segments = torch.stack(segments)  # (num_segments, segment_samples)
        return segments, label

# Flatten segments and prepare preocessor inputs

def collate_fn(batch, processor=None):
    all_waveforms = []
    all_labels = []
    for segments, label in batch:
        for seg in segments:
            all_waveforms.append(seg.numpy())
            all_labels.append(label)
    inputs = processor(all_waveforms, sampling_rate=processor.feature_extractor.sampling_rate,
                       return_tensors='pt', padding=True)
    labels = torch.tensor(all_labels)
    return inputs, labels

def get_dataloader(file_list, labels, processor, batch_size=8, shuffle=True, num_workers=4, resample=False, segment_length=5.0,):
    dataset = AudioWaveformDataset(file_list, labels, sample_rate=processor.feature_extractor.sampling_rate, segment_length=segment_length)

    if resample:
        counts = torch.tensor([labels.count(i) for i in sorted(set(labels))], dtype=torch.float)
        weights = 1.0 / counts
        sample_weights = torch.tensor([weights[l] for l in labels])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                            num_workers=num_workers,
                            collate_fn=lambda batch: collate_fn(batch, processor))
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            num_workers=num_workers,
                            collate_fn=lambda batch: collate_fn(batch, processor))
    return loader

if __name__ == "__main__":
    freeze_support()
    wav_dir = Path(r"C:\Users\jimmy\Documents\...\RespiratoryDatabase@TR")
    excel_path = Path(r"C:\Users\jimmy\Documents\...\COPD.xlsx")

    all_files = sorted(str(p) for p in wav_dir.rglob("*.wav"))
    df = pd.read_excel(excel_path, dtype=str)
    id2lbl = dict(zip(df["Patient ID"], df["Diagnosis"]))
    labels = [id2lbl[Path(f).stem[:4]] for f in all_files]
    uniq = sorted(set(labels))
    lbl2idx = {l: i for i, l in enumerate(uniq)}
    labels = [lbl2idx[l] for l in labels]
    train_files, val_files, train_lbls, val_lbls = train_test_split(
        all_files, labels, test_size=0.2, random_state=42
    )
    num_classes = len(uniq)

    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        "facebook/wav2vec2-base-960h",
        num_labels=num_classes
    ).to('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = get_dataloader(train_files, train_lbls, processor,
                                  batch_size=8, shuffle=False, resample=True)
    val_loader   = get_dataloader(val_files, val_lbls, processor,
                                  batch_size=8, shuffle=False, resample=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, 11):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(input_values, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
        avg_train = total_loss / len(train_loader.dataset)

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                input_values = inputs.input_values.to(device)
                attention_mask = inputs.attention_mask.to(device)
                labels = labels.to(device)
                outputs = model(input_values, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item() * labels.size(0)
                preds = outputs.logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
        avg_val = val_loss / len(val_loader.dataset)
        acc = correct / len(val_loader.dataset)
        scheduler.step(avg_val)
        print(f"Epoch {epoch} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val Acc: {acc:.4f}")
