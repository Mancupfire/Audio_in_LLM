import os
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, AutoTokenizer, AutoModel
import transformers
from peft import LoraConfig, get_peft_model, IA3Config
import time
import collections
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split


import random
from torch.utils.data import Sampler


def train_test_split_from_list(data_list, train_ratio=0.8, shuffle=True):
    if shuffle:
        random.shuffle(data_list)
    split_idx = int(len(data_list) * train_ratio)
    train_data = data_list[:split_idx]
    test_data = data_list[split_idx:]
    return train_data, test_data

def plot_tsne(features, labels, title="t-SNE Visualization", perplexity=30, n_iter=1000, random_state=42):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    # Reduce dimensions to 2D using t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)
    features_2d = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()


class CategoriesSampler(Sampler):
    def __init__(self, labels, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        
        # Build a dictionary mapping each label to a list of indices.
        self.m_ind = {}
        for idx, label in enumerate(labels):
            self.m_ind.setdefault(label, []).append(idx)
        
        self.classes = list(self.m_ind.keys())

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            # Randomly sample n_cls classes (without replacement)
            selected_classes = random.sample(self.classes, self.n_cls)
            for cls in selected_classes:
                # Randomly sample n_per indices from the current class
                batch.extend(random.sample(self.m_ind[cls], self.n_per))
            yield batch


class SplitCategoriesSampler(Sampler):
    def __init__(self, labels, n_batch, n_cls, n_support, n_query):

        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_support = n_support
        self.n_query = n_query
        
        self.m_ind = {}
        for idx, label in enumerate(labels):
            self.m_ind.setdefault(label, []).append(idx)
        
        self.classes = list(self.m_ind.keys())

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            support = []
            query = []
            selected_classes = random.sample(self.classes, self.n_cls)
            for cls in selected_classes:
                # Sample (n_support + n_query) indices from each class
                indices = random.sample(self.m_ind[cls], self.n_support + self.n_query)
                support.extend(indices[:self.n_support])
                query.extend(indices[self.n_support:])
            yield support, query


class TrainCategoriesSampler(Sampler):
    def __init__(self, labels, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        
        self.m_ind = {}
        for idx, label in enumerate(labels):
            self.m_ind.setdefault(label, []).append(idx)
        
        self.classes = list(self.m_ind.keys())

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            # Here we use random.choices to allow the same class to be picked more than once.
            selected_classes = random.choices(self.classes, k=self.n_cls)
            for cls in selected_classes:
                batch.extend(random.sample(self.m_ind[cls], self.n_per))
            yield batch


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data, from_npy=False, from_audio=False, prompt=""):
        self.data = data[0]
        self.metadata = data[1]
        self.label = data[2]

        # self.max_len = max_len
        # self.augment = augment
        self.from_npy = from_npy
        # self.crop_mode = crop_mode
        self.from_audio = from_audio
        self.prompt = prompt

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.from_npy:
            npy_path = self.data[idx]
            x = np.load(npy_path + ".npy")
        else:
            x = self.data[idx]

        label = self.label[idx]
        metadata =  self.metadata[idx]

        if self.from_audio:
            return x, label

        x = torch.tensor(x, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)

        return x, self.prompt, metadata, label


def get_prompt(configs, dataset="ssbpr", label="snoring", modality="snoring"):
    data_description = {
        "ssbpr": "This data comes from a snore-based sleep body position recognition dataset (SSBPR).",
        "covid19sounds": "This data comes from the COVID-19 Sounds dataset.",
        "coviduk": "This data comes from the UK COVID-19 Vocal Audio Dataset.",
        "icbhidisease": "This data comes from the ICBHI Respiratory Sound Database Dataset.",
        "coswara": "This data comes from the Coswara Covid-19 dataset. ",
        "kauh": "This data comes from the KAUH lung sound dataset, containing lung sounds recorded from the chest wall using an electronic stethoscope.",
        "copd": "This data comes from the RespiratoryDatabase@TR dataset, containing auscultation sounds recorded from the chest wall using an electronic stethoscope.",
        "coughvid": "This data comes from the CoughVID dataset. ",
    }
    task_description = {
        "snoring": "body position of the participant",
        "symptom": "whether the participant has respiratory symptoms (dry cough, wet cough, fever, sore throat, shortness of breath, runny nose, headache, dizziness, or chest tightness)",
        "covid": "whether the participant has COVID-19",
        "copd": " whether the person has Chronic obstructive pulmonary disease (COPD)",
        "smoker": "whether the person is a smoker or not",
        "obstructive": "whether the person has obstructive respiratory disease including asthma and COPD",
        "copdlevel": "the severity of the COPD patient",
        "asthma": " whether the person has asthma",

    }
    classes = {
        "snoring": "supine, supine but left lateral head, supine but right lateral head, left-side lying, right-side lying",
        "symptom": "symptomatic, asymptomatic",
        "covid": "COVID19, non-COVID19",
        "copd": "COPD, healthy",
        "smoker": "smoker, non-smoker",
        "obstructive": "obstructive, healthy",
        "copdlevel": "COPD 0, COPD 1, COPD 2, COPD 3, COPD 4",
        "asthma": "asthma, healthy"

    }
    n_cls = len(classes[label].split(","))
    if configs.use_audio:
        prompt = (
                    f"<|start_prompt|>"
                    f"Dataset description: {data_description[dataset]} "
                    f"Task description: classifiy {task_description[label]} given the following information and audio of the person's {modality} sounds. "
                    f"The {n_cls} classes are: {classes[label]}. "
                    f"Please output the class index, from 0 to {n_cls-1}."
                    "<|<end_prompt>|>"
                )
    else:
        prompt = (
            f"<|start_prompt|>Dataset description: {data_description[dataset]} "
            f"Task description: classifiy {task_description[label]} given the following information."
            f"The {n_cls} classes are: {classes[label]}. <|<end_prompt>|>"
        )
    return prompt


def get_context(metadata):
    context = ""
    if "gender" in metadata:
        context += "Gender: {}. ".format(metadata["gender"])
    if "age" in metadata:
        context += "Age: {}. ".format(metadata["age"])
    # if "location" in metadata:
    #     context += "Recording location: {}. ".format(metadata["location"])
    if "device" in metadata:
        context += "Recording device: {}. ".format(metadata["device"])
    
    if "vaccination" in metadata:
        context += "Vaccination status: {}. ".format(metadata["vaccination"])

    if "location" in metadata:
        l = metadata["location"]
        if len(l) == 2:
            location_dict = {
                "1": "posterior-upper lung", 
                "2": "posterior-middle lung", 
                "3": "posterior-lower lung", 
                "4": "posterior-inferior lung",
                "5": "posterior-costophrenic angle lung",
                "6": "anterior-lower lung",
                # ICBHI
                'Tc': 'trachea', 
                'Al': 'left anterior chest',
                'Ar': 'right anterior chest', 
                'Pl': 'left posterior chest', 
                'Pr': 'right posterior chest', 
                'Ll': 'left lateral chest', 
                'Lr': 'right lateral chest',
            }
            if l[1].isnumeric():
                location = "left" if l[0] == "L" else "right"
                location += location_dict[l[1]]
            else:
                location = location_dict[l]
        else:
            l = l.replace(" ", "")
            l = l.replace("PLR", "PRL")
            l1_dict = {"P" : "posterior", "A": "anterior"}
            l2_dict = {"L": "left", "R": "right"}
            l3_dict = {"U": "upper", "M": "middle", "L": "lower"}
            try:
                location = l1_dict[l[0]]
                location += " " + l2_dict[l[1]]
                location += " " + l3_dict[l[2]]
            except:
                print(l)

        context += f"Record location: {location}."

    if "medhistory" in metadata:
        userMedHistory = metadata["medhistory"]
        # print(userMedHistory)
        med_history_dict = { 
            'angina':"Angina",
            'asthma': "Asthma",
            'cancer':"Cancer",
            'copd': "COPD/Emphysema",
            'cystic': "Cystic fibrosis", 
            'diabetes': "Diabetes", 
            'hbp': "High Blood Pressure", 
            'heart': "Previous heart attack",
            'hiv': "HIV or impaired immune system",
            'long': "Other long-term condition",
            'longterm': "Other long-term condition",
            'lung':"Other lung disease",
            'otherHeart': "Other heart disease",
            'organ': "Previous organ transplant",
            'pulmonary':"Pulmonary fibrosis", 
            'stroke': "Previous stroke or Transient ischaemic attack", 
            'valvular': "Valvular heart disease",
            "respiratory_condition_asthma": "asthma",
            "respiratory_condition_other": "other respiratory health condition"
            }
        if pd.isna(userMedHistory) or userMedHistory == "" or "None" in userMedHistory or "none" in userMedHistory:
            # print(row)
            context += "Patient presents with no medical history conditions. "
        elif "pnts" in userMedHistory:
            pass
        else:
            if userMedHistory[0] == ",": userMedHistory = userMedHistory[1:]
            if userMedHistory[-1] == ",": userMedHistory = userMedHistory[:-1]
            context += "Patient presents with the following medical history conditions: " 
            # print(userMedHistory)
            context += ", ".join([med_history_dict[med].lower() for med in userMedHistory.split(",")])  + ". "
    
    if "symptoms" in metadata:
        userSymptoms = metadata["symptoms"]
        symptoms_dict = { 
            'drycough': "Dry cough", 
            'wetcough': "Wet cough", 
            'sorethroat': "Sore throat", 
            'runnyblockednose': "Runny or blocked nose",
            'runny': "Runny or blocked nose",
            'tightness': "Tightness in the chest", 
            'smelltasteloss': "Loss of taste and smell", 
            'fever': "Fever", 
            'chills': "Chills",  
            'shortbreath': "Difficulty breathing or feeling short of breath", 
            'dizziness': "Dizziness, confusion or vertigo", 
            'headache': "Headache", 
            'muscleache': "Muscle aches",
            # covid uk
            "cough_any": "cough",
            "new_continuous_cough": "a new continuous cough", 
            "runny_or_blocked_nose": "runny or blocked nose", 
            "shortness_of_breath": "shortness of breath", 
            "sore_throat": "sore throat", 
            "abdominal_pain": "abdominal pain", 
            "diarrhoea": "diarrhoea", 
            "fatigue": "fatigue", 
            "fever_high_temperature": "fever or a high temperature", 
            "headache": "headache", 
            "change_to_sense_of_smell_or_taste": "a change to sense of smell or taste", 
            "loss_of_taste": "loss of sense of taste",
            # coswara
            "cold": "cold", 
            "cough": "cough", 
            "fever":"fever", 
            "diarrhoea": "diarrhoea",
            "st": "sore throat", 
            "loss_of_smell": "loss of smell and/or taste", 
            "mp": "muscle pain", 
            "ftg": "fatigue", 
            "bd": "breathing difficulties",
            # coughvid
            "fever_muscle_pain": "fever or muscle pain"
            }

        if pd.isna(userSymptoms) or userSymptoms == "" or "None" in userSymptoms:
            # print(row)
            context += "Patient presents with no obvious respiratory symptoms."
        elif "pnts" in userSymptoms:
            pass
        else:
            if userSymptoms[0] == ",": userSymptoms = userSymptoms[1:]
            context += "Patient presents with the following respiratory symptoms: " 
            context += ", ".join([symptoms_dict[sym].lower() for sym in userSymptoms.split(",")])  + ". "

    # print(context)
    return context


def downsample_balanced_dataset(x_train, metadata_train, y_train):
    # Find unique classes in y_train
    classes = np.unique(y_train)

    # Find the minimum number of samples among classes
    min_samples = min(np.bincount(y_train))

    # Initialize lists to store downsampled data
    x_downsampled = []
    metadata_downsampled = []
    y_downsampled = []

    # Downsample each class
    for c in classes:
        # Get indices of samples belonging to class c
        indices = np.where(y_train == c)[0]

        # Randomly select min_samples samples
        selected_indices = np.random.choice(
            indices, min_samples, replace=False)

        # Add selected samples to downsampled data
        x_downsampled.extend(x_train[selected_indices])
        metadata_downsampled.extend(metadata_train[selected_indices])
        y_downsampled.extend(y_train[selected_indices])

    # Convert lists to numpy arrays
    x_downsampled = np.array(x_downsampled)
    metadata_downsampled = np.array(metadata_downsampled)
    y_downsampled = np.array(y_downsampled)

    return x_downsampled, metadata_downsampled, y_downsampled


def upsample_balanced_dataset(x_train, metadata_train, y_train):
    # print(x_train.shape, metadata_train.shape, y_train.shape)
    from sklearn.utils import resample, shuffle

    # Separate the dataset into classes
    class_0 = x_train[y_train == 0]
    metadata_0 = metadata_train[y_train == 0]
    class_1 = x_train[y_train == 1]
    metadata_1 = metadata_train[y_train == 1]

    # Find the size of the larger class
    size_0 = len(class_0)
    size_1 = len(class_1)
    max_size = max(size_0, size_1)

    # Upsample the smaller class
    if size_0 < size_1:
        # print(metadata_0.shape)
        class_0_upsampled, metadata_0_upsampled = resample(class_0, metadata_0, replace=True, n_samples=max_size, random_state=42)
        # print(metadata_0_upsampled.shape)
        class_1_upsampled, metadata_1_upsampled = class_1, metadata_1
        # print(metadata_1_upsampled.shape)
        y_class_0 = np.zeros(max_size)
        y_class_1 = y_train[y_train == 1]
    else:
        class_1_upsampled, metadata_1_upsampled = resample(class_1, metadata_1, replace=True, n_samples=max_size, random_state=42)
        class_0_upsampled, metadata_0_upsampled = class_0, metadata_0
        y_class_1 = np.ones(max_size)
        y_class_0 = y_train[y_train == 0]

    # Combine the upsampled classes
    x_train_upsampled = np.concatenate((class_0_upsampled, class_1_upsampled))
    metadata_upsampled = np.concatenate((metadata_0_upsampled, metadata_1_upsampled))
    y_train_upsampled = np.concatenate((y_class_0, y_class_1))

    # print(metadata_upsampled.shape)

    # Shuffle the upsampled dataset
    x_train_upsampled, metadata_upsampled, y_train_upsampled = shuffle(x_train_upsampled, metadata_upsampled, y_train_upsampled, random_state=42)

    print("Balanced dataset sizes:")
    print(f"Class 0: {len(y_train_upsampled[y_train_upsampled == 0])}")
    print(f"Class 1: {len(y_train_upsampled[y_train_upsampled == 1])}")
    return x_train_upsampled, metadata_upsampled, y_train_upsampled


import librosa
import numpy as np

def get_split_signal_librosa(file_path, sr=22050, top_db=60, min_duration=0.0):
    # Load the audio file. `y` is the audio time series, and `sr` is the sample rate.
    y, sr = librosa.load(file_path, sr=sr)
    
    # Use librosa's effects.split to get intervals where the signal is above the silence threshold.
    # `intervals` is an array of shape (n_intervals, 2), where each row represents [start, end] in samples.
    intervals = librosa.effects.split(y, top_db=top_db)

    # Initialize an empty list to collect segments.
    segments = []
    
    # Convert minimum duration (in seconds) to minimum number of samples.
    min_samples = int(min_duration * sr)
    
    # Iterate over the intervals and extract the segments from the audio signal.
    for start, end in intervals:
        # Only keep the segment if it meets the minimum duration requirement.
        if (end - start) >= min_samples:
            segment = y[start:end]
            segments.append(segment)
    
    return segments, intervals, y, sr

import torch

def initialize_pretrained_model(model_class, pretrained=True, weights_path=None, device=None, **kwargs):
    # Determine device if not provided
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if pretrained and (weights_path is None):
        try:
            model = model_class(pretrained=True, **kwargs)
        except TypeError:
            model = model_class(**kwargs)
    else:
        if "pretrained" in model_class.__init__.__code__.co_varnames:
            model = model_class(pretrained=False, **kwargs)
        else:
            model = model_class(**kwargs)

        # If a weights_path is provided, load the weights from the checkpoint.
        if weights_path is not None:
            state_dict = torch.load(weights_path, map_location=device)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # Remove 'module.' prefix if present
                new_key = k[7:] if k.startswith("module.") else k
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict)

    model.to(device)
    return model


