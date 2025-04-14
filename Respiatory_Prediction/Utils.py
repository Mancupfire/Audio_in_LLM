import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torchmetrics import AUROC
from tqdm import tqdm
import os
import collections
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, ConcatDataset
import torch.optim as optim
import pickle
import copy
from numpy import linalg as LA
import random
from Support_Utils import AudioDataset, get_prompt, get_context, downsample_balanced_dataset, upsample_balanced_dataset, CategoriesSampler, SplitCategoriesSampler, TrainCategoriesSampler, train_test_split_from_list, plot_tsne
from Support_Utils import get_split_signal_librosa

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def get_dataloader(configs, task, sample=False, deft_seed=None):
    tasks_config = {
        "S1": ("coviduk", "covid", "exhalation"),
        "S2": ("coviduk", "covid", "cough"),
        "S3": ("covid19sounds", "covid", "breath"),
        "S4": ("covid19sounds", "covid", "cough"),
        "S5": ("covid19sounds", "smoker", "breath"),
        "S6": ("covid19sounds", "smoker", "cough"),
        "S7": ("icbhidisease", "copd", "lung"),
        "T1": ("coswara", "covid", "cough-shallow"),
        "T2": ("coswara", "covid", "cough-heavy"),
        "T3": ("coswara", "covid", "breathing-shallow"),
        "T4": ("coswara", "covid", "breathing-deep"),
        "T5": ("kauh", "copd", "lung"),
        "T6": ("kauh", "asthma", "lung"),

    }
    n_cls = collections.defaultdict(lambda:2, {"11": 5, "12": 5})

    dataset, label, modality = tasks_config[task]
    
    feature_dirs = {"covid19sounds": "feature/covid19sounds_eval/downsampled/", 
                    "coswara": "feature/coswara_eval/",
                    "coviduk": "feature/coviduk_eval/",
                    "kauh": "feature/kauh_eval/",
                    "icbhidisease": "feature/icbhidisease_eval/",
                    }
    
    pad_len_htsat = {"covid19sounds": 8.18, 
                    "coswara": 8.18,
                    "coviduk": 8.18,
                    "kauh": 8.18,
                    "icbhidisease": 8.18,
                    }
    feature_dir = feature_dirs[dataset]
    if task in ["S3", "S4"]:
        feature_dir = "feature/covid19sounds_eval/covid_eval/"
    elif task in ["S5", "S6"]:
        feature_dir = "feature/covid19sounds_eval/smoker_eval/"
    
    if dataset in ["ssbpr", "copd", "kauh", "icbhidisease"]:
        suffix_dataset =  ".npy"
    elif dataset in ["covid19sounds", "coviduk"]:
        suffix_dataset = "_{}.npy".format(modality)
    elif dataset in ["coswara"]:
        suffix_dataset = "_{}_{}.npy".format(modality, label)
    elif dataset in ["coughvid"]:
        suffix_dataset = "_{}.npy".format(label)
    else:
        raise NotImplementedError

    # prompt
    prompt = get_prompt(configs, dataset=dataset, label=label, modality=modality)
    print(prompt)

    # metadata
    if dataset == "coviduk":
        y_label = np.load(feature_dir + f"label_{modality}.npy")
    elif dataset == "coswara":
        broad_modality = modality.split("-")[0]
        y_label = np.load(feature_dir + "{}_aligned_{}_label_{}.npy".format(broad_modality, label, modality))
    elif dataset == "kauh":
        y_label = np.load(feature_dir + "labels_both.npy")
        if label == "copd":
            label_dict = {"healthy": 0, "asthma": 2, "COPD": 1, "obstructive": 2}
            y_label = np.array([label_dict[y] for y in y_label])
        elif label == "asthma":
            label_dict = {"healthy": 0, "asthma": 1, "COPD": 2, "obstructive": 2}
            y_label = np.array([label_dict[y] for y in y_label])
        else:
            label_dict = {"healthy": 0, "asthma": 1, "COPD": 1, "obstructive": 1}
            y_label = np.array([label_dict[y] for y in y_label])
    elif dataset == "coughvid":
        y_label = np.load(feature_dir + "label_{}.npy".format(label))
    else:
        y_label = np.load(feature_dir + "labels.npy")
    
    if dataset == "coswara":
        sound_dir_loc = np.load(
        feature_dir + "{}_aligned_filenames_{}_w_{}.npy".format(broad_modality, label, modality))
    elif dataset == "kauh":
        sound_dir_loc = np.load(feature_dir + "sound_dir_loc_subset.npy")
    else:
        sound_dir_loc = np.load(feature_dir + "sound_dir_loc" + suffix_dataset)

    if configs.use_context:
        if dataset == "ssbpr":
            gender = [filename.split("/")[-3] for filename in sound_dir_loc]
            metadata = [{'gender': gender[i]} for i in range(len(gender))]
        elif dataset == "covid19sounds":
            if task in ["S3", "S4"]:
                import glob as gb
                metadata = []
                data_df = pd.read_csv("datasets/covid19-sounds/data_0426_en_task2.csv")
                metadata_dir = np.array(gb.glob("datasets/covid19-sounds/covid19_data_0426_metadata/*.csv"))
                df = None
                # use metadata as outer loop to enable quality check
                for file in metadata_dir:
                    if df is None:
                        df = pd.read_csv(file, delimiter=";")
                    else:
                        df = pd.concat([df, pd.read_csv(file, delimiter=";")])
                    # print(len(df))
                        # df.append(pd.read_csv(file, delimiter=";"))
                    
                    # if "cough" in modality:
                    #     df = df[df["Cough check"].str.contains("c")]
                    # if "breath" in modality:
                    #     df = df[df["Breath check"].str.contains("b")]
                    # if "voice" in modality:
                    #     df = df[df["Voice check"].str.contains("v")]
                # df = df.set_index(["Uid", "Folder Name"])
                # print(df.head(10))
                for _, data_row in data_df.iterrows():
                    try:
                        # print(data_row)
                        uid = data_row["cough_path"].split("\\")[0]
                        # print(uid)
                        folder_name = data_row["cough_path"].split("\\")[1]
                        # print(folder_name)
                        row = df[df['Uid'] == uid]
                        row = row[row['Folder Name'] == folder_name]
                        row = row.iloc[0]
                        # print(row)
                        metadata.append({'age': row["Age"], 
                                    "gender": row["Sex"], 
                                    "medhistory": row["Medhistory"], 
                                    "symptoms": row["Symptoms"]
                                    } )
                    except IndexError:
                        print("metadata nonexist", uid, folder_name)
                        metadata.append({} )
                    # symptoms = df.at[(uid, folder_name), "Symptoms"]
                    # metadata.append({'age': df.at[(uid, folder_name), "Age"],
                    #             "gender": df.at[(uid, folder_name), "Sex"],
                    #             "medhistory": df.at[(uid, folder_name), "Medhistory"], 
                    #             "symptoms": df.at[(uid, folder_name), "Symptoms"]
                    #             } )
            elif task in ["S5", "S6"]:
                df = pd.read_csv("datasets/covid19-sounds/data_0426_en_task1.csv", delimiter=";", index_col="Uid")
                # print(df.head(5))
                df = df[~df.index.duplicated(keep='first')]
                metadata = []
                for filename in sound_dir_loc:
                    uid = filename.split("/")[-3]
                    if uid == "form-app-users":
                        uid = filename.split("/")[-2]
                    row = df.loc[uid]
                    # print(row)
                    metadata.append({'age': row["Age"], 
                                    "gender": row["Sex"], 
                                    "medhistory": row["Medhistory"], 
                                     "symptoms": row["Symptoms"]
                                    } )
            else:
                raise NotImplementedError(f"task {task} not implemented")
        elif dataset == "coviduk":
            participant_data = pd.read_csv("datasets/covidUK/audio_metadata.csv")
            split_data = pd.read_csv("datasets/covidUK/participant_metadata.csv")
            df = pd.merge(participant_data, split_data, on='participant_identifier')
            metadata = []
            # df.fillna(0)
            df = df.replace(np.nan, 0, regex=True)
            for filename in sound_dir_loc:
                audio_name = filename.split("/")[-1]
                row = df.loc[df[f"{modality}_file_name"] == audio_name]
                # print(row)
                # print(int(row["respiratory_condition_asthma"]))
                medhistory = ",".join([med for med in ["respiratory_condition_asthma", "respiratory_condition_other"] if int(row[med])])

                symptoms = ""
                if int(row["symptom_none"]):
                    symptoms = "None"
                elif int(row["symptom_prefer_not_to_say"]):
                    symptoms = "pnts"
                else:
                    syms = ["cough_any", "new_continuous_cough", "runny_or_blocked_nose", "shortness_of_breath", "sore_throat", "abdominal_pain", "diarrhoea", "fatigue", "fever_high_temperature", "headache", "change_to_sense_of_smell_or_taste", "loss_of_taste"]
                    symptoms = ",".join([sym for sym in syms if int(row["symptom_" + sym])])
                    # print(symptoms)
                
                metadata.append({'age': row["age"].values[0], 
                                "gender": row["gender"].values[0], 
                                "smoke status": row["smoker_status"].values[0],
                                "medhistory": medhistory, 
                                 "symptoms": symptoms
                                } )
        elif dataset == "copd":
            location = [filename.split("/")[-1][5:7] for filename in sound_dir_loc]
            metadata = [{"location": location[i]} for i in range(len(location))]
        elif dataset == "kauh":
            metadata = []
            for filename in sound_dir_loc:
                location = filename.split(",")[-3]
                gender = filename.split(",")[-1].split(".")[0]
                metadata.append({
                    "location": location,
                    "gender": gender
                })
        elif dataset == "icbhidisease":
            metadata = []
            df = pd.read_csv('datasets/icbhi/ICBHI_Challenge_demographic_information.txt',
                                dtype=str, sep='\t', names=['userId', 'Age', 'Sex', 'Adult_BMI', 'Child Weight', 'Child Height'],  index_col="userId")
            for filename in sound_dir_loc:
                userID = int(filename.split("/")[-1].split("_")[0])
                location = filename.split("/")[-1].split("_")[2]
                row = df.loc[userID]
                metadata.append({
                    "age": row["Age"],
                    "gender": row["Sex"],
                    "location": location
                })
        elif dataset == "coswara":
            df = pd.read_csv("datasets/Coswara-Data/combined_data.csv", index_col="id")
            metadata = []
            for filename in sound_dir_loc:
                uid = filename.split("/")[-2]
                row = df.loc[uid]
                # print(row)
                syms = ["cold", "cough", "fever", "diarrhoea", "st", "loss_of_smell", "mp", "ftg", "bd"]
                symptoms = ",".join([sym for sym in syms if row[sym] is True])
                # print(symptoms)
                metadata.append({
                    "age": row["a"],
                    "gender": row["g"],
                    "symptoms": symptoms,
                    # TODO : vacc?
                })
        elif dataset == "coughvid":
            df = pd.read_csv("datasets/coughvid/metadata_compiled.csv", index_col="uuid")
            metadata = []
            for filename in sound_dir_loc:
                uid = filename.split("/")[-1][:-4]
                
                try:
                    row = df.loc[uid]
                    metadata.append({
                        "age": row["age"],
                        "gender": row["gender"],
                        "symptoms": "fever_muscle_pain" if row["fever_muscle_pain"] else "",
                    })
                except KeyError:
                    print(uid, "not found")
                    metadata.append({})
        else:
            print("metadata not included for", dataset)
            metadata = np.array([{} for x in range(len(sound_dir_loc))])
        # for key in metadata[0]:
        #     print(key, collections.Counter([data[key] for data in metadata]))
        x_metadata = np.array([get_context(d) for d in metadata])
    
    else:
        x_metadata = np.array(["" for x in range(len(sound_dir_loc))])

    for sample in [0, 11, 12, 13, 24, 34, 666, 717, 1024][:]:
        if sample < len(y_label):
            print("sound_dir_loc", sound_dir_loc[sample])
            print(x_metadata[sample])
            print("y_label", y_label[sample])
    
    # audio data
    from_audio = False

    spec_file_name = feature_dir + f"spectrogram_pad{str(int(pad_len_htsat[dataset]))}" + suffix_dataset if dataset != "icbhidisease" else feature_dir + f"segmented_spectrogram_pad{str(int(pad_len_htsat[dataset]))}" + suffix_dataset
    if not os.path.exists(spec_file_name):
        x_data = []
        if dataset == "icbhidisease":
            y_segmented, y_set_segmented = [], []
            # x_metadata_segmented = []
            index_segmented = []
            y_set = np.load(feature_dir + "split.npy")
            for idx, audio_file in enumerate(sound_dir_loc):
                data = get_split_signal_librosa("", audio_file[:-4], spectrogram=True, input_sec=pad_len_htsat[dataset], trim_tail=False)
                if y_set[idx] == "train":
                    # print([y_set[idx]], len(data))
                    x_data.extend(data)
                    y_segmented.extend([y_label[idx]] * len(data))
                    y_set_segmented.extend([y_set[idx]] * len(data))
                    # x_metadata_segmented.extend([x_metadata[idx]] * len(data))
                    index_segmented.extend([idx] * len(data))
                else:
                    # print([y_set[idx]])
                    x_data.append(data[0])
                    y_segmented.append(y_label[idx])
                    y_set_segmented.append(y_set[idx])
                    # x_metadata_segmented.append([x_metadata[idx]])
                    index_segmented.append(idx)
            x_data = np.array(x_data)
            y_segmented = np.array(y_segmented)
            y_set_segmented = np.array(y_set_segmented)
            np.save(spec_file_name, x_data)
            np.save(feature_dir + f"segmented_split.npy", y_set_segmented)
            np.save(feature_dir + f"segmented_labels.npy", y_segmented)
            np.save(feature_dir + f"segmented_index.npy", index_segmented)
        else:
            for audio_file in sound_dir_loc:
                data = get_split_signal_librosa("", audio_file[:-4], spectrogram=True, input_sec=pad_len_htsat[dataset], trim_tail=False)[0]
                # print(data.shape)
                x_data.append(data)
            x_data = np.array(x_data)
            np.save(spec_file_name, x_data)

    seed = 42

    if dataset == "icbhidisease":
        x_data = np.load(feature_dir + f"segmented_spectrogram_pad{str(int(pad_len_htsat[dataset]))}" + suffix_dataset)
        y_label = np.load(feature_dir + f"segmented_labels.npy")
        index_sampled = np.load(feature_dir + f"segmented_index.npy")
        x_metadata = x_metadata[index_sampled]
    else:
        x_data = np.load(feature_dir + f"spectrogram_pad{str(int(pad_len_htsat[dataset]))}" + suffix_dataset)
    print(len(x_data), len(x_metadata), len(y_label))
    
    print(collections.Counter(y_label))
    
    if dataset == "covid19sounds":
        y_set = np.load(feature_dir + "data_split.npy")
        
        if task in ["S3", "S4"]:
            x_data_train = x_data[y_set == "train"]
            x_metadata_train = x_metadata[y_set == "train"]
            y_label_train = y_label[y_set == "train"]
            
            x_data_vad = x_data[y_set == "validation"]
            x_metadata_vad = x_metadata[y_set == "validation"]
            y_label_vad = y_label[y_set == "validation"]

            x_data_test = x_data[y_set == "test"]
            x_metadata_test = x_metadata[y_set == "test"]
            y_label_test = y_label[y_set == "test"]
        else:
            x_data_train = x_data[y_set == 0]
            x_metadata_train = x_metadata[y_set == 0]
            y_label_train = y_label[y_set == 0]
            
            x_data_vad = x_data[y_set == 1]
            x_metadata_vad = x_metadata[y_set == 1]
            y_label_vad = y_label[y_set == 1]

            x_data_test = x_data[y_set == 2]
            x_metadata_test = x_metadata[y_set == 2]
            y_label_test = y_label[y_set == 2]
        
    elif dataset == "coswara":
        if True: #label == "covid":
            set_all_seed(seed)
            symptoms = np.array([1 if 'following respiratory symptoms' in m else 0 for m in x_metadata])
            np.save(feature_dir + f"symptom" + suffix_dataset, symptoms)
            # symptoms = np.load(feature_dir + f"symptom" + suffix_dataset)

            group1_indices = np.where((y_label == 0) & (symptoms == 1))[0]
            group2_indices = np.where((y_label == 0) & (symptoms == 0))[0]
            group3_indices = np.where((y_label == 1) & (symptoms == 1))[0]
            group4_indices = np.where((y_label == 1) & (symptoms == 0))[0]
            random.seed(seed)

            test_size = np.min([len(group) for group in [group1_indices, group2_indices, group3_indices, group4_indices]]) - (configs.meta_val_shot // 2)

            def sample_indices(group_indices, test_size):
                print(f"sampling {test_size} from", len(group_indices))
                test_sample_indices = np.random.choice(group_indices, size=test_size, replace=False)
                remaining_indices = np.setdiff1d(group_indices, test_sample_indices)
                return test_sample_indices, remaining_indices
        
            # Step 2: Sample 30 indices for each group for the test set
            group1_indices_test, group1_indices_train = sample_indices(group1_indices, test_size)
            group2_indices_test, group2_indices_train = sample_indices(group2_indices, test_size)
            group3_indices_test, group3_indices_train = sample_indices(group3_indices, test_size)
            group4_indices_test, group4_indices_train = sample_indices(group4_indices, test_size)

            # Combine test and training indices
            indices_test = np.concatenate([group1_indices_test, group2_indices_test, group3_indices_test, group4_indices_test])
            indices_train = np.concatenate([group1_indices_train, group2_indices_train, group3_indices_train, group4_indices_train])

            print("train")
            for indices_array in [group1_indices_train, group2_indices_train, group3_indices_train, group4_indices_train]:
                print(len(indices_array), end=";")
            print("\ntest")
            for indices_array in[group1_indices_test, group2_indices_test, group3_indices_test, group4_indices_test]:
                print(len(indices_array), end=";")
            print()
            # Step 3: Use the sampled indices to get the test and training data
            x_data_test = x_data[indices_test]
            x_metadata_test = x_metadata[indices_test]
            y_label_test = y_label[indices_test]

            x_data_train = x_data[indices_train]
            x_metadata_train = x_metadata[indices_train]
            y_label_train = y_label[indices_train]

            x_data_vad, x_metadata_vad, y_label_vad = x_data_train, x_metadata_train, y_label_train 

            group_idxs = []
            for i in range(len(x_data_train)):
                y = y_label_train[i]
                m = x_metadata_train[i]
                if y == 0 and 'following respiratory symptoms' in m:
                    group = 1
                if y == 0 and 'following respiratory symptoms' not in m:
                    group = 2
                if y == 1 and 'following respiratory symptoms' in m:
                    group = 3
                if y == 1 and 'following respiratory symptoms' not in m:
                    group = 4
                group_idxs.append(group)
        
            group_idxs = np.array(group_idxs)

    elif dataset == "kauh":
        y_set = np.load(feature_dir + "train_test_split.npy")
        if label in ["copd", "asthma"]:
            mask = (y_label == 0) | (y_label == 1)
            y_label = y_label[mask]
            y_set = y_set[mask]
            x_data = x_data[mask]
            x_metadata = x_metadata[mask]
        x_data_train, x_data_test, _, _ = train_test_split_from_list(x_data, y_label, y_set)
        x_metadata_train, x_metadata_test, y_label_train, y_label_test = train_test_split_from_list(x_metadata, y_label, y_set)
        x_data_train, x_data_vad, x_metadata_train, x_metadata_vad, y_label_train, y_label_vad = train_test_split(
                x_data_train, x_metadata_train, y_label_train, test_size=0.1, 
                random_state=1337, stratify=y_label_train
            )
    elif dataset == "icbhidisease":
        # y_set = np.load(feature_dir + "split.npy")
        y_set = np.load(feature_dir + "segmented_split.npy")
        mask = (y_label == "Healthy") | (y_label == "COPD")
        y_label = y_label[mask]
        y_set = y_set[mask]
        x_data = x_data[mask]
        x_metadata = x_metadata[mask]
        label_dict = {"Healthy": 0, "COPD": 1}
        y_label = np.array([label_dict[y] for y in y_label])

        x_data_train, x_data_test, y_label_train, y_label_test = train_test_split_from_list(x_data, y_label, y_set)
        x_metadata_train, x_metadata_test, y_label_train, y_label_test = train_test_split_from_list(x_metadata, y_label, y_set)

        x_data_train, x_data_vad, x_metadata_train, x_metadata_vad, y_label_train, y_label_vad = train_test_split(
                x_data_train, x_metadata_train, y_label_train, test_size=0.2, 
                random_state=1337, stratify=y_label_train
            )

    else:
        if dataset == "coviduk":
            y_set = np.load(feature_dir + "split_{}.npy".format(modality))
        x_data_train = x_data[y_set == "train"]
        y_label_train = y_label[y_set == "train"]
        x_metadata_train = x_metadata[y_set == "train"]

        x_data_vad = x_data[y_set == "val"]
        y_label_vad = y_label[y_set == "val"]
        x_metadata_vad = x_metadata[y_set == "val"]

        x_data_test = x_data[y_set == "test"]
        y_label_test = y_label[y_set == "test"]
        x_metadata_test = x_metadata[y_set == "test"]
        # split = np.load(feature_dir + "split.npy")


    if task in ["S5", "S6", "T6"]:
        x_data_train, x_metadata_train, y_label_train = downsample_balanced_dataset(x_data_train, x_metadata_train, y_label_train)
    if task in ["S7"]:
        x_data_train, x_metadata_train, y_label_train = upsample_balanced_dataset(x_data_train, x_metadata_train, y_label_train)
    if task in ["T6"]:
        x_data_train, x_metadata_train, y_label_train = downsample_balanced_dataset(x_data_train, x_metadata_train, y_label_train)
        x_data_test, x_metadata_test, y_label_test = downsample_balanced_dataset(x_data_test, x_metadata_test, y_label_test)

        x_data_train, x_metadata_train, y_label_train, x_data_test, x_metadata_test, y_label_test = x_data_test, x_metadata_test, y_label_test, x_data_train, x_metadata_train, y_label_train


    # !! didn't split the metadata as needed, all results were wrong
    train_data_percentage = configs.train_pct
    if not sample and train_data_percentage < 1:
        x_data_train, _, y_label_train, _, x_metadata_train, _ = train_test_split(
            x_data_train, y_label_train, x_metadata_train, test_size=1 - train_data_percentage, random_state=seed, stratify=y_label_train
        )

    print(collections.Counter(y_label_train))
    min_train_cls = min(collections.Counter(y_label_train).values())
    print(collections.Counter(y_label_vad))
    print(collections.Counter(y_label_test))
    min_test_cls = min(collections.Counter(y_label_test).values())

    # print(x_metadata_test)

    if sample and configs.few_shot:
        sample_info = [configs.meta_val_iter, n_cls[task], configs.meta_val_shot, configs.meta_val_query]
        if min_train_cls < configs.meta_val_shot:
            sample_info[2] = min_train_cls
        if min_test_cls < configs.meta_val_query:
            sample_info[3] = min_test_cls

        if dataset == "coswara" and label =="covid":
            sample_info = [configs.meta_val_iter, n_cls[task] * 2, configs.meta_val_shot // 2, configs.meta_val_query]
            # all_data = AudioDataset((np.array(x_data_train + x_data_test), np.array(x_metadata_train + x_metadata_test), np.array(y_label_train + y_label_test)),  from_audio=from_audio, prompt=prompt)
            sampler = TrainCategoriesSampler(group_idxs, *sample_info)
        else:
            sampler = TrainCategoriesSampler(y_label_train, *sample_info)
        
        train_data = AudioDataset((x_data_train, x_metadata_train, y_label_train),  from_audio=from_audio, prompt=prompt)
        test_data = AudioDataset((x_data_test, x_metadata_test, y_label_test),  from_audio=from_audio, prompt=prompt)
        val_data = AudioDataset((x_data_vad, x_metadata_vad, y_label_vad),  from_audio=from_audio, prompt=prompt)
        
        train_loader = DataLoader(
            train_data, num_workers=2,  batch_sampler=sampler,
        )
        val_loader = DataLoader(
            val_data, num_workers=2, # batch_sampler=sampler,
        )
        test_loader = DataLoader(
            test_data, batch_size=configs.batch_size, shuffle=False, num_workers=2
        )
        return train_loader, val_loader, test_loader

    else:
        train_data = AudioDataset((x_data_train, x_metadata_train, y_label_train),  from_audio=from_audio, prompt=prompt)
        test_data = AudioDataset((x_data_test, x_metadata_test, y_label_test),  from_audio=from_audio, prompt=prompt)
        val_data = AudioDataset((x_data_vad, x_metadata_vad, y_label_vad),  from_audio=from_audio, prompt=prompt)
        train_loader = DataLoader(
            train_data, batch_size=configs.batch_size, num_workers=2, shuffle=True
        )
        val_loader = DataLoader(
            val_data, batch_size=configs.batch_size, num_workers=2, shuffle=True
        )
        test_loader = DataLoader(
            test_data, batch_size=configs.batch_size, shuffle=False, num_workers=2
        )
        return train_loader, val_loader, test_loader
    

def test(model, test_loader, loss_func, n_cls, plot_feature="", plot_only=False, return_auc=False, verbose=True):
    total_loss = []
    test_step_outputs = []
    features = []
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    with torch.no_grad():
        for i,  (x1, x2, x3, y) in enumerate(test_loader):
            x1 = x1.to(device)
            # print(x3)
            y = y.to(device)
            # print(n_cls, y)
            y_hat = model(x1, x2, x3)
            if plot_feature: 
                feature = model(x1, x2, x3, no_fc=True)
                features.append(feature.detach().cpu().numpy())
            if plot_only:
                test_step_outputs.append((y.detach().cpu().numpy(), None, None))
                continue
            loss = loss_func(y_hat, y)
            total_loss.append(loss.item())

            _, predicted = torch.max(y_hat, 1)
            probabilities = F.softmax(y_hat, dim=1)
            test_step_outputs.append((y.detach().cpu().numpy(), predicted.detach().cpu().numpy(), probabilities.detach().cpu().numpy() ))
    
    all_outputs = test_step_outputs
    y = np.concatenate([output[0] for output in all_outputs])
    if plot_feature:
        features = np.concatenate(features, axis=0)
        plot_tsne(features, y, title=plot_feature)

    if plot_only:
        return
    
    total_loss = np.average(total_loss)
    

    predicted = np.concatenate([output[1] for output in all_outputs])
    probs = np.concatenate([output[2] for output in all_outputs])

    # print(y)
    # print(probs[11])

    acc = np.mean(predicted == y)

    auroc = AUROC(task="multiclass", num_classes=n_cls)
    auc = auroc(torch.from_numpy(probs), torch.from_numpy(y))

    if verbose:
        print("loss", total_loss)
        print("acc", acc)
        print("auc", auc)

    if return_auc:
        return acc, auc

    return total_loss / (i+1)

def set_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)



