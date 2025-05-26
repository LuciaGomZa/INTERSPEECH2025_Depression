import time
import os
import random
import numpy as np
import pandas as pd
import warnings

from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, cohen_kappa_score, confusion_matrix, accuracy_score, balanced_accuracy_score,
    roc_auc_score, recall_score, precision_score
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset, BatchSampler

def makedir(directory):
    """Create folder if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def set_seed(seed_value):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    try:
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    except Exception:
        print('Environment without pytorch')

# ===========================
# Prepare data
# ===========================

def load_partitions_deptalk(label_path):
    df = pd.read_csv(label_path)
    df['phq_2'] = df['phq_2'].replace({'Control (<=9)': 0, 'Depression (>9)': 1})
    df['phq_3'] = df['phq_3'].replace({'Minimal (<=4)': 0, 'Moderate (5-14)': 1, 'Severe (>14)': 2})
    df['phq_5'] = df['phq_5'].replace({
        'Minimal (0-4)': 0, 'Mild (5-9)': 1, 'Moderate (10-14)': 2,
        'Moderately severe (15-19)': 3, 'Severe (20-27)': 4
    })
    df.rename(columns={'user': 'User', 'phq_2': 'PHQ_binary', 'outer_fold': 'outer_partition'}, inplace=True)
    df = df.drop_duplicates(subset=['User'], keep='first')
    df.drop(columns=['emotion', 'gender', 'number', 'folder', 'path'], inplace=True)
    df.reset_index(inplace=True, drop=True)
    # Adds five new columns (fold_1 to fold_5), initializing them to -1
    # For each outer fold, it creates five inner folds for CV
    for i in range(1, 6):
        df[f'fold_{i}'] = -1
    for outer in range(1, 6):
        train_idx = df[df['outer_partition'] != outer].index
        X_train = df.loc[train_idx, 'User']
        y_train = df.loc[train_idx, 'PHQ_binary']
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
        for fold, (_, val_idx_fold) in enumerate(skf.split(X_train, y_train)):
            val_users = X_train.iloc[val_idx_fold]
            df.loc[df['User'].isin(val_users), f'fold_{outer}'] = fold + 1
    return df

def load_Xygroup_deptalk(split, prediction, model_speech, model_text, window, partitions, fold, modality, embeddings_path):
    # Choose external partition 1
    if split == 'train':
        df_select = partitions.loc[~partitions['fold_1'].isin([fold, -1])].copy()
    elif split == 'dev':
        df_select = partitions.loc[partitions['fold_1'] == fold].copy()
    elif split == 'test':
        df_select = partitions.loc[partitions['fold_1'] == -1].copy()
    df_select.reset_index(inplace=True, drop=True)

    
    path_emb = os.path.join(embeddings_path, f"W{window['window_size']}H{window['hop_length']}")
    
    X, y, group, files = [], [], [], []
    for user in df_select['User'].values:
        label = df_select.loc[df_select.User == user][prediction].values[0]
        if modality == 'speech':
            path = path_emb + 'speech/mean/' + model_speech.replace('/', '-') + '/' + user + '_' + model_speech.split('/')[1] + '.npy'
            embeddings = np.load(path)
            for n in range(embeddings.shape[0]):
                X.append(embeddings[n])
                y.append(label)
                group.append(user)
                files.append(path + f'_{n}')
        elif modality == 'text':
            path = path_emb + 'text/' + model_text.replace('/', '-') + '/' + user + '_' + model_text.split('/')[1] + '.npy'
            embeddings = np.load(path)
            for n in range(embeddings.shape[0]):
                X.append(embeddings[n])
                y.append(label)
                group.append(user)
                files.append(path + f'_{n}')
        elif modality == 'multimodal':
            path_speech = path_emb + 'speech/mean/' + model_speech.replace('/', '-') + '/' + user + '_' + model_speech.split('/')[1] + '.npy'
            path_text = path_emb + 'text/' + model_text.replace('/', '-') + '/' + user + '_' + model_text.split('/')[1] + '.npy'
            embeddings_speech = np.load(path_speech)
            embeddings_text = np.load(path_text)
            if embeddings_speech.shape[0] != embeddings_text.shape[0]:
                print(f"Shape mismatch for user {user}: speech {embeddings_speech.shape}, text {embeddings_text.shape}")
            for n in range(embeddings_text.shape[0]):
                X.append(np.concatenate((embeddings_speech[n], embeddings_text[n]), axis=-1))
                y.append(label)
                group.append(user)
                files.append(path_text + f'_{n}')
        else:
            raise ValueError("modality must be 'speech', 'text', or 'multimodal'")
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)
    group = np.array(group).reshape(-1, 1)
    files = np.array(files).reshape(-1, 1)
    return X, y, group, files

def prepare_data_deptalk(label, model_speech, model_text, window, fold, modality, embeddings_path, label_path, verbose=False):
    """Load data X, y, group, file data for each data partition."""
    df_partitions = load_partitions_deptalk(label_path)
    partitions = shuffle(df_partitions.copy(), random_state=13)
    partitions.reset_index(inplace=True, drop=True)
    scaler = StandardScaler()
    data_splits = {}
    for split in ['train', 'dev', 'test']:
        X, y, group, files = load_Xygroup_deptalk(split, label, model_speech, model_text, window, partitions, fold, modality, embeddings_path)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        data_splits[split] = (X, y, group, files)
        if verbose:
            print(f'{split.capitalize()} - X shape: {X.shape}, y shape: {y.shape}, group shape: {group.shape}')
    return data_splits

def load_partitions_daic(dataset, label_path):
    """Create data partitions for each dataset."""
    df_labels = pd.read_csv(label_path)
    df_labels.rename(columns={'PHQ-9 Total':'PHQ_score'}, inplace=True)
    common = ['User', 'Mode', 'Gender','PHQ_binary','PHQ_score']

    if dataset in ['DAIC+']:
        df = df_labels.loc[df_labels['DAIC_459'] == 'yes'].loc[:, common + ['459_part_CV']].copy()
        df.rename(columns={'459_part_CV':'partition'}, inplace=True)
    elif dataset == 'WozDAIC':
        df = df_labels.loc[~df_labels['WozDAIC_part_SoA_CV'].isna()].loc[:, common + ['WozDAIC_part_SoA_CV']].copy()
        df.rename(columns={'WozDAIC_part_SoA_CV':'partition'}, inplace=True)
    elif dataset == 'EDAIC':
        df = df_labels.loc[~df_labels['EDAIC_part_CV'].isna()].loc[:, common + ['EDAIC_part_CV']].copy()
        df.rename(columns={'EDAIC_part_CV':'partition'}, inplace=True)
    df.reset_index(inplace=True,drop=True)
    return df

def load_Xygroup_daic(split, prediction, model_speech, model_text, window, partitions, fold, modality, embeddings_path):
    """Load data X, y, group data for the specified modality."""
    if split == 'train':
        df_select = partitions.loc[~partitions.partition.isin([str(fold), 'test'])].copy()
    elif split == 'dev':
        df_select = partitions.loc[partitions.partition == str(fold)].copy()
    elif split == 'test':
        df_select = partitions.loc[partitions.partition == split].copy()
    df_select.reset_index(inplace=True, drop=True)

    path_emb = os.path.join(embeddings_path, f"W{window['window_size']}H{window['hop_length']}")
    X, y, group, files = [], [], [], []
    for user in df_select['User'].values:
        label = df_select.loc[df_select.User == user][prediction].values[0]
        if modality == 'speech':
            path_emb_i = os.path.join(path_emb, 'speech/mean', model_speech.replace('/','-'), f"{user}_P_{model_speech.split('/')[1]}.npy")
            embeddings = np.load(path_emb_i)
        elif modality == 'text':
            path_emb_i = os.path.join(path_emb, 'text', model_text.replace('/','-'), f"{user}_P_{model_text.split('/')[1]}.npy")
            embeddings = np.load(path_emb_i)
        elif modality == 'multimodal':
            path_speech = os.path.join(path_emb, 'speech/mean', model_speech.replace('/','-'), f"{user}_P_{model_speech.split('/')[1]}.npy")
            path_text = os.path.join(path_emb, 'text', model_text.replace('/','-'), f"{user}_P_{model_text.split('/')[1]}.npy")
            path_emb_i = path_speech
            embeddings_speech = np.load(path_speech)
            embeddings_text = np.load(path_text)
        else:
            raise ValueError("Unknown modality")
        for n in range(embeddings.shape[0]):
            if modality == 'multimodal':
                X.append(np.concatenate((embeddings_speech[n],embeddings_text[n])))
            else:
                X.append(embeddings[n])
            y.append(label)
            group.append(user)
            files.append(path_emb_i+'_'+str(n))
    X = np.array(X); y = np.array(y); group = np.array(group); files = np.array(files)
    y = y.reshape(-1, 1)
    group = group.reshape(-1, 1)
    files = files.reshape(-1, 1)
    return X, y, group, files

def prepare_data_daic(dataset, label, model_speech, model_text, window, fold, modality, embeddings_path, label_path, verbose=False):
    """Load data X, y, group, file data for each data partition."""
    df_partitions = load_partitions_daic(dataset, label_path)
    partitions = shuffle(df_partitions.copy(), random_state=13)
    partitions.reset_index(inplace=True, drop=True)
    scaler = StandardScaler()
    data_splits = {}
    for split in ['train', 'dev', 'test']:
        X, y, group, files = load_Xygroup_daic(split, label, model_speech, model_text, window, partitions, fold, modality, embeddings_path)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        data_splits[split] = (X, y, group, files)
        if verbose:
            print(f'{split.capitalize()} - X shape: {X.shape}, y shape: {y.shape}, group shape: {group.shape}')
    return data_splits

# ===========================
# Modeling
# ===========================

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_labels, num_layers=1, bidirectional=False):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, num_labels)
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_hidden_state = gru_out[:, -1, :]
        out = self.fc(last_hidden_state)
        return out

class BalancedBatchSampler(BatchSampler):
    """
    This sampler ensures that each batch contains the same number of samples from each class.
    """
    def __init__(self, class_indices, batch_size_per_class):
        self.class_indices = class_indices
        self.batch_size_per_class = batch_size_per_class
        self.num_classes = len(class_indices)
        self.batch_size = self.num_classes * self.batch_size_per_class
    def __iter__(self):
        """
        Generate batches with balanced classes.
        It calculates how many full batches can be created, based on the minimum number of samples available across classes.
        """
        class_samples = [np.random.permutation(indices) for indices in self.class_indices]
        num_batches = min([len(indices) // self.batch_size_per_class for indices in self.class_indices])
        for i in range(num_batches):
            batch = []
            for class_sample in class_samples:
                start_idx = i * self.batch_size_per_class
                end_idx = start_idx + self.batch_size_per_class
                batch.extend(class_sample[start_idx:end_idx])
            np.random.shuffle(batch) # Shuffle batch for randomness
            yield batch
    def __len__(self):
        # Returns the total number of batches
        return min([len(indices) // self.batch_size_per_class for indices in self.class_indices])

def create_balanced_dataloader(X_train, y_train, batch_size_per_class):
    """
    Creates a DataLoader that ensures balanced batches (equal samples from each class).
    """
    unique_classes = torch.unique(y_train)
    class_indices = []
    # Collect the indices for each class
    for cls in unique_classes:
        class_idx = torch.where(y_train == cls)[0]
        class_indices.append(class_idx)
    # Create a custom balanced batch sampler
    sampler = BalancedBatchSampler(class_indices, batch_size_per_class)
    dataset = TensorDataset(X_train, y_train)
    my_data_loader = DataLoader(dataset, batch_sampler=sampler)
    return my_data_loader

# ===========================
# Training and evaluation
# ===========================

def train_model(model_params, data_splits, device, label):
    input_size = data_splits['test'][0].shape[2]
    num_labels = data_splits['test'][1].shape[1]
    if label == 'PHQ_binary':
        loss_fn = nn.BCEWithLogitsLoss() if num_labels == 1 else nn.CrossEntropyLoss()
        train_hist = {'epoch': [], 'loss': [], 'WAcc': [], 'UAcc': [], 'kappa': [], 'auc': [], 'f1_0': [],'f1_1': [],'f1_avg': [], 'precision': [], 'recall': [], 'cm': []}
        val_hist =   {'epoch': [], 'loss': [], 'WAcc': [], 'UAcc': [], 'kappa': [], 'auc': [], 'f1_0': [],'f1_1': [],'f1_avg': [], 'precision': [], 'recall': [], 'cm': []}
        test_hist =  {'loss': [], 'WAcc': [], 'UAcc': [], 'kappa': [], 'auc': [], 'f1_0': [],'f1_1': [],'f1_avg': [], 'precision': [], 'recall': [], 'cm': []}

    bidirectional = model_params['bidirectional']
    n_epochs = model_params['n_epochs_train']
    patience = model_params['patience_train']
    lr = model_params['lr']
    hidden_size = model_params['hidden_size']
    GRU_layers = model_params['GRU_layers']
    batch_size = model_params['batch_size']
    path_save = model_params['path_save']

    X_train = data_splits['train'][0]; X_dev = data_splits['dev'][0]; X_test = data_splits['test'][0]
    y_train = data_splits['train'][1]; y_dev = data_splits['dev'][1]; y_test = data_splits['test'][1]
    X_dev = X_dev.to(device); X_test = X_test.to(device)
    y_dev = y_dev.to(device); y_test = y_test.to(device)
    users_train = data_splits['train'][2]; users_dev = data_splits['dev'][2]; users_test = data_splits['test'][2]
    files_train = data_splits['train'][3]; files_dev = data_splits['dev'][3]; files_test = data_splits['test'][3]

    model = GRUModel(input_size, hidden_size, num_labels, GRU_layers, bidirectional)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=10)

    # TRAINING ==========================================================
    early_stop_counter = 0; best_loss = np.inf
    for epoch in range(n_epochs):
        train_loss = 0.0
        model.train()
        optimizer.zero_grad()
        if label == 'PHQ_binary':
            batch_size_per_class = int(batch_size // 2)
            balanced_loader = create_balanced_dataloader(X_train, y_train, batch_size_per_class)
        train_y_true = torch.tensor([], dtype=torch.float)
        train_y_pred = torch.tensor([], dtype=torch.float)
        for X_batch, y_batch in balanced_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            train_y_true = torch.cat((train_y_true, y_batch.cpu()), dim=0)
            train_y_pred = torch.cat((train_y_pred, y_pred.cpu()), dim=0)
        train_loss /= (len(X_train) // batch_size)
        train_metrics = calculate_metrics(train_y_true, train_y_pred, num_labels, label)
        for key, value in zip(train_hist.keys(), [epoch] + [train_loss] + list(train_metrics)):
            train_hist[key].append(value)

        # VALIDATION ==========================================================
        model.eval()
        with torch.no_grad():
            y_pred = model(X_dev)
            val_loss = loss_fn(y_pred, y_dev).item()
            val_metrics = calculate_metrics(y_dev, y_pred, num_labels, label)
            for key, value in zip(val_hist.keys(), [epoch] + [val_loss] + list(val_metrics)):
                val_hist[key].append(value)
        lr_scheduler.step(val_loss)
        if val_loss < best_loss:
            early_stop_counter = 0
            best_loss = val_loss
            best_epoch = epoch
            y_pred_save = y_pred
            torch.save({'epoch': best_epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss}, path_save + 'best_model.pth')
            # torch.save(model, path_save + 'best_model.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                break
    print(' ===> best_epoch', best_epoch, 'best_loss', best_loss)

    # Save metrics to CSV 
    df_train = pd.DataFrame(train_hist)
    df_train.columns = ['train_' + col for col in df_train.columns]
    df_train.to_csv(path_save + 'train_metrics.csv', index=False)
    df_val = pd.DataFrame(val_hist)
    df_val.columns = ['val_' + col for col in df_val.columns]
    df_val.to_csv(path_save + 'val_metrics.csv', index=False)
    # Save the validation predictions of best model
    df_predictions = save_predictions(y_pred_save, y_dev, users_dev, files_dev, num_labels, path_save, 'val', label)
    df_val_spk = aggregate_predictions_spk(label, 'val', df_predictions, path_save)


    # TESTING ==========================================================
    model = GRUModel(input_size, hidden_size, num_labels, GRU_layers, bidirectional)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    checkpoint = torch.load(path_save + 'best_model.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # model = torch.load(path_save + 'best_model.pth')
    model = model.to(device)
    model.eval()

    X_test_loader = torch.utils.data.DataLoader(X_test, batch_size=4)
    y_pred = []
    with torch.no_grad():
        for X_batch in X_test_loader:
            y_batch_pred = model(X_batch)
            y_pred.append(y_batch_pred)
        y_pred = torch.cat(y_pred, dim=0)
        test_loss = loss_fn(y_pred, y_test).item()
        test_metrics = calculate_metrics(y_test, y_pred, num_labels, label)
        for key, value in zip(test_hist.keys(), [test_loss] + list(test_metrics)):
            test_hist[key].append(value)
    # Save metrics to CSV
    df_test = pd.DataFrame(test_hist)
    df_test.columns = ['test_' + col for col in df_test.columns]
    df_test.to_csv(path_save + 'test_metrics.csv', index=False)
    # Save the test predictions
    df_predictions = save_predictions(y_pred, y_test, users_test, files_test, num_labels, path_save, 'test', label)
    df_test_spk = aggregate_predictions_spk(label, 'test', df_predictions, path_save)

    # Create dataframe with final best values
    df_val_best = pd.DataFrame(df_val.loc[best_epoch,:]).T
    df_val_best.reset_index(inplace=True, drop=True)
    results = pd.concat([df_val_best, df_val_spk, df_test, df_test_spk], axis=1)
    return results

def calculate_metrics(y_true, y_pred, num_labels, label):
    if label == 'PHQ_binary':
        # Move tensors to CPU for compatibility with sklearn functions
        y_true = y_true.cpu().detach().numpy()
        if num_labels == 1:
            # Apply the sigmoid function to the logits to get probabilities
            # Convert probabilities to binary predictions using a threshold of 0.5
            sigmoid_function = nn.Sigmoid()
            y_pred_prob = sigmoid_function(y_pred).cpu().detach().numpy()
            y_pred_binary = (y_pred_prob > 0.5).astype(int)
            # Calculate metrics
            WAcc = accuracy_score(y_true, y_pred_binary)
            UAcc = balanced_accuracy_score(y_true, y_pred_binary)
            kappa = cohen_kappa_score(y_true, y_pred_binary)
            auc = roc_auc_score(y_true, y_pred_prob)
            f1_0, f1_1 = f1_score(y_true, y_pred_binary, average=None)
            f1_avg = f1_score(y_true, y_pred_binary, average='macro')
            precision = precision_score(y_true, y_pred_binary, zero_division=0)
            recall = recall_score(y_true, y_pred_binary, zero_division=0)
            cm = confusion_matrix(y_true, y_pred_binary)
        else:
            # (Assuming one-hot encoding of labels)
            # Apply the Softmax function to the logits to get probabilities
            softmax_function = nn.Softmax(dim=1)
            y_pred_prob = softmax_function(y_pred).cpu().detach().numpy()
            y_pred_classes = np.argmax(y_pred_prob, axis=1)
            y_true_classes = np.argmax(y_true, axis=1)
            # Calculate metrics
            WAcc = accuracy_score(y_true_classes, y_pred_classes)
            UAcc = balanced_accuracy_score(y_true_classes, y_pred_classes)
            kappa = cohen_kappa_score(y_true_classes, y_pred_classes)
            cm = confusion_matrix(y_true_classes, y_pred_classes)
            if num_labels == 2:
                auc = roc_auc_score(y_true, y_pred_prob[:, 1])
                f1_avg = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
                precision = precision_score(y_true_classes, y_pred_classes, zero_division=0)
                recall = recall_score(y_true_classes, y_pred_classes, zero_division=0)
            else:
                auc = roc_auc_score(y_true, y_pred_prob, multi_class='ovr')
                f1_avg = f1_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
                precision = precision_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
                recall = recall_score(y_true_classes, y_pred_classes, average='weighted', zero_division=0)
            f1_0 = np.nan; f1_1 = np.nan
        metrics = [WAcc, UAcc, kappa, auc, f1_0, f1_1, f1_avg, precision, recall, cm]
    return metrics

def calculate_metrics_simple(y_true, y_pred, label):
    if label == 'PHQ_binary':
        WAcc = accuracy_score(y_true, y_pred)
        UAcc = balanced_accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        f1_0, f1_1 = f1_score(y_true, y_pred, average=None)
        f1_avg = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        metrics = [WAcc, UAcc, kappa, f1_0, f1_1, f1_avg, precision, recall, cm]
    return metrics

def save_predictions(y_pred, y_true, users, files, num_labels, path_save, split, label):
    if label == 'PHQ_binary':
        if num_labels == 1:
            sigmoid_function = nn.Sigmoid()
            y_pred_prob = sigmoid_function(y_pred)
            y_pred_binary = (y_pred_prob > 0.5).int()
            df_predictions = pd.DataFrame({
                'file': files.reshape(1, -1)[0],
                'subject': users.reshape(1, -1)[0],
                'y_true': y_true.int().cpu().numpy().flatten(),
                'y_pred': y_pred_binary.cpu().numpy().flatten(),
                'y_pred_prob': y_pred_prob.cpu().numpy().flatten()
            })
        else:
            softmax_function = nn.Softmax(dim=1)
            y_pred_prob = softmax_function(y_pred)
            df_predictions = pd.DataFrame({
                'file': files.reshape(1, -1)[0],
                'subject': users.reshape(1, -1)[0],
                'y_true': torch.argmax(y_true.cpu(), 1).numpy(),
                'y_pred': torch.argmax(y_pred.cpu(), 1).numpy(),
                'y_pred_prob': [str(i) for i in y_pred_prob.cpu().numpy()]
            })
    df_predictions.to_csv(path_save + split + '_predictions.csv', index=False)
    return df_predictions

def aggregate_predictions_spk(label, partition, df_predictions, path_save):
    if label == 'PHQ_binary':
        def aggregate_with_fraction_threshold(x, threshold):
            fraction_of_ones = (x == 1).sum() / len(x)
            return 1 if fraction_of_ones >= threshold else 0
        hist_spk = {}
        threshold = 0.5
        df_majVot = df_predictions.groupby('subject').agg({
            'y_true': 'first',
            'y_pred': lambda x: aggregate_with_fraction_threshold(x, threshold)
        }).reset_index()
        metrics_spk = calculate_metrics_simple(df_majVot['y_true'], df_majVot['y_pred'], label)
        suffix = f"_{int(threshold * 100)}"
        for key, value in zip(['WAcc', 'UAcc', 'kappa', 'f1_0', 'f1_1', 'f1_avg', 'precision', 'recall', 'cm'], metrics_spk):
            hist_spk[f"{key}{suffix}"] = [value]
        df_spk = pd.DataFrame(hist_spk)
        df_spk.columns = [partition + '_spk_' + col for col in df_spk.columns]
        df_spk.to_csv(path_save + partition + '_spk_metrics.csv', index=False)
    return df_spk

# ===========================
# Run experiments
# ===========================

def run_experiment(dataset, label, modality, window, options_speech, options_text, options_params, model_params, paths):
    """
    Run the training and evaluation of the model.
    """
    final_results = pd.DataFrame()
    for model_speech in options_speech: 
        for model_text in options_text: 
            for fold in range(5):
                print('(o)',model_speech, model_text,'fold',fold)
                if dataset == 'DEPTALK':
                    data_splits = prepare_data_deptalk(label, model_speech, model_text, window, fold, 
                                                       modality, paths['embeddings_path_deptalk'], paths['label_path_deptalk'])
                else:
                    data_splits = prepare_data_daic(dataset, label, model_speech, model_text, window, fold, 
                                            modality, paths['embeddings_path_daic'], paths['label_path_daic']) 
                for bidirectional in options_params['bidirectional']:
                    for batch_size in options_params['batch_size']: 
                        for hidden_size in options_params['hidden_size']:
                            for GRU_layers in options_params['GRU_layers']: 
                                    lr = model_params['lr']
                                    if modality == 'multimodal':
                                        path_save = paths['path_folder_results']+dataset+'/'+label+'/'+ model_speech.replace("/","-") + '__' + model_text.replace("/","-")+'_W'+str(window['window_size'])+'H'+str(window['hop_length'])+'_bi'+str(bidirectional)+'_GRU'+str(GRU_layers)+'_hidden'+str(hidden_size)+'_lr'+str(lr)+'_batch'+str(batch_size)+'/fold'+str(fold)+'/'
                                    elif modality == 'speech':
                                        model_text = ''
                                        path_save = paths['path_folder_results']+dataset+'/'+label+'/'+ model_speech.replace("/","-") +'_W'+str(window['window_size'])+'H'+str(window['hop_length'])+'_bi'+str(bidirectional)+'_GRU'+str(GRU_layers)+'_hidden'+str(hidden_size)+'_lr'+str(lr)+'_batch'+str(batch_size)+'/fold'+str(fold)+'/'
                                    elif modality == 'text':
                                        model_speech = ''
                                        path_save = paths['path_folder_results']+dataset+'/'+label+'/'+ model_text.replace("/","-") +'_W'+str(window['window_size'])+'H'+str(window['hop_length'])+'_bi'+str(bidirectional)+'_GRU'+str(GRU_layers)+'_hidden'+str(hidden_size)+'_lr'+str(lr)+'_batch'+str(batch_size)+'/fold'+str(fold)+'/'
                                    if os.path.isdir(path_save):
                                        continue
                                    else:
                                        print('====> ',dataset,'- bidirectional:',bidirectional,'- batch:',batch_size,'- hidden:',hidden_size,'- GRUlayer',GRU_layers,'- lr',lr,'-',)
                                    makedir(path_save)
                                    model_params['path_save'] = path_save ; model_params['batch_size'] = batch_size
                                    model_params['GRU_layers'] = GRU_layers  ; model_params['hidden_size'] = hidden_size
                                    model_params['bidirectional'] = bidirectional
                                    
                                    # Training models
                                    try:
                                        df_results_i = train_model(model_params, data_splits, device, label)
                                        df_results_i['dataset'] = dataset; 
                                        df_results_i['model_speech'] = model_speech; df_results_i['model_text'] = model_text
                                        df_results_i['context_window'] = 'W'+str(window['window_size'])+'H'+str(window['hop_length'])
                                        df_results_i['model'] = 'biGRU' if  bidirectional else 'GRU'
                                        df_results_i['GRU_layers'] = GRU_layers; df_results_i['lr'] = lr; 
                                        df_results_i['batch_size'] = batch_size; df_results_i['hidden_size'] = hidden_size; 
                                        df_results_i['patience_train'] = model_params['patience_train']; df_results_i['n_epochs_train'] = model_params['n_epochs_train']
                                        df_results_i['fold'] = fold; 
                                        final_results = pd.concat([final_results, df_results_i], ignore_index=True)
                                        final_results.to_csv(paths['path_folder_results'], index=False)
                                    except:
                                        print('Error in training model:', model_speech, model_text, 'fold', fold)
                                        continue
    return final_results


# -----------------------------------------------------
# Example Usage
# -----------------------------------------------------
if __name__ == "__main__":

    # Parameters:
    modality = 'multimodal'  # 'speech', 'text', or 'multimodal'
    dataset = 'DAIC+' # 'WozDAIC', 'EDAIC', 'DAIC+', or 'DEPTALK'
    options_speech = {"facebook/hubert-base-ls960", "facebook/hubert-base-ls960_L9"}
    options_text = {"j-hartmann/emotion-english-distilroberta-base", "distilbert/distilroberta-base"}
    options_params = {'bidirectional': [True, False],
                        'batch_size': [64, 128, 256],
                        'hidden_size': [32, 64, 128],
                        'GRU_layers': [1, 3]}
    # Fixed parameters:
    label = 'PHQ_binary'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_seed(13)
    window = {'window_size': 5, 'hop_length': 2} if dataset == 'DEPTALK' else {'window_size': 20, 'hop_length': 10}
    model_params = {'patience_train': 30, 
                    'n_epochs_train': 500,
                    'lr': 0.0001} 
    paths = {   'embeddings_path_daic': 'D:/lugoza/Databases/DAIC/2-embeddings_embeddings_windows/',
                'embeddings_path_deptalk': 'D:/lugoza/Databases/REMDE/embeddings_windows/',
                'label_path_daic': 'data/0-labels/df_labels_new_CV.csv',
                'label_path_deptalk': '../REMDE/data/data_partitions.csv',
                'path_folder_results': 'results/4-utterances_v2_CV/' + modality + '/',
                'path_file_results': 'results/4-utterances_v2_CV/' + modality + '/' + dataset + '/' + 'results_' + dataset + '_' + label + '.csv'
                }
    
    df_results = run_experiment(dataset, label, modality, window, options_speech, options_text, options_params, model_params, paths)
    
                                    
