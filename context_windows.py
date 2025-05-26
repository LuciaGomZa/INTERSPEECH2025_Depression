import numpy as np
import logging
from pathlib import Path
from typing import Union

def create_context_windows(
    data: np.ndarray, 
    window_size: int = 20, 
    hop_length: int = 10
) -> np.ndarray:
    """
    Slices data into overlapping context windows.
    """
    num_frames = data.shape[0]
    context_windows = []

    # Generate the context windows, ignoring frames at the end of the array
    for start_idx in range(0, num_frames - window_size + 1, hop_length):
        window = data[start_idx:start_idx + window_size]
        context_windows.append(window)
    
    # Convert the list of windows into a numpy array
    context_windows = np.array(context_windows)
    
    return context_windows

def extract_context_windows_deptalk(
    path_data: Union[str, Path],
    windows: list,
    models: list,
    modality: str,
    save_root: Union[str, Path],
):
    """
    Extracts context windows from embeddings and saves them per user.
    """
    path_data = Path(path_data)
    save_root = Path(save_root)
    for window in windows:
        path_save = save_root / f"W{window['window_size']}H{window['hop_length']}"
        for model_name in models:
            model_folder = model_name.replace('/', '-')
            if modality == 'text':
                path_modality = path_data / modality / model_folder
                save_dir = path_save / modality / model_folder
            else:
                path_modality = path_data / modality / 'mean' / model_folder
                save_dir = path_save / modality / 'mean' / model_folder
            save_dir.mkdir(parents=True, exist_ok=True)
            files = list(path_modality.glob("*.npy"))
            users = ["_".join(f.stem.split('_')[0:3]) for f in files]
            for subject in set(users):
                user_data = []
                for file in [f for f, u in zip(files, users) if u == subject]:
                    try:
                        data = np.load(str(file))
                        if data.shape[0] >= window['window_size']:
                            emb = create_context_windows(data, window['window_size'], window['hop_length'])
                            user_data.append(emb)
                        else:
                            logging.warning(f"{file}: {data.shape[0]}")
                    except Exception as e:
                        logging.error(f"Error loading {file}: {e}")
                if user_data:
                    user_data_concat = np.concatenate(user_data, axis=0)
                    out_path = save_dir / f"{subject}_{model_name.split('/')[1]}.npy"
                    np.save(out_path, user_data_concat)

def extract_context_windows_daic(
    path_data: Union[str, Path],
    windows: list,
    models: list,
    modality: str,
    save_root: Union[str, Path],
):
    """
    Extracts context windows from DAIC speech embeddings and saves them per user/file.
    """
    path_data = Path(path_data)
    save_root = Path(save_root)
    for window in windows:
        path_save = save_root / f"W{window['window_size']}H{window['hop_length']}"
        for model_name in models:
            model_folder = model_name.replace('/', '-')
            if modality == 'text':
                path_modality = path_data / modality / model_folder
                save_dir = path_save / modality / model_folder
            else:
                path_modality = path_data / modality / 'mean' / model_folder
                save_dir = path_save / modality / 'mean' / model_folder
            save_dir.mkdir(parents=True, exist_ok=True)
            for file in path_modality.glob("*.npy"):
                try:
                    data = np.load(str(file))
                    emb = create_context_windows(data, window['window_size'], window['hop_length'])
                    out_path = save_dir / file.name
                    np.save(out_path, emb)
                except Exception as e:
                    logging.error(f"Error processing {file}: {e}")

# =========
# DEPTALK
# =========

# extract_context_windows_deptalk(
#     path_data='D:/lugoza/Databases/REMDE/embeddings_utterances/',
#     windows=[{'window_size':5, 'hop_length':2}],
#     models=["facebook/hubert-base-ls960"],
#     modality='speech',
#     save_root='D:/lugoza/Databases/REMDE/embeddings_windows/'
# )

# extract_context_windows_deptalk(
#     path_data='D:/lugoza/Databases/REMDE/embeddings_utterances/',
#     windows=[{'window_size':5, 'hop_length':2}],
#     models=["distilbert/distilroberta-base"],
#     modality='text',
#     save_root='D:/lugoza/Databases/REMDE/embeddings_windows/'
# )

# =========
# DAIC
# =========

# extract_context_windows_daic(
#     path_data='D:/lugoza/Databases/DAIC/2-embeddings_embeddings_utterances/',
#     windows=[{'window_size':20, 'hop_length':10}],
#     models=["facebook/hubert-base-ls960"],
#     modality='speech',
#     save_root='D:/lugoza/Databases/DAIC/2-embeddings_embeddings_windows/'
# )
    
# extract_context_windows_daic(
#     path_data='D:/lugoza/Databases/DAIC/2-embeddings_embeddings_utterances/',
#     windows=[{'window_size':20, 'hop_length':10}],
#     models=["distilbert/distilroberta-base"],
#     modality='text',
#     save_root='D:/lugoza/Databases/DAIC/2-embeddings_embeddings_windows/'
# )
