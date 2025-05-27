import os
import re
import logging
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import librosa
import torch
from transformers import AutoFeatureExtractor, AutoModel, AutoTokenizer

# For delays information in DAIC, see: 
# M. Hu, L. Liu, X. Wang, Y. Tang, J. Yang, and N. An, “Parallel multiscale bridge fusion network for audio–visual automatic depression assessment,” IEEE Transactions on Computational Social Systems, vol. 11, no. 5, pp. 6830–6842, 2024
# DEFAULT_DELAYS_DAIC = {'318': 34.3199, '321': 3.8379, '341': 6.1892, '362': 16.8582}
DEFAULT_DELAYS_DAIC = {'318': 33.8199, '321': 3.3379, '341': 5.6892, '362': 16.3582, '300': 35.7}

def get_speech_embedding(
    audio: np.ndarray,
    feature_extractor,
    model,
    model_name: str,
    sr: int,
    device: torch.device, 
) -> np.ndarray:
    """
    Extracts the mean embedding from an audio array, using the specified layer if model_name ends with _L<X>.
    """
    match = re.search(r'_L(\d+)$', model_name)
    with torch.no_grad():
        inputs = feature_extractor(audio, return_tensors="pt", sampling_rate=sr).to(device)
        if match:
            layer_idx = int(match.group(1))
            outputs = model(inputs.input_values, output_hidden_states=True)
            x = outputs.hidden_states[layer_idx].detach().cpu().numpy()
        else:
            outputs = model(inputs.input_values)
            x = outputs.last_hidden_state.detach().cpu().numpy()
    emb_reshape = x.reshape(x.shape[1], x.shape[2])
    emb_mean = np.mean(emb_reshape, axis=0)
    return emb_mean


def extract_speech_embeddings_deptalk(
    input_folder: str,
    output_folder: str,
    model_name: str,
    df_clean: pd.DataFrame,
    df_times: pd.DataFrame,
    df_times_errors: pd.DataFrame,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    sr: int = 16000,
) -> None:
    """
    Extracts mean embeddings from each audio file in a folder using a HuggingFace model.
    Uses get_speech_embedding for each utterance segment.
    """
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_folder)

    # Check for layer specification in model_name (e.g., ..._L9)
    match = re.search(r'_L(\d+)$', model_name)
    if match:
        base_model_name = model_name.rsplit('_L', 1)[0]
    else:
        base_model_name = model_name

    feature_extractor = AutoFeatureExtractor.from_pretrained(base_model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(base_model_name, cache_dir=cache_dir).to(device)

    for user_folder in input_path.iterdir():
        if not user_folder.is_dir():
            continue
        user = "_".join(user_folder.name.split('_')[0:-1]).upper()
        for avatar_folder in user_folder.iterdir():
            if not avatar_folder.is_dir():
                continue
            number_avatar = "_".join(avatar_folder.name.split('_')[-2:])
            if df_clean.loc[(df_clean.user == user) & (df_clean.folder == number_avatar)].shape[0] != 1:
                continue
            embeddings_speech = []
            # 1) Create dataframe for order of audios
            rows_order = []
            audios_dir = avatar_folder / 'Audios'
            for audio_file in audios_dir.iterdir():
                if not audio_file.is_file():
                    continue
                number = int(audio_file.stem.split('=')[1])
                path_audio = str(audio_file)
                if path_audio in df_times_errors.loc[df_times_errors['last'] == 'yes']['path'].values:
                    continue
                # If silence in the middle, use full duration
                if path_audio in list(df_times_errors.loc[df_times_errors['last'] == 'no']['path'].values):
                    input_audio, sample_rate = librosa.load(path_audio)
                    duration = librosa.get_duration(y=input_audio, sr=sample_rate)
                    rows_order.append({'path': path_audio, 'number': number, 'duration': duration})
                else:
                    max_duration = df_times.loc[df_times.path == path_audio]['end_vad'].values[0]
                    rows_order.append({'path': path_audio, 'number': number, 'duration': max_duration})
            df_order = pd.DataFrame(rows_order)
            df_order = df_order.sort_values(by='number', ascending=True)
            df_order.reset_index(inplace=True, drop=True)

            # 2) Load each audio and calculate mean embeddings using get_speech_embedding
            for i in range(df_order.shape[0]):
                audio, _ = librosa.load(df_order.loc[i, 'path'], duration=df_order.loc[i, 'duration'] + 0.1, sr=sr)
                emb_mean = get_speech_embedding(audio=audio, 
                                                feature_extractor=feature_extractor,model=model,
                                                model_name=model_name,sr=sr,device=device)
                embeddings_speech.append(emb_mean)
            # 3) Save embeddings per utterance (one file per avatar)
            out_file = output_path / f"{user}_{avatar_folder.name}_{model_name.replace('/','-')}.npy"
            np.save(out_file, embeddings_speech)
            logging.info(f"Saved embedding for {avatar_folder.name} as {out_file}")


def extract_speech_embeddings_daic(
    users: List[str],
    transcriptions_folder: str,
    audios_folder: str,
    output_folder: str,
    model_name: str,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
    sr: int = 16000,
    delays: Optional[Dict[str, float]] = None,
) -> None:
    """
    Extracts mean embeddings from each utterance in DAIC sessions using a HuggingFace model.
    Uses get_speech_embedding for each utterance segment.
    """
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # Model loading
    match = re.search(r'_L(\d+)$', model_name)
    if match:
        base_model_name = model_name.rsplit('_L', 1)[0]
    else:
        base_model_name = model_name
    feature_extractor = AutoFeatureExtractor.from_pretrained(base_model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(base_model_name, cache_dir=cache_dir).to(device)

    if delays is None:
        delays = DEFAULT_DELAYS_DAIC

    for user in users:
        asr = False
        # Read file with transcriptions and audio (of full session)
        try:
            df = pd.read_csv(os.path.join(transcriptions_folder, f'{user}_P.tsv'), sep='\t')
        except Exception:
            df = pd.read_csv(os.path.join(transcriptions_folder, f'{user}_G_ASR.csv'))
            asr = True
        audio, _ = librosa.load(os.path.join(audios_folder, f'{user}_AUDIO.wav'), sr=sr)

        # Handle delays
        delay = delays.get(str(user), 0)
        if asr:
            start_name, end_name, text_name = 'Start_Time', 'End_Time', 'Text'
        else:
            start_name, end_name, text_name = 'P_Start_Time', 'P_Stop_Time', 'Participant_Resonse'
        df[start_name] += delay
        df[end_name] += delay

        end_time_ans = 0
        last_row = df.shape[0]
        embeddings_speech = []
        for row in range(df.shape[0]):
            # Obtain audio segments from original audio file
            if end_time_ans > df.at[row, start_name]:
                if row != last_row - 1:
                    logging.warning(f"{user} {row} {end_time_ans} {df.loc[row, start_name]}")
                continue
            start_time = df.at[row, start_name]
            start_sample = int(start_time * sr)
            end_time = df.at[row, end_name] + 0.5  # Add 0.5s to avoid cutting speech
            end_sample = int(end_time * sr)
            end_time_ans = df.at[row, end_name]
            utterance_speech = audio[start_sample:end_sample]
            emb_mean = get_speech_embedding(
                audio=utterance_speech,
                feature_extractor=feature_extractor,
                model=model,
                model_name=model_name,
                sr=sr,
                device=device
            )
            embeddings_speech.append(emb_mean)
        # Save embeddings
        out_file = output_path / f"{user}_P_{model_name.split('/')[1]}.npy"
        np.save(out_file, embeddings_speech)
        logging.info(f"Saved embedding for {user} as {out_file}")


def get_text_embedding(
    text: str,
    tokenizer,
    model,
    device: torch.device
) -> np.ndarray:
    """
    Extracts the [CLS] token embedding from a text string.
    """
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(inputs.input_ids)
    x = outputs.last_hidden_state.detach().cpu().numpy()
    emb_cls = x[:, 0, :]  # [CLS] token
    return emb_cls.reshape(emb_cls.shape[1])


def extract_text_embeddings_deptalk(
    input_folder: str,
    output_folder: str,
    model_name: str,
    df_clean: pd.DataFrame,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> None:
    """
    Extracts text embeddings from each utterance in conversation CSVs using a HuggingFace model.
    Uses get_text_embedding for each utterance text.
    """
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    input_path = Path(input_folder)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)

    for user_folder in input_path.iterdir():
        if not user_folder.is_dir():
            continue
        user = "_".join(user_folder.name.split('_')[0:-1])
        for avatar_folder in user_folder.iterdir():
            if not avatar_folder.is_dir():
                continue
            number_avatar = "_".join(avatar_folder.name.split('_')[-2:])
            if df_clean.loc[(df_clean.user == user.upper()) & (df_clean.folder == number_avatar)].shape[0] != 1:
                continue
            csv_file = avatar_folder / f'Conv_{number_avatar}.csv'
            if not csv_file.exists():
                continue
            df_text = pd.read_csv(csv_file, sep=';')
            sentences = df_text.loc[df_text['Source'] == 'Person']['SpanishMessage'].values
            embeddings_text = []
            for text in sentences:
                emb_text = get_text_embedding(
                    text=text,
                    tokenizer=tokenizer,
                    model=model,
                    device=device
                )
                embeddings_text.append(emb_text)
            out_file = output_path / f"{avatar_folder.name}_{model_name.split('/')[1]}.npy"
            np.save(out_file, embeddings_text)
            logging.info(f"Saved text embeddings for {avatar_folder.name} as {out_file}")


def extract_text_embeddings_daic(
    users: List[str],
    transcriptions_folder: str,
    output_folder: str,
    model_name: str,
    device: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> None:
    """
    Extracts text embeddings from each utterance in DAIC sessions using a HuggingFace model.
    Uses get_text_embedding for each utterance text.
    """
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir).to(device)

    delays = DEFAULT_DELAYS_DAIC

    for user in users:
        asr = False
        # Read file with transcriptions
        try:
            df = pd.read_csv(os.path.join(transcriptions_folder, f'{user}_P.tsv'), sep='\t')
        except Exception:
            df = pd.read_csv(os.path.join(transcriptions_folder, f'{user}_G_ASR.csv'))
            asr = True

        delay = delays.get(str(user), 0)
        if asr:
            start_name, end_name, text_name = 'Start_Time', 'End_Time', 'Text'
        else:
            start_name, end_name, text_name = 'P_Start_Time', 'P_Stop_Time', 'Participant_Resonse'
        df[start_name] += delay
        df[end_name] += delay

        embeddings_text = []
        for _, row in df.iterrows():
            start_time = row[start_name]
            # Obtain and clean transcription
            text = str(row[text_name]).lstrip()
            cleaned_text = ' '.join(text.split())
            utterance_text = cleaned_text[:1].upper() + cleaned_text[1:] + '.'

            # Use get_text_embedding for each utterance
            emb_text = get_text_embedding(text=utterance_text,
                tokenizer=tokenizer, model=model, device=device)
            embeddings_text.append(emb_text)

        # Save embeddings
        out_dir = output_path / 'text' / model_name.replace('/', '-')
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{user}_P_{model_name.split('/')[1]}.npy"
        np.save(out_file, embeddings_text)
        logging.info(f"Saved text embeddings for {user} as {out_file}")