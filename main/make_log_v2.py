import os
import glob
import argparse
import random

import librosa
import numpy as np
import numpy.typing as npt


parser = argparse.ArgumentParser()
parser.add_argument('--folder_dataset', type=str, required=True, default="/home/SpeechSeparation/dataset")
parser.add_argument('--log_path_train', type=str, required=True, default="/home/SpeechSeparation/dataset/mix_2_spk_tr.txt")
parser.add_argument('--log_path_valid', type=str, required=True, default="/home/SpeechSeparation/dataset/mix_2_spk_vd.txt")
parser.add_argument('--max_mixture_audio_train', type=int, required=True, default=20000)
parser.add_argument('--max_mixture_audio_valid', type=int, required=True, default=600)


def load_wav(
        path: str
    ) -> npt.ArrayLike:

    signal, sr = librosa.load(path)
    return signal


def write_txt( data: list, path: str) -> None:
    with open(path, mode="w", encoding="utf8") as fp:
        fp.writelines(data)


def signal_to_noise(
            wav_array: npt.ArrayLike,
            axis=0, 
            ddof=0
        ) -> npt.ArrayLike:

    wav_array = np.asanyarray(wav_array)
    mean = wav_array.mean(axis)
    std = wav_array.std(axis = axis, ddof = ddof)

    return np.where(std == 0, 0, mean / std)


if __name__ == "__main__":
    args_input = parser.parse_args()
    folder_dataset = args_input.folder_dataset
    max_mixture_audio_train = args_input.max_mixture_audio_train
    max_mixture_audio_valid = args_input.max_mixture_audio_valid

    type_folder = ['tr', 'vd']
    
    for type_ in type_folder:
        if type_ == "tr":
            log_path = args_input.log_path_train
            max_mixture_audio = max_mixture_audio_train
        else:
            log_path = args_input.log_path_valid
            max_mixture_audio = max_mixture_audio_valid

        data_path = os.path.join(folder_dataset, "data_{}".format(type_))
        spk_folders = os.listdir(data_path)
        num_spk = len(spk_folders)
        output_data = list()

        max_pair_per_spk = int(max_mixture_audio/num_spk)

        for spk_name in spk_folders:
            spk1_file_paths = glob.glob(os.path.join(data_path, spk_name, "*.wav"))
            wav_spk1_path = random.sample(spk1_file_paths, k= 1)[0]
            s1_wav = load_wav(wav_spk1_path)
            s1_snr = signal_to_noise(s1_wav)

            other_spk_folders = random.sample(spk_folders, k= max_pair_per_spk)

            for other_name in other_spk_folders:
                if other_name != spk_name:
                    spk2_file_paths = glob.glob(os.path.join(data_path, other_name, "*.wav"))
                    wav_spk2_path = random.sample(spk2_file_paths, k= 1)[0]
                    s2_wav = load_wav(wav_spk2_path)
                    s2_snr = signal_to_noise(s2_wav)

                    output_line = "{} {} {} {}\n".format(wav_spk1_path, s1_snr, wav_spk2_path, s2_snr)
                    print(output_line)
                    output_data.append(output_line)

        write_txt(output_data, log_path)

