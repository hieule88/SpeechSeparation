import os
import numpy as np
from tqdm import tqdm
from speechbrain.dataio.dataio import read_audio, write_audio
from scipy.io import wavfile
from scipy import signal
import pickle
import csv
import torchaudio

def prepare_wsjmix(datapath, savepath, n_spks=2, skip_prep=False):
    """
    Prepared wsj2mix if n_spks=2 and wsj3mix if n_spks=3.

    Arguments:
    ----------
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file.
        n_spks (int): number of speakers
        skip_prep (bool): If True, skip data preparation
    """
    if skip_prep:
        return
    if n_spks == 2:
        create_wsj_csv(datapath, savepath)
    if n_spks == 3:
        create_wsj_csv_3spks(datapath, savepath)


# load or create the csv files for the data
def create_wsj_csv(datapath, savepath):
    """
    This function creates the csv files to get the speechbrain data loaders.

    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
    """
    # tr: train; cv: valid; tt: test
    mix_path = os.path.join(datapath, "wav16k_vd" , "max", "mix/")
    s1_path = os.path.join(datapath, "wav16k_vd", "max", "s1/")
    s2_path = os.path.join(datapath, "wav16k_vd", "max", "s2/")

    # ten cac file trong mix,s1,s2 giong nhau
    files = os.listdir(mix_path)

    mix_fl_paths = [mix_path + fl for fl in files]
    s1_fl_paths = [s1_path + fl for fl in files]
    s2_fl_paths = [s2_path + fl for fl in files]

    csv_columns = [
        "ID",
        "duration",
        "mix_wav",
        "mix_wav_format",
        "mix_wav_opts",
        "s1_wav",
        "s1_wav_format",
        "s1_wav_opts",
        "s2_wav",
        "s2_wav_format",
        "s2_wav_opts",
    ]
    # 
    with open(savepath + "/zalo_vd" + ".csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for i, (mix_path, s1_path, s2_path) in enumerate(
            zip(mix_fl_paths, s1_fl_paths, s2_fl_paths)
        ):

            row = {
                "ID": i,
                "duration": 3.0,
                "mix_wav": mix_path,
                "mix_wav_format": "wav",
                "mix_wav_opts": None,
                "s1_wav": s1_path,
                "s1_wav_format": "wav",
                "s1_wav_opts": None,
                "s2_wav": s2_path,
                "s2_wav_format": "wav",
                "s2_wav_opts": None,
            }
            writer.writerow(row)


def create_wsj_csv_3spks(datapath, savepath):
    """
    This function creates the csv files to get the speechbrain data loaders.
    Arguments:
        datapath (str) : path for the wsj0-mix dataset.
        savepath (str) : path where we save the csv file
    """
    mix_path = os.path.join(datapath, "zalo3spk/mix/")
    s1_path = os.path.join(datapath, "zalo2spk/s1/")
    s2_path = os.path.join(datapath, "zalo3spk/s2/")
    s3_path = os.path.join(datapath, "zalo3spk/s3/")

    files = os.listdir(mix_path)

    mix_fl_paths = [mix_path + fl for fl in files]
    s1_fl_paths = [s1_path + fl for fl in files]
    s2_fl_paths = [s2_path + fl for fl in files]
    s3_fl_paths = [s3_path + fl for fl in files]

    csv_columns = [
        "ID",
        "duration",
        "mix_wav",
        "mix_wav_format",
        "mix_wav_opts",
        "s1_wav",
        "s1_wav_format",
        "s1_wav_opts",
        "s2_wav",
        "s2_wav_format",
        "s2_wav_opts",
        "s3_wav",
        "s3_wav_format",
        "s3_wav_opts",
    ]

    with open(savepath + "/zalo_3spk"+ ".csv", "w") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for i, (mix_path, s1_path, s2_path, s3_path) in enumerate(
            zip(mix_fl_paths, s1_fl_paths, s2_fl_paths, s3_fl_paths)
        ):

            row = {
                "ID": i,
                "duration": 3.0,
                "mix_wav": mix_path,
                "mix_wav_format": "wav",
                "mix_wav_opts": None,
                "s1_wav": s1_path,
                "s1_wav_format": "wav",
                "s1_wav_opts": None,
                "s2_wav": s2_path,
                "s2_wav_format": "wav",
                "s2_wav_opts": None,
                "s3_wav": s3_path,
                "s3_wav_format": "wav",
                "s3_wav_opts": None,
            }
            writer.writerow(row)


def save_mixture(
    s1,
    s2,
    min_max,
    weight_1,
    weight_2,
    num_files,
    lev1,
    lev2,
    save_fs,
    output_dir,
    mix_name,
    i,
):
    """
    This function creates the mixtures, and saves them

    Arguments:
        s1, s1 (numpy array): source1 and source2 wav files in numpy array.
        weight_1, weight_2 (float): weights for source1 and source2 respectively.
        num_files (int): number of files
        lev1, lev2 (float): levels for each souce obtained with octave.activlev() function
        save_fs (str): in ['wav8k', 'wav16k']
        output_dir (str): the save directory
        data_type (str): in ['tr', 'cv', 'tt']
        mix_name (str): name given to the mixture. (see the main function get_wsj_files())
        i (int): number of the mixture. (see the main function get_wsj_files())

    """
    scaling = np.zeros((num_files, 2))
    scaling16bit = np.zeros((num_files, 1))

    if min_max == "max":
        mix_len = max(s1.shape[0], s2.shape[0])

        s1 = np.pad(
            s1, (0, mix_len - s1.shape[0]), "constant", constant_values=(0, 0),
        )
        s2 = np.pad(
            s2, (0, mix_len - s2.shape[0]), "constant", constant_values=(0, 0),
        )
    else:
        mix_len = min(s1.shape[0], s2.shape[0])

        s1 = s1[:mix_len]
        s2 = s2[:mix_len]
    
    
    if save_fs == "wav8k" :
        const_len = 8000*3
    else :
        const_len =16000*3

    if s1.shape[0] > const_len :
        s1 = s1[:const_len]
        s2 = s2[:const_len]
    else :
        s1 = np.pad(
            s1, (0, const_len - s1.shape[0]), "constant", constant_values=(0, 0),
        )
        s2 = np.pad(
            s2, (0, const_len - s2.shape[0]), "constant", constant_values=(0, 0),
        )

    mix = s1 + s2

    max_amp = max(np.abs(mix).max(), np.abs(s1).max(), np.abs(s2).max(),)
    mix_scaling = 1 / max_amp * 0.9
    s1 = mix_scaling * s1
    s2 = mix_scaling * s2
    mix = mix_scaling * mix

    scaling[i, 0] = weight_1 * mix_scaling / np.sqrt(lev1)
    scaling[i, 1] = weight_2 * mix_scaling / np.sqrt(lev2)
    scaling16bit[i] = mix_scaling
    
    sampling_rate = 8000 if save_fs == "wav8k" else 16000

    write_audio(
        output_dir
        + "/"
        + save_fs
        + "_vd/"
        + min_max
        + "/s1/"
        + mix_name
        + ".wav",
        s1,
        samplerate=sampling_rate,
    )
    write_audio(
        output_dir
        + "/"
        + save_fs
        + "_vd/"
        + min_max
        + "/s2/"
        + mix_name
        + ".wav",
        s2,
        samplerate=sampling_rate,
    )
    write_audio(
        output_dir
        + "/"
        + save_fs
        + "_vd/"
        + min_max
        + "/mix/"
        + mix_name
        + ".wav",
        mix,
        samplerate=sampling_rate,
    )
    return scaling, scaling16bit


def arrange_task_files(TaskFile, min_max, log_dir):
    """
    This function gets the specifications on on what file to read
    and also opens the files for the logs.

    Arguments:
        TaskFile (str): The path to the file that specifies the sources.
        min_max (list): Specifies whether we use min. or max. of the sources,
                        while creating mixtures
        data_type (list): Specifies which set to create, in ['tr', 'cv', 'tt']
        log_dir (str): The string which points to the logs for data creation.
    """
    with open(TaskFile, "r") as fid:
        lines = fid.read()
        C = []

        for i, line in enumerate(lines.split("\n")):
            if not len(line) == 0:
                C.append(line.split())

    Source1File = os.path.join(
        log_dir, "mix_2_spk_" + min_max + "_1"
    )
    Source2File = os.path.join(
        log_dir, "mix_2_spk_" + min_max + "_2"
    )
    MixFile = os.path.join(
        log_dir, "mix_2_spk_" + min_max + "_mix"
    )
    return Source1File, Source2File, MixFile, C


def get_wsj_files(output_dir, save_fs="wav16k", min_maxs=["max"]):
    """
    This function constructs the wsj0-2mix dataset out of wsj0 dataset.
    (We are assuming that we have the wav files and not the sphere format)

    Argument:
        wsj0root (str): This string specifies the root folder for the wsj0 dataset.
        output_dir (str): The string that species the save folder.
        save_fs (str): The string that specifies the saving sampling frequency, in ['wav8k', 'wav16k']
        min_maxs (list): The list that contains the specification on whether we take min. or max. of signals
                         to construct the mixtures. example: ["min", "max"]
    """

    from oct2py import octave

    filedir = os.getcwd()
    filedir = filedir.split("/")
    filedir = "/".join(filedir[:-2])

    octave.addpath(
        filedir +"/SpeechSeparation/main/meta"
    )  # add the matlab functions to octave dir here

    fs_read = 16000 if save_fs == "wav16k" else 8000

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if not os.path.exists(os.path.join(output_dir, save_fs)):
        os.mkdir(os.path.join(output_dir, save_fs))

    log_dir = os.path.join(output_dir, save_fs + "_vd/mixture_definitions_log")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    inner_folders = ["s1", "s2", "mix"]
    for min_max in min_maxs:
        save_dir = os.path.join(
            output_dir, save_fs + "_vd/" + min_max
        )

        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for inner_folder in inner_folders:
            if not os.path.exists(os.path.join(save_dir, inner_folder)):
                os.mkdir(os.path.join(save_dir, inner_folder))

        TaskFile = os.path.join(
            filedir, "dataset", "mix_2_spk_vd.txt"
        )
        Source1File, Source2File, MixFile, C = arrange_task_files(
            TaskFile, min_max, log_dir
        )

        fid_s1 = open(Source1File, "w")
        fid_s2 = open(Source2File, "w")
        fid_m = open(MixFile, "w")

        num_files = len(C)
        print("{} \n".format(min_max))

        for i, line in tqdm(enumerate(C)):

            inwav1_name = line[0].split("/")[-1]
            inwav2_name = line[2].split("/")[-1]

            # write the log data to the log files
            fid_s1.write("{}\n".format(line[0]))
            fid_s2.write("{}\n".format(line[2]))

            inwav1_snr = line[1]
            inwav2_snr = line[3]

            mix_name = (
                inwav1_name
                + "_"
                + str(inwav1_snr)
                + "_"
                + inwav2_name
                + "_"
                + str(inwav2_snr)
            )
            fid_m.write("{}\n".format(mix_name))

            _, fs1 = torchaudio.load(line[0])
            _, fs2 = torchaudio.load(line[2])
            s1 = read_audio(line[0])
            s2 = read_audio(line[2])

            # resample, determine levels for source 1
            s1_16k = signal.resample(s1, int((fs_read / fs1) * len(s1)))
            out = octave.activlev(s1_16k, fs_read, "n")
            s1_16k, lev1 = out[:-1].squeeze(), out[-1]
            # print('lev1 {}'.format(lev1))

            # resample, determine levels for source 2
            s2_16k = signal.resample(s2, int((fs_read / fs2) * len(s2)))
            out = octave.activlev(s2_16k, fs_read, "n")
            s2_16k, lev2 = out[:-1].squeeze(), out[-1]

            weight_1 = 10 ** (float(inwav1_snr) / 20)
            weight_2 = 10 ** (float(inwav2_snr) / 20)

            # apply same gain to 16 kHz file
            if save_fs == "wav8k":
                s1_8k = weight_1 * s1_8k
                s2_8k = weight_2 * s2_8k

                scaling_8k, scaling16bit_8k = save_mixture(
                    s1_8k,
                    s2_8k,
                    min_max,
                    weight_1,
                    weight_2,
                    num_files,
                    lev1,
                    lev2,
                    save_fs,
                    output_dir,
                    mix_name,
                    i,
                )
            elif save_fs == "wav16k":
                s1_16k = weight_1 * s1_16k / np.sqrt(lev1)
                s2_16k = weight_2 * s2_16k / np.sqrt(lev2)

                scaling_16k, scaling16bit_16k = save_mixture(
                    s1_16k,
                    s2_16k,
                    min_max,
                    weight_1,
                    weight_2,
                    num_files,
                    lev1,
                    lev2,
                    save_fs,
                    output_dir,
                    mix_name,
                    i,
                )
            else:
                raise ValueError("Incorrect sampling frequency for saving")

            if save_fs == "wav8k":
                pickle.dump(
                    {
                        "scaling_8k": scaling_8k,
                        "scaling8bit_8k": scaling16bit_8k,
                    },
                    open(
                        output_dir
                        + "/"
                        + save_fs
                        + "/"
                        + min_max
                        + "/scaling.pkl",
                        "wb",
                    ),
                )
            elif save_fs == "wav16k":
                pickle.dump(
                    {
                        "scaling_16k": scaling_16k,
                        "scaling16bit_16k": scaling16bit_16k,
                    },
                    open(
                        output_dir
                        + "/"
                        + save_fs
                        + "_vd/"
                        + min_max
                        + "/scaling.pkl",
                        "wb",
                    ),
                )
            else:
                raise ValueError("Incorrect sampling frequency for saving")


if __name__ == "__main__":
    root = os.getcwd()
    print(root)
    root = root.split('/')
    root = '/'.join(root[:-2])
    data_path = os.path.join(root, "dataset")
    print(data_path)
    get_wsj_files(data_path)
