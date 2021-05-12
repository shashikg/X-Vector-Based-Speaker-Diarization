import os
import shutil

from google_colab_plugins import playVideo
from pytube import YouTube
from moviepy.editor import VideoFileClip
from google.colab import files
from utils import make_rttm

from tqdm.auto import tqdm
import moviepy.editor as mpe
import cv2

import numpy as np

def downloadYouTube(videourl, path):
    '''
    To download youtube video
    '''

    yt = YouTube(videourl)
    yt = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().last()
    if not os.path.exists(path):
        os.makedirs(path)

    save_dir = yt.download(path)
    return save_dir.split("/")[-1]

def loadVideoFile(playvideo_file=False):
    '''
    Helper module to make demo interactive
    '''

    if os.path.exists("demo/"):
        shutil.rmtree("demo/")

    choice_ip = input("How do you want to load the video files? Enter the option as A, B, or C.\nA) You have a YouTube link\nB) You have a direct video link\nC) You want to upload a video file from your local drive (slow)\n\nEnter your choice here: ")
    choice_ip = choice_ip.capitalize()

    choice_dic = {'A': 'A) You have a YouTube link',
                'B': 'B) You have a direct video link',
                'C': 'C) You want to upload a video file from your local drive'}

    if choice_ip in choice_dic.keys():
        print("You selected the following choice: " + choice_dic[choice_ip], "\n")
    else:
        print("Please enter a valid option. Make sure you do not enter any spaces or dots or brackets.\n")
        loadVideoFile()

    if choice_ip == 'A':
        video_link = input("Please enter your YouTube video link: ")
        video_name = downloadYouTube(video_link, 'demo/video')
    elif choice_ip == 'B':
        print("This option is currently not enabled. Please select another option.\n")
        loadVideoFile()
        video_link = input("Please enter your direct video link: ")
    elif choice_ip == 'C':
        print("Please choose your video file using the button below")
        uploaded = files.upload()
        for fn in uploaded.keys():
            video_name = fn
            shutil.copy(video_name, "demo/video/"+video_name)
            os.remove(video_name)

    if " " in video_name:
        os.rename("demo/video/"+video_name, "demo/video/"+video_name.replace(" ", "_"))
        video_name = video_name.replace(" ", "_")

    print("\nYour video file name is:", video_name, "\n")

    clip = VideoFileClip("demo/video/"+video_name)

    # Create dummy rttm file
    video_len = int((clip.duration*1000)//750)
    tmp_diarization_prediction = np.zeros(video_len)
    tmp_diarization_prediction[-1] = -1
    make_rttm("demo/rttm/", video_name.split(".")[0], tmp_diarization_prediction, 750)

    # Extract audio file
    if not os.path.exists("demo/audio/"):
        os.makedirs("demo/audio/")

    clip.audio.write_audiofile("demo/audio/" + video_name.split(".")[0] + ".wav", fps=16000, bitrate='256k')

    if playvideo_file:
        print("Opening video player this will take some time...")
        playVideo(filename="demo/video/"+video_name)

    return "demo/video/"+video_name

def read_rttm(path):
    '''
    Read RTTM Diarization file
    '''

    rttm_out = [] # returns list of list containing start frame, end frame, spkid
    with open(path, "r") as f:
        for line in f:
            entry = line[:-2].split()

            indexes = [0, 1, 2, 5, 6, 8, 9]
            for index in sorted(indexes, reverse=True):
                del entry[index]

            entry[0] = int(float(entry[0])*16000) # Start frame
            entry[1] = entry[0] + int(float(entry[1])*16000) # End frame
            entry[2] = int(entry[2][-1:]) # Label
            rttm_out.append(entry)

        # Sort rttm list according to start frame
        rttm_out.sort(key = lambda x: x[0])

    hypothesis_labels = np.array(rttm_out)
    spkCnt = 0
    spkMap = {}
    for tmp in hypothesis_labels[:, 2]:
        if not tmp in spkMap.keys():
            spkCnt += 1
            spkMap[tmp] = spkCnt

    for i in range(len(hypothesis_labels)):
        hypothesis_labels[i, 2] = spkMap[hypothesis_labels[i, 2]]

    return hypothesis_labels

def combine_audio(vidname, audname, outname, fps=25):
    '''
    openCV makes video file without audio. So explicitly merge the extracted audio to annotated video file.
    '''

    my_clip = mpe.VideoFileClip(vidname)
    audio_background = mpe.AudioFileClip(audname)
    final_clip = my_clip.set_audio(audio_background)
    final_clip.write_videofile(outname,fps=fps)

def createAnnotatedVideo(audio_dataset, hypothesis_dir):
    '''
    Used to annotate the video with predicted diarization label.
    Simply adds the predicted label as text on the video.

    Assues that the audio_dataset contains only one single file. And its predicted diarization labels are inside hypothesis_dir.
    '''
    
    orig_video_dir = audio_dataset.data_dir+"video/" + audio_dataset.filelist[0].split(".")[0] + ".mp4"
    cap = cv2.VideoCapture(orig_video_dir)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    hypothesis_labels = read_rttm(hypothesis_dir + audio_dataset.filelist[0].split(".")[0] + ".rttm")
    hypothesis_labels[:,0] = hypothesis_labels[:,0]*fps/16000.0 # start time in frames
    hypothesis_labels[:,1] = hypothesis_labels[:,1]*fps/16000.0 # end time in frames
    spk_dict = {i: str(i//10) + str(i%10) for i in np.unique(hypothesis_labels[:, 2])}

    currspk = "Speaker: None"
    font = cv2.FONT_HERSHEY_TRIPLEX
    org = (W//12 , H*3//4)
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2
    ip_vid_name = './temp.mp4'

    if not cap.isOpened(): # check if video file is opened
        cap.open()

    size = (W, H)
    out = cv2.VideoWriter(ip_vid_name, cv2.VideoWriter_fourcc(*'MJPG'), fps, size)
    curr_idx_hypo_array = 0
    for frame_count in tqdm(range(1, n_frames+1), desc="Annotating video file..."):
        ret, frame = cap.read()
        if ret == False:
            break

        # Get current speaker from hypothesis_labels array
        if hypothesis_labels[curr_idx_hypo_array, 0] > max(0, frame_count):
            currspk = "Speaker: None"
        elif hypothesis_labels[curr_idx_hypo_array, 0] <= max(0, frame_count) and\
            hypothesis_labels[curr_idx_hypo_array, 1] >= max(0, frame_count):
            currspk = "Speaker: " + spk_dict[hypothesis_labels[curr_idx_hypo_array, 2]]
        else:
            curr_idx_hypo_array = min(curr_idx_hypo_array + 1, len(hypothesis_labels) - 1)
            currspk = "Speaker: None"

        # Using cv2.putText() method
        frame_annotated = cv2.putText(frame, currspk, org, font, fontScale, [0,0,255], 3*thickness, cv2.LINE_AA) # Give black outline to text
        frame_annotated = cv2.putText(frame, currspk, org, font, fontScale, color, thickness, cv2.LINE_AA)
        out.write(frame) # write frames to video file

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if not os.path.exists(audio_dataset.data_dir + "out_video/"):
        os.makedirs(audio_dataset.data_dir + "out_video/")

    print("Combine audio and video files...\n")
    ip_aud_name = audio_dataset.root_dir + audio_dataset.filelist[0]
    op_vid_name = audio_dataset.data_dir + "out_video/" + audio_dataset.filelist[0].split(".")[0] + ".mp4"
    combine_audio(ip_vid_name, ip_aud_name, op_vid_name, fps=fps)

    return op_vid_name
