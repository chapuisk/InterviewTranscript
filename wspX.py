import os.path

import whisperx
from pydub import AudioSegment
import gc

people = "NameOfFile"
the_path = "/Path/to/the/Interview/"
wav_ext = ".wav"
m4a_ext = ".m4a"

# Test .wav audio file first
audio_file = the_path + people + wav_ext

# If the file does not exist, try .m4a format and converts it to .wav
if not os.path.exists(audio_file):
    audio_file = the_path + people + m4a_ext
    track = AudioSegment.from_file(audio_file,  format="m4a")
    track.export(the_path + people + wav_ext, format='wav')
    audio_file = the_path + people + wav_ext

# WhisperX on cpu with low tech requirements
device = "cpu"
batch_size = 16 # reduce if low on GPU mem
compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
# print(result["segments"]) # before alignment

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

output = ""
for t in result["segments"]:
    output += " "+t["text"]

with open("/Path/to/the/interview/" + people + '.txt', 'a') as f:
    f.write(output)

# Next section is about testing diarization of discourse, but fail to obtain something coherent
# Required a WhisperX token
# mytoken = "hf_TOaLvcbzJS************yTFrmWWrQirp"

# delete model if low on GPU resources
# import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
# diarize_model = whisperx.DiarizationPipeline(use_auth_token=mytoken, device=device)

# add min/max number of speakers if known
# diarize_segments = diarize_model(audio_file)
# diarize_model(audio_file, min_speakers=2, max_speakers=2)
#
# result = whisperx.assign_word_speakers(diarize_segments, result)
#print(diarize_segments)
#print(result["segments"]) # segments are now assigned speaker IDs

# speaker = ""
# speaker_diar = ""
# for t in result["segments"]:
#     s = t["speaker"]
#     if s == speaker:
#         speaker_diar += t["text"]
#     else:
#         with open("/Path/to/the/interview/" + people + '.txt', 'a') as f:
#             f.write(speaker+": "+speaker_diar+"\n")
#         speaker = s
#         speaker_diar = ""

