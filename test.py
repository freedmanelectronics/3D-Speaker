from speakerlab.bin.infer_diarization import Diarization3Dspeaker
import glob
import os
from tqdm import tqdm
import time

hf_token = PUT_YOUR_HF_TOKEN_HERE
audiofile = '/mnt/dataset/VoxConverse/wav/dev/abjxc.wav'
pipeline = Diarization3Dspeaker(device="cuda", include_overlap=True, hf_access_token=hf_token)

total_time = 0
for audiofile in tqdm(audiofiles):
    start_time = time.time()
    wav_id = os.path.basename(audiofile).split('.')[0]
    out_file = os.path.join(out_dir, f'{wav_id}.rttm')
    x = pipeline(audiofile)
    end_time = time.time()
    total_time += end_time - start_time
    with open(out_file, 'w') as f:
        for seg in x:
            seg_st, seg_ed, cluster_id = seg
            f.write(f'SPEAKER {wav_id} 0 {seg_st:.3f} {seg_ed-seg_st:.3f} <NA> <NA> spk_{cluster_id} <NA> <NA>\n')

print(f'Total time: {total_time}')
print(f'Average time: {total_time / len(audiofiles)}')