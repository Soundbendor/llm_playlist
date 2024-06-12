import pandas as pd
import os,csv,json,datetime,re

import getter as UG
import routes as G

cond_num = 10
gen_num = 100
num_samples = 100
context_num = 500

val_path = "data/filtered_validation_set.csv"
model = "llama3"
songs_pth = G.songs_path
res_dir = "res/"
context_dir = f"baseline_bm25_filt_500_real"
res_path = f"llama_preds/{context_dir}_{model}.json"
context_dir = f"{res_dir}{context_dir}"

import transformers
import torch

transformers.logging.set_verbosity_error()

os.environ['HF_HOME'] = '/nfs/guille/eecs_research/soundbendor/zontosj/.cache/'
os.environ['TRANSFORMERS_CACHE'] = '/nfs/guille/eecs_research/soundbendor/zontosj/.cache/'

# model_id = "meta-llama/Llama-2-13b-hf"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
# model_id = "unsloth/llama-3-8b-bnb-4bit"
hf_token = ""

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    token= hf_token,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

songs_df = pd.read_csv(songs_pth, index_col=None, header=0)
songs_df.set_index(["uri"], inplace=True)

# Initialize a list to store the lists of URLs
candidates = {}

# Loop through each .txt file in the context directory
for filename in sorted(os.listdir(context_dir)):
    if filename.endswith('.txt'):
        print(filename)
        match = re.search(r'\d+', filename)
        if match:
            pl_idx = int(match.group(0))
            print(pl_idx)
        else:
            print("No number found in filename.")

        file_path = os.path.join(context_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            pl_cands = []
            for track_idx, line in enumerate(file):
                uri = line[:-1]
                track = songs_df.loc[uri]
                track_name = track["track_name"]
                artist_name = track["artist_name"]
                # str_track_artist = f"[{track_idx}] {track_name} - {artist_name}"
                str_track_artist = f"{track_name} - {artist_name}"
                pl_cands.append((uri, str_track_artist))
            candidates[pl_idx] = pl_cands

def get_playlists(csv_path, sample_num=500):
    df = pd.read_csv(csv_path)
    cur_playlists = df.to_dict('records')
    
    return cur_playlists[:sample_num]

playlists = get_playlists(val_path, sample_num = num_samples)

requests = []

def process_llm_output(output: str) -> int:
    output = output.strip()

    match = re.search(r'\d+', output)

    if match:
        return int(match.group(0))
    else:
        raise ValueError("No valid number found in the output")


def pick_left(left, right, mask_tracks):
    l_uri, l_track_str = left
    r_uri, r_track_str = right

    # Create the prompt
    mask_str = ""
    for t in mask_tracks:
        mask_str += f"{t['track_name']} - {t['artist_name']}\n"
    prompt = (
        f"Given the first 10 tracks in the playlist:\n"
        f"{mask_str}"
        f"Which of the following songs best fits this playlist?\n"
        f"[1] {l_track_str}\n"
        f"[2] {r_track_str}\n"
        f"Please respond with just the number of the chosen song."
    )

    messages = [
        {"role": "system", "content": "You are a music recommendation chatbot."},
        {"role": "user", "content": prompt},
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=32,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    # Extract the chosen song number from the generated text
    response = outputs[0]["generated_text"][-1]['content']

    try:
        chosen_song = process_llm_output(response)
    except ValueError as e:
        print(e)
        chosen_song = 1

    # print(prompt)
    # print(response)
    # print(chosen_song)
    
    return chosen_song != 2


def merge(left, right, mask_tracks):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        # if left[i] <= right[j]:
        if pick_left(left[i], right[j], mask_tracks):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    
    return result

def merge_sort(arr, mask_tracks):
    if len(arr) <= 1:
        return arr
    
    mid = len(arr) // 2
    left = merge_sort(arr[:mid], mask_tracks)
    right = merge_sort(arr[mid:], mask_tracks)
    
    return merge(left, right, mask_tracks)

res = []

for idx, gt_pl in enumerate(playlists):
    pl_candidates = candidates[idx]
    mask_tracks = UG.get_playlist(gt_pl['file'], int(gt_pl['idx']))['tracks'][:cond_num]

    for t in mask_tracks:
        print(f"{t['track_name']} - {t['artist_name']}")
    for uri, track in pl_candidates:
        print(uri)
        print(track)
    start_time = datetime.datetime.now()
    sorted_pl_candidates = merge_sort(pl_candidates, mask_tracks)
    end_time = datetime.datetime.now()
    duration = end_time - start_time

    print("===============SORTED===============")
    print(f"Duration: {duration.total_seconds()}")
    for uri, track_str in sorted_pl_candidates:
        print(uri)
        print(track_str)
    
    sorted_uris = [ uri for uri, _ in sorted_pl_candidates]
    res.append(sorted_uris)


print(f"done. savinf res to {res_path}")
with open(res_path, 'w') as json_file:
    json.dump(res, json_file)
