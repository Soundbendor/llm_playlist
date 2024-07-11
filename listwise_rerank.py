# listwise rerank
# res/bline-chall_bm25_500_retrain2_full4_joinedstitched
import pandas as pd
import os,json,re,sys
import numpy as np
import torch

os.environ['HF_HOME'] = '/nfs/guille/eecs_research/soundbendor/zontosj/cache/'
import transformers

import getter as UG
import metrics as UM
import routes as G

num_failed = 0
gen_num = 500
test_num = 999

data_dir = "data/"
valid_dir = "valid_retrain2/"
res_dir = "res/"
df_path = G.joined_csv2_path
all_df = pd.read_csv(df_path, index_col=["uri"])

challenges = UG.get_challenges()
all_songs = UG.get_all_songs()

print(f'running on {valid_dir}')

model = "meta-llama/Meta-Llama-3-8B-Instruct"
hf_token = "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"


pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    token= hf_token,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto"
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

rng = np.random.default_rng(seed=73)

chall_todo = set(int(x) for x in sys.argv[1:]) if len(sys.argv) > 1 else set()

res_path = os.path.join(res_dir, f'chall_llama_{gen_num}_full_joined')

def llm_listwise_rerank(challenge, playlist_ids, playlist_name, candidates):
    global num_failed

    cond_num = challenge['num_cond']
    num_cand = len(candidates)

    # Construct playlist name string if the challenge has a title
    pl_name_str = f" called '{playlist_name}'" if challenge['has_title'] else ""

    # Create a string of the initial tracks in the playlist
    mask_str = ""
    for idx, uri in enumerate(playlist_ids):
        track_name = all_df.loc[uri]['track_name']
        artist_name = all_df.loc[uri]['artist_name']
        mask_str += f"{idx}. {track_name} - {artist_name}\n"

    # Create a string of the candidate tracks with indices
    cand_str = ""
    for idx, uri in enumerate(candidates):
        track_name = all_df.loc[uri]['track_name']
        artist_name = all_df.loc[uri]['artist_name']
        cand_str += f"{idx}. {track_name} - {artist_name}\n"
    
    # Construct the prompt with an example
    example_prompt = (
        "Example:\n"
        "Given the first 3 tracks in the playlist:\n"
        "1. Song A - Artist A\n"
        "2. Song B - Artist B\n"
        "3. Song C - Artist C\n\n"
        "Re-order the following candidate songs to best match the playlist:\n"
        "0. Song D - Artist D\n"
        "1. Song E - Artist E\n"
        "2. Song F - Artist F\n"
        "3. Song G - Artist G\n"
        "Please respond with the re-ordered song numbers only: 2 0 3 1"
    )
    
    prompt = (
        f"Given the first {cond_num} tracks in the playlist{pl_name_str}:\n"
        f"{mask_str}\n"
        f"Re-order the following {num_cand} candidate songs to best match the playlist:\n"
        f"{cand_str}\n"
        f"Please respond with the {num_cand} re-ordered song numbers only, separated by spaces.\n\n"
        f"{example_prompt}"
    )

    sys_prompt = "You are an AI system specialized in completing and re-ordering playlists based on given tracks and titles. Your goal is to generate the best possible continuation of the playlist by re-ordering the candidate songs provided."

    # Messages for the LLM
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": prompt},
    ]

    # Call the LLM pipeline
    outputs = pipeline(
        messages,
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.3,
        top_p=1.0,
        top_k=100,
    )

    response = outputs[0]["generated_text"][-1]['content']
    # print(f"OUTPUTS: {outputs}")
    # print(f"RESPONSE: {response}")

    # Process the response to extract numbers and reorder candidates
    try:
        special_case_indices = {num_cand, cond_num}

        # Use regex to find all numbers in the response
        reordered_indices = list(map(int, re.findall(r'\b\d+\b', response)))

        # Remove duplicates by keeping only the first occurrence
        reordered_indices = list(dict.fromkeys(reordered_indices))

        # Check for special cases and remove them
        for i in range(len(special_case_indices)):
            if reordered_indices[i] in special_case_indices:
                reordered_indices = reordered_indices[1:]
                print(f"Special case occured and removed\nSpecial case: {reordered_indices[i]}")
            
            
        # Check if the number of indices matches the number of candidates
        if len(reordered_indices) != num_cand:
            print(f"Incorrect number of indices returned by the LLM.\nnum indicies:{len(reordered_indices)}")
            if len(reordered_indices) > num_cand:
                # Remove extra indices from the end if too long
                reordered_indices = reordered_indices[:num_cand]
            else:
                # Fill in missing values in order if too short
                missing_indices = [i for i in range(num_cand) if i not in reordered_indices]
                reordered_indices.extend(missing_indices[:num_cand - len(reordered_indices)])

        # Re-order candidates according to the LLM output
        reordered_candidates = [candidates[idx] for idx in reordered_indices]
    except (ValueError, IndexError) as e:
        # If the response is incorrectly formatted, fall back to original order or handle error
        print(f"Error processing LLM output: {e}")
        print(f"LLM outputs: {outputs}")
        reordered_candidates = candidates

    print(reordered_candidates)
    exit(0)
    return reordered_candidates
    


def get_guess(callenge, candidate_songs, playlist = None, playlist_name="", is_random = False, random_uris = None):
    cond_num = callenge['num_cond']
    num_uris = len(candidate_songs)
    print(f'calculating guess... random: {is_random} with gen_num: {gen_num}, cond_num: {cond_num}, num candidates: {num_uris}')
    guess = None

    playlist_ids = None
    if is_random == False:
        playlist_ids = playlist[:cond_num]
    else:
        playlist_ids = random_uris
    
    # Call lama for listwise rerank
    
    # running this should be the same as BM25 results 
    # guess = np.array(candidate_songs)
    
    # running this 
    guess = llm_listwise_rerank(callenge, playlist_ids, playlist_name, candidate_songs)
    return guess


chall_avgarr = []
for chall in challenges:
    chall_num = chall['challenge']
    if chall_num <= 1:
        # non baseline challenge
        continue
    if len(chall_todo) > 0:
        # specified challenges to do
        if chall_num not in chall_todo:
            # not in specified challenges
            continue
    val_idx = 0
    cond_num = chall['num_cond']
    file_idx = chall['file_idx']
    is_random = chall['random']
    chall_file = chall['file']
    val_plgen = UG.playlist_csv_generator(chall_file, csv_path = valid_dir,rows = test_num)
    chall_res = []
    guess_arr = []
    anyskip = False
    for val_pl in val_plgen:
        if val_idx >= test_num:
            continue
        cfile = val_pl['file']
        cidx = int(val_pl['idx'])
        pl_json = UG.get_playlist_json(cfile)
        query_pl = pl_json['playlists'][cidx]
        cur_uris = [x['track_uri'].strip() for x in query_pl['tracks']]
        cur_pl_name = query_pl['name']

        ground_truth = np.array(cur_uris[cond_num:])
        #print(ground_truth)
        ground_truth_len = ground_truth.shape[0]
        if ground_truth_len <= 0:
            print("skip")
            anyskip = True
            break
        print(f'RUNNING EXPERIMENT {model} {val_idx+1}: CHALLENGE {chall_num}')
        print(f'file: {cfile}, guess_num: {gen_num}, cond_num: {cond_num}, gt_len: {ground_truth_len}')
        print('---------')
        random_uris = None
        if is_random == True:
            print("getting random tracks")
            random_uris = UG.get_random_songs(all_songs, rng, num=cond_num)

        # Read the JSON file
        bm25_guess_path = os.path.join("res/bline-chall_bm25_500_retrain2_full4_joinedstitched/guess_all", f'chall-bin_{file_idx}-guess_all.json')
        with open(bm25_guess_path, 'r') as f:
            bm25_data = json.load(f)

        # Set candidate_songs to the list at val_idx
        candidate_songs = bm25_data[val_idx]

        guess = get_guess(chall, candidate_songs, playlist = cur_uris, playlist_name=cur_pl_name, is_random=is_random, random_uris=random_uris)
        
        
        cur_m = UM.calc_metrics(ground_truth, guess, max_clicks=gen_num)
        guess_fname2 = f'guess_temp/chall-bin_{file_idx}-guess_{val_idx}.json'
        UM.guess_writer_flat(guess, fname=guess_fname2, fpath=res_path)
        chall_res.append(cur_m)
        guess_arr.append(np.ndarray.tolist(guess))
        UM.metrics_printer(cur_m)
        val_idx += 1
    if anyskip == True:
        print(f'skips in challenge {chall_num}')
    else:
        chall_avg = UM.get_mean_metrics(chall_res)
        chall_avgarr.append(chall_avg)
        cur_fname = f'chall-bin_{file_idx}-res.csv'
        cur_fname_avg = f'chall-bin_{file_idx}-resavg.csv'
        guess_fname = f'chall-bin_{file_idx}-guess_all.json'
        UM.metrics_writer(chall_res, fname=cur_fname, fpath=res_path)
        UM.metrics_writer([chall_avg], fname=cur_fname_avg, fpath=res_path)
        UM.guess_writer(guess_arr, fname=guess_fname, fpath=res_path)

    overall_avg = UM.get_mean_metrics(chall_avgarr)
    cur_fname2 = f'overall-res.csv'
    cur_fname_avg2 = f'overall-resavg.csv'
            
    UM.metrics_writer(chall_avgarr, fname=cur_fname2, fpath=res_path)
    UM.metrics_writer([overall_avg], fname=cur_fname_avg2, fpath=res_path)

print("num_failed:", num_failed)