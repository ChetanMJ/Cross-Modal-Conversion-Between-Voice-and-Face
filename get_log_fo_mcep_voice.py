import confirguration


for v in voice_dir_list:
    if "log_f0_"+v+".npz" in  os.listdir(os.path.join(data_dir, v)):
        continue
    print("Preprocess: " + v)
    preprocess_voice(os.path.join(data_dir, v), v)
    
