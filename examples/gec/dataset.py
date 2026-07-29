from datasets import load_dataset, Dataset, concatenate_datasets

# ds_coedit = load_dataset("grammarly/coedit")
# ds_c4 = load_dataset("liweili/c4_200m")

# gec = ds_coedit.filter(lambda x:  x["task"] == 'gec').shuffle(seed=42)
# c4 = ds_c4["train"][:30000]

coedit = load_dataset("grammarly/coedit")
coedit_train = coedit["train"].shuffle(seed=42)

def normalize_coedit_train(dataset):
    rows = [
        {
            "src": "Correct grammar, agreement, tense, articles, and punctuation in this text: " + row["src"].split(": ", 1)[1],
            "tgt": row["tgt"],
            "task": row["task"],
            "source": "coedit",
        }
        for row in dataset
        if row["task"] == "gec"
    ]
    print(rows[0])
    return Dataset.from_list(rows)

normalize_coedit_train(coedit_train)



# def preprocess_function_jfleg(examples):
#     inputs = [ examples["sentence"][idx] for idx, correction_group in enumerate(examples['corrections']) for input in range(len(correction_group))]
#     targets = [ input for correction_group in examples['corrections'] for input in correction_group]

#     # Tokenize inputs and outputs
#     model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length')
#     labels = tokenizer(targets, max_length=160, truncation=True, padding='max_length')

#     labels["input_ids"] = [
#         [tok if tok != tokenizer.pad_token_id else -100 for tok in label]
#         for label in labels["input_ids"]
#     ]

#     model_inputs["labels"] = labels["input_ids"]

#     return model_inputs

# jfleg_gec.map(preprocess_function_jfleg, batched=True)


# coherence = ds.filter(lambda x:  x["task"] == 'coherence').shuffle(seed=42)

# gec_test = gec["train"][:2500]
# gec_validation = gec["train"][2500:5000]
# gec_train = gec["train"][5000:]

# coherence_test = coherence["train"][:2500]
# coherence_validation = coherence["train"][2500:5000]
# coherence_train = coherence["train"][5000:]

# # GEC splits
# Dataset.from_dict(gec_test).to_csv("data/gec_test.csv")
# Dataset.from_dict(gec_validation).to_csv("data/gec_validation.csv")

# # Coherence splits
# Dataset.from_dict(coherence_test).to_csv("data/coherence_test.csv")
# Dataset.from_dict(coherence_validation).to_csv("data/coherence_validation.csv")

# # Wrap back into Dataset objects
# gec_val_ds = Dataset.from_dict(gec_validation)
# coh_val_ds = Dataset.from_dict(coherence_validation)

# # Combine and shuffle
# combined_validation = concatenate_datasets([gec_val_ds, coh_val_ds]).shuffle(seed=42)

# # Wrap back into Dataset objects
# gec_test_ds = Dataset.from_dict(gec_test)
# coh_test_ds = Dataset.from_dict(coherence_test)

# # Combine and shuffle
# combined_test = concatenate_datasets([gec_test_ds, coh_test_ds]).shuffle(seed=42)

# combined_validation.to_csv("data/combined_validation.csv")
# combined_test.to_csv("data/combined_test.csv")

# gec_train = load_dataset("csv", data_files="data/gec_train.csv")["train"]
# coherence_train = load_dataset("csv", data_files="data/coherence_train.csv")["train"]

# print(len(gec_train["task"]))
# print(len(coherence_train["task"]))

# combined_train = concatenate_datasets([gec_train, coherence_train]).shuffle(seed=42)
# combined_train.to_csv("data/combined_train.csv")


