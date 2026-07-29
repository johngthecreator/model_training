from datasets import load_dataset, DatasetDict

dataset = DatasetDict({
    "train": load_dataset("csv", data_files="training_data/combined_train.csv", split="train"),
    "validation": load_dataset("csv", data_files="training_data/combined_validation.csv", split="train"),
    "test": load_dataset("csv", data_files="training_data/combined_test.csv", split="train"),
})

# Replace with your Hugging Face username and dataset name
dataset.push_to_hub("your-username/your-dataset-name")
