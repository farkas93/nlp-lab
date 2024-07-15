import config
import logging
from typing import List
from datasets import load_dataset, concatenate_datasets

def generate_splits(d_name:str, splits: List, subset=None, split_aliases=None):

    train_data = None
    test_data = None

    dataset = load_dataset(d_name, name=subset, cache_dir=config.DATA_CACHE_DIR) if subset else load_dataset(d_name, cache_dir=config.DATA_CACHE_DIR)
    logging.info(f"Loaded {d_name} with all splits and subset {subset}")
    if splits is None or len(splits) == 1:
            # Create a split
            if splits is None:
                shuffled_dataset = dataset.shuffle(seed=42)
            else:
                shuffled_dataset = dataset[splits[0]].shuffle(seed=42)
            split_dataset = shuffled_dataset.train_test_split(test_size=0.1, seed=42)
            train_data = split_dataset['train']
            test_data = split_dataset['test']
    else:
        if split_aliases:
            train_split_name = split_aliases['train']
            test_split_name = split_aliases['test']
            return dataset[train_split_name], dataset[test_split_name]
        
        if 'train' in splits and 'test' in splits:
            return dataset['train'], dataset['test']
        else:
            if not ('train' in splits):
                logging.warn(f"SOMETHING IS WEIRD. THESE ARE THE SPLITS: {splits}")
                for s in splits:
                    shuffled_dataset = dataset[s].shuffle(seed=42)
                    split_dataset = shuffled_dataset.train_test_split(test_size=0.1, seed=42)

                    if train_data is None:
                        train_data = split_dataset['train']
                        test_data = split_dataset['test']
                    else:
                        train_data = concatenate_datasets([train_data, split_dataset['train']])
                        test_data = concatenate_datasets([test_data, split_dataset['test']])
            else:
                # We know that the train split is always listed as first.
                return dataset['train'], dataset[splits[1]]
    return train_data, test_data


def load_dataset_with_splits_and_subsets(dataset_name, dataset_conf):
    splits = dataset_conf.get("splits")
    subsets = dataset_conf.get("subsets")
    dsc_keys= dataset_conf.keys()
    
    train_data = None
    test_data = None
    sub = "None"
    try:
        if "split_aliases" in dsc_keys:
            train_data, test_data = generate_splits(d_name=dataset_name, 
                                                    splits=None, 
                                                    subset=sub, 
                                                    split_aliases=dataset_conf['split_aliases'])

        elif subsets is None:
            train_data, test_data = generate_splits(d_name=dataset_name, splits=splits, subset=sub)
        else:
            for sub in subsets:
                temp_train, temp_test = generate_splits(d_name=dataset_name, splits=splits, subset=sub)
                if train_data is None:
                    train_data = temp_train
                    test_data = temp_test
                else:
                    train_data = concatenate_datasets([train_data, temp_train])
                    test_data = concatenate_datasets([test_data, temp_test])
    except Exception as e:
        logging.info(f"Failed to load {dataset_name} with subset {sub}: {str(e)}")

    
    formatter_func = dataset_conf['formatter']

    if not formatter_func:
        # Early out if no formatting needed
        return train_data, test_data

    # Debugging: Print the sample formatted output
    logging.debug("Before Mapping:")
    for idx in range(min(7, len(train_data))):
        sample_output = formatter_func(train_data[idx])
        logging.debug(f"Sample formatted output {idx}: {sample_output}")

    # Apply formatter to the dataset
    formatted_train_data = train_data.map(lambda example: formatter_func(example), remove_columns=train_data.column_names)
    formatted_test_data = test_data.map(lambda example: formatter_func(example), remove_columns=train_data.column_names)
     
    # Debugging: Print the dataset after mapping
    logging.debug("After Mapping:\n==========\n")
    for idx in range(min(5, len(formatted_train_data))):
        logging.debug(f"Mapped output {idx}: {formatted_train_data[idx]}")

    return formatted_train_data, formatted_test_data 


def create_data_partition(train, test, ind, max_samples = config.MAX_TRAIN_SAMPLES_IN_MEMORY):
   
    max_test_samples = max(500, int(0.2*max_samples))

    #create a test batch 10% the size of the train batch
    if len(test) < max_test_samples:
        test_batch = test
    else:
        shuffled_dataset = test.shuffle() 
        test_batch = shuffled_dataset.select(range(100)) 

    if len(train) < max_samples:
        return train, test_batch, 0, 0
    
    next_ind = ind+1
    max_ind = len(train)/max_samples +1
    if next_ind*max_samples < len(train):
        train_batch = train.select(range(ind*max_samples, next_ind*max_samples))
    else:
        train_batch = train.select(range(ind*max_samples, len(train)))

    logging.info(f"selected {len(train_batch)} samples. {ind}/{max_ind}")
    return train_batch, test_batch, next_ind, max_ind



def sample_from_dataset(train, test, max_samples = config.MAX_TRAIN_SAMPLES_IN_MEMORY, sample_seed=42):    
    max_test_samples = max(500, int(0.2*max_samples))

    #create a test batch 10% the size of the train batch
    if len(test) < max_test_samples:
        test_batch = test
    else:
        shuffled_dataset = test.shuffle() 
        test_batch = shuffled_dataset.select(range(max_test_samples)) 

    if len(train) < max_samples:
        return train, test_batch
    else:
        shuffled_dataset = train.shuffle(seed=sample_seed) 
        train_batch = shuffled_dataset.select(range(max_samples))

    return train_batch, test_batch