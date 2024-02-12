"""
This script contains model functions
"""

import os
import random
import numpy as np
import pandas as pd

def split_train_test(
        features: pd.DataFrame,
        labels: pd.DataFrame,
        p: float = 0.2,
        tolerance: float = 0.025
):
    """
    To create the train dev split, each camera site is used
    either for testing or training to avoid overfitting
    to environmental conditions in the images. This should mean
    the model can generalise to new unseen sites well.
    To do this the following steps are executed:
        1. calculate the dataset distribution of animals
        2. randomly sample dev sites to approximately meet 1
        3. check if dev proportion and train proportion of animal
        observations are roughly similar
        4. If not repeat steps 2-3, otherwise store dev/train set
    This method may face issues with sufficiently small dev sets that 1 or 2 sites
    can make up the entire dev set - therefore a lower bound of 10% has been
    set on the size of the dev set.

    If such a split already exists, it is read in and returned, otherwise
    it is replaced with the new split

    Arguments:
        - features: the features dataframe with columns filepath and site indxed by id
        - labels: labels corresponding to each id in features indexed by id
        - p: proportion of dataset to put into dev set, 20% by default
        - tolerance: mean squared difference limit on full dataset and dev set
        set to 2.5% by default

    """
    assert p >= 0.1, "Dev size proportion should not be below 0.1 to ensure it cannot be met by 1 large site"

    full_dataset = features.join(labels)

    # m = number of observations in dataset
    m = full_dataset.shape[0]
    # s = number of sites
    s = len(full_dataset['site'].unique())

    # 1. calculate the dataset distribution of animals
    distribution = full_dataset.select_dtypes(include='number').sum(axis=0).divide(m)

    # check if version already exists and if so if it
    # meets the criteria implied by p
    dir = os.path.join("data", "interim_files")
    
    dev_feat_dir = os.path.join(dir, "dev_features.csv")
    dev_lab_dir = os.path.join(dir, "dev_labels.csv")
    train_feat_dir = os.path.join(dir, "train_features.csv")
    train_lab_dir = os.path.join(dir, "train_labels.csv")
    paths = [dev_feat_dir, dev_lab_dir, train_feat_dir, train_lab_dir]

    if all(os.path.exists(path) for path in paths):
        print("Loading existing train / dev sets...")
        dev_feat = pd.read_csv(dev_feat_dir)
        dev_lab = pd.read_csv(dev_lab_dir)
        dev_full = dev_feat.join(dev_lab)
        dev_dist = dev_full.sum(axis=0).divide(dev_full.shape[0])
        dev_p = dev_full.shape[0] / m

        # check that test set found matches critera set out
        print("Checking loaded test set to match conditions...")
        if abs(dev_p - p) <= 0.01:
            print(f"Test size falls within acceptable range:\np = {p}\ndev_p = {dev_p}")
            if check_mean_diff(distribution, dev_dist, tolerance):
                print(
                    f"Test proportion falls within acceptable proportion:\n"
                    f"Dataset proportion = {distribution}\n"
                    f"Dev proportion = {dev_dist}\n"
                    f"Returning loaded train and dev sets"
                )
                train_feat = pd.read_csv(train_feat_dir)
                train_lab = pd.read_csv(train_lab_dir)
                return train_feat, train_lab, dev_feat, dev_lab
            else:
                print(
                    f"Test proportion does not fall within acceptable proportion:\n"
                    f"Dataset proportion = {distribution}\n"
                    f"Dev proportion = {dev_dist}"
                )
        else:
            print(f"Test size does not fall within acceptable range:\np = {p}\ndev_p = {dev_p}")

            
    else:
        # check in interim file folder exists and if not create it
        dir = os.path.join("data", "interim_files")
        if not os.path.exists(dir):
            os.makedirs(dir)

        # 1. calculate the dataset distribution of animals
        distribution = full_dataset.select_dtypes(include='number').sum(axis=0).divide(m)

        # 2. randomly sample test sites to approximately meet 1
        random_sites = full_dataset['site'].unique()
        random.shuffle(random_sites)

        train_set = pd.DataFrame()
        dev_set = pd.DataFrame()

        current_p = 0.0

        # allocate sites to either dev or train such that
        # dev size = target_size (to 2 dp)
        for site in random_sites:
            temp = full_dataset.loc[full_dataset['site'] == site, :]
            current_p += temp.shape[0] / m

            if abs(current_p - p) <= 0.01:
                dev_set = dev_set.concat(temp)
            else:
                train_set = train_set.concat(temp)

        # 3. check if dev proportion and train proportion of animal
        # observations are roughly similar
        dev_m = dev_set.shape[0]
        dev_dist = dev_set.select_dtypes(include='number').sum(axis=0).divide(dev_m)

        # 4. If not repeat steps 2-3, otherwise store test/train sets
        count = 1
        while not check_mean_diff(distribution, dev_dist, tolerance):
            if count == 10:
                raise TimeoutError("No suitable train/test split after 10 iterations, please relax tolerance")
            # resample testing sites
            random.shuffle(random_sites)
            train_set = pd.DataFrame()
            dev_set = pd.DataFrame()

            current_p = 0.0

            # allocate sites to either dev or train such that
            # dev size = target_size (to 2 dp)
            for site in random_sites:
                temp = full_dataset.loc[full_dataset['site'] == site, :]
                current_p += temp.shape[0] / m

                if abs(current_p - p) <= 0.01:
                    dev_set = dev_set.concat(temp)
                else:
                    train_set = train_set.concat(temp)

            dev_m = dev_set.shape[0]
            dev_dist = dev_set.select_dtypes(include='number').sum(axis=0).divide(dev_m)
            count += 1
        
        # split into features and labels
        dev_lab, dev_feat = split_by_type(dev_set)
        train_lab, train_feat = split_by_type(train_set)
            
        # save files
        files = [dev_feat, dev_lab, train_feat, train_lab]
        for file, path in files, paths:
            file.to_csv(path)
        
        return train_feat, train_lab, dev_feat, dev_lab

def check_mean_diff(a, b, tolerance):
    """
    This function checks if the mean difference between 2 series
    fall within a range of tolerance
    """
    mean_diff = sum(abs(a - b)) / len(a)
    if mean_diff < tolerance:
        return True
    else:
        return False


def split_by_type(df, type: str = 'number'):
    df1 = df.select_dtypes(include=type)
    df2 = df.select_dtypes(exclude=type)
    return df1, df2