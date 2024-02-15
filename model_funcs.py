"""
This script contains model functions
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from typing import Tuple, Union
from tensorflow.keras.preprocessing import image
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

def split_train_test(
        features: pd.DataFrame,
        labels: pd.DataFrame,
        p: float = 0.2,
        mean_diff_tol: float = 0.01,
        max_diff_tol: float = 0.04
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
        - mean_diff_tol: mean difference limit on full dataset and dev set
        set to 1%
        - max_diff_tol: max difference limit set to 2% by default

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
        print("Loading existing train, dev sets...")
        dev_feat = pd.read_csv(dev_feat_dir, index_col=0)
        dev_lab = pd.read_csv(dev_lab_dir, index_col=0)
        dev_full = dev_feat.join(dev_lab)
        dev_dist = dev_full.select_dtypes(include='number').sum(axis=0).divide(dev_full.shape[0])
        dev_p = dev_full.shape[0] / m

        # check that test set found matches critera set out
        print("Checking loaded test set to match conditions...")
        if abs(dev_p - p) <= 0.01:
            print(f"Test size falls within acceptable range:\np = {p}\ndev_p = {dev_p}")
            if check_distributions(distribution, dev_dist, mean_diff_tol, max_diff_tol):
                print(
                    f"Test proportion falls within acceptable proportion:\n"
                    f"Dataset proportion = {distribution}\n"
                    f"Dev proportion = {dev_dist}\n"
                    f"Returning loaded train and dev sets"
                )
                train_feat = pd.read_csv(train_feat_dir, index_col=0)
                train_lab = pd.read_csv(train_lab_dir, index_col=0)
                execute_split = False
                return train_feat, train_lab, dev_feat, dev_lab
            else:
                print(
                    f"Test proportion does not fall within acceptable proportion:\n"
                    f"Dataset proportion:\n{distribution}\n"
                    f"Dev proportion:\n{dev_dist}"
                )
                execute_split = True
        else:
            print(f"Test size does not fall within acceptable range:\np = {p}\ndev_p = {dev_p}")
            execute_split = True
    else:
        execute_split = True

    if execute_split:
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
            current_p += (temp.shape[0] / m)

            if p >= current_p:
                if current_p - p > 0.01:
                    train_set = pd.concat([train_set, temp])
                else:
                    dev_set = pd.concat([dev_set, temp])
            else:
                train_set = pd.concat([train_set, temp])

        # 3. check if dev proportion and train proportion of animal
        # observations are roughly similar
        dev_m = dev_set.shape[0]
        dev_dist = dev_set.select_dtypes(include='number').sum(axis=0).divide(dev_m)

        # 4. If not repeat steps 2-3, otherwise store test/train sets
        count = 1
        while not check_distributions(distribution, dev_dist, mean_diff_tol, max_diff_tol):
            if count == 1500:
                raise TimeoutError("No suitable train/test split after 1500 iterations, please relax tolerance")

            # resample testing sites
            random.shuffle(random_sites)
            train_set = pd.DataFrame()
            dev_set = pd.DataFrame()

            current_p = 0.0
            # allocate sites to either dev or train such that
            # dev size = target_size (to 2 dp)
            for site in random_sites:
                temp = full_dataset.loc[full_dataset['site'] == site, :]
                current_p += (temp.shape[0] / m)

                if p >= current_p:
                    if current_p - p > 0.01:
                        train_set = pd.concat([train_set, temp])
                    else:
                        dev_set = pd.concat([dev_set, temp])
                else:
                    train_set = pd.concat([train_set, temp])

            dev_m = dev_set.shape[0]
            dev_dist = dev_set.select_dtypes(include='number').sum(axis=0).divide(dev_m)
            count += 1
        
        # split into features and labels
        dev_lab, dev_feat = split_by_type(dev_set)
        train_lab, train_feat = split_by_type(train_set)
            
        # save files
        files = [dev_feat, dev_lab, train_feat, train_lab]
        for file, path in zip(files, paths):
            file.to_csv(path)
        
        return train_feat, train_lab, dev_feat, dev_lab


def check_distributions(dist1, dist2, mean_tol, max_tol):
    mean_diff = check_mean_diff(dist1,dist2, mean_tol)
    max_diff = check_max_diff(dist1, dist2, max_tol)
    if mean_diff and max_diff:
        return True
    else:
        return False
    

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
    

def check_max_diff(a,b,tolerance):
    diff = abs(a -b)
    if max(diff) < tolerance:
        return True
    else:
        return False


def split_by_type(df, type: str = 'number'):
    df1 = df.select_dtypes(include=type)
    df2 = df.select_dtypes(exclude=type)
    return df1, df2


class ImageDataset(tf.data.Dataset):
    """
    Reads in an image, transforms pixel values, and saves
    a dictionary with the image id, tensors, and label
    """
    def __init__(
            self,
            features:pd.DataFrame,
            labels:pd.DataFrame=None,
            img_target: Tuple[int, int] = (224, 224),
            num_classes: int = 8
        ):
        self.data=features
        self.labels=labels
        self.indexes = features.index
        self.img_target = img_target
        self.classes = num_classes

    def _generator(self):
        """
        For a given index in each batch, we:
            1.) retrieve the filepath
            2.) load in the image and convert to array
            3.) conduct normalisation used in ResNet50
            4.) retrive the label if it exists
            5.) return the processed image tensor and label
        """
        for idx in self.indexes:

            # 1.) retrieve the filepath
            img_path = self.data.loc[idx, 'filepath']
            # 2.) load in the image and convert to array
            img = image.load_img(
                img_path,
                target_size=self.img_target,
                color_mode="rgb"
            )
            img_arr = image.img_to_array(img)
            # 3.) conduct normalisation used in ResNet50
            img_proc = preprocess_input(img_arr)

            # 4.) retrieve the label if it exists
            if self.labels is not None:
                label = self.labels.loc[idx].values.astype(float)
            else:
                label = None

            # 5.) return the processed image tensor and label
            yield img_proc, label

    def _as_variant_tensor(self):
        return self
    
    def _inputs(self):
        return []
    
    @property
    def element_spec(self):
        if self.labels is not None:
            return(
                tf.TensorSpec(shape=(*self.img_target, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(self.classes,), dtype=tf.float32)
            )
        else:
            return(
                tf.TensorSpec(shape=(*self.img_target, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32)
            )

    def _as_dataset(self):
        return self


class ConvModel(tf.keras.Model):

    def __init__(
            self,
            input_shape: Tuple = (224, 224, 3),
            dropout: bool = True,
            dropout_rate: float = 0.1,
            num_classes: int = 8):

        super().__init__()
        self.dropout_bool = dropout
        
        # Load pre-trained ResNet50
        # exlcude classification layers and batch dimension
        base_resnet = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        # freeze weights in pre trained layers
        for layer in base_resnet.layers:
            layer.trainable=False

        self.base = base_resnet
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(100, activation='relu')
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    
    def call(self, inputs):
        """
        This function implements the forward
        pass of the model
        """
        x = self.base(inputs)
        x = self.flatten(x)
        x = self.dense1(x)
        if self.dropout_bool:
            x = self.dropout(x)
        x = self.dense2(x)
        return x


def plot_loss(model, num_epochs: int, eval_set: bool = True):
    # start from epoch = 1 rather than 0
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, model.history['loss'], label='Training Crossentropy loss')
    if eval_set:
        plt.plot(epochs, model.history['val_loss'], label='Dev Crossentropy loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()