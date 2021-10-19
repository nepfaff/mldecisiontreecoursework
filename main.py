#!/usr/bin/env python3

from numpy.random import default_rng

from data_loading import load_txt_data, split_dataset_into_train_and_test


def main() -> None:
    # Load data
    x_clean, y_clean = load_txt_data(
        "./Data/intro2ML-coursework1/wifi_db/clean_dataset.txt", 7
    )
    x_noisy, y_noisy = load_txt_data(
        "./Data/intro2ML-coursework1/wifi_db/noisy_dataset.txt", 7
    )

    # Seed random generator for randomly splitting dataset into train and test sets
    seed = 500
    rg = default_rng(seed)

    # Split datasets into train and test sets
    (
        x_clean_train,
        y_clean_train,
        x_clean_test,
        y_clean_test,
    ) = split_dataset_into_train_and_test(x_clean, y_clean, 0.2, rg)
    (
        x_noisy_train,
        y_noisy_train,
        x_noisy_test,
        y_noisy_test,
    ) = split_dataset_into_train_and_test(x_noisy, y_noisy, 0.2, rg)

    # TODO: Continue here


if __name__ == "__main__":
    main()
