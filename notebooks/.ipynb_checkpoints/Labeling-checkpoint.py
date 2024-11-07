# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Imports

# %%
import os

import numpy as np
import pandas as pd

from transformers import pipeline


# %% [markdown]
# # Data

# %%
fake = pd.read_csv("../data/Fake.csv")
true = pd.read_csv("../data/True.csv")

# %%
fake

# %%
data = pd.concat([
    fake.iloc[:100],
    true.iloc[:100]
], axis=0).reset_index()

# %% [markdown]
# # Labeling

# %%
emotional_clickbait_classifier = pipeline("text-classification", model="elozano/bert-base-cased-clickbait-news")

def emotional_clickbait(data: pd.DataFrame) -> pd.DataFrame:
    title = data["title"].to_list()
    text = data["text"].to_list()

    title_class = emotional_clickbait_classifier(title)
    text_class = emotional_clickbait_classifier(title)
    
    title_labels = pd.DataFrame(title_class)["label"].map(lambda label: 1 if label == "Clickbait" else 0)
    title_scores = pd.DataFrame(title_class)["score"]
    
    text_labels = pd.DataFrame(text_class)["label"].map(lambda label: 1 if label == "Clickbait" else 0)
    text_scores = pd.DataFrame(text_class)["score"]

    result = pd.concat([title_labels, title_scores, text_labels, text_scores], axis=1)
    result.columns = ["title_labels", "title_scores", "text_labels", "text_scores"]
    
    return result


# %%
emotional_clickbait(data.iloc[:5])


# %%
def whataboutism(title: str, text: str) -> bool:
    raise NotImplementedError


# %%
def trolling(title: str, text: str) -> bool:
    raise NotImplementedError


# %%
def polarization(title: str, text: str) -> bool:
    raise NotImplementedError
