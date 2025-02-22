import openml
import os
import json


import re


def format_string(text):
    # Normalize whitespace and line breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(
        r"^\s*([A-Z ]+)\s*\n\s*[-~]*\s*$",
        lambda m: f"### {m.group(1).title()}",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^\*?\s*\*{2}([a-zA-Z ]+)\*{2}:?(.*)",
        lambda m: f"**{m.group(1).strip()}**: {m.group(2).strip()}",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^\s*([•*-])\s*(\*{2}.*\*{2})",
        lambda m: f"* {m.group(2)}",
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"```(.*?)```",
        lambda m: "```\n"
        + "\n".join([line.strip() for line in m.group(1).split("\n")])
        + "\n```",
        text,
        flags=re.DOTALL,
    )
    text = re.sub(
        r"^\s*\* \*{2}(\w+)\*{2}: (\d+)", r"* **\1**: \2", text, flags=re.MULTILINE
    )
    text = re.sub(r"^(\s{4,})(\* )", r"  \2", text, flags=re.MULTILINE)
    text = "\n".join([line.strip() for line in text.split("\n")])
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def create_ipynb_files(dataset_ids_dict, output_dir="notebooks"):
    os.makedirs(output_dir, exist_ok=True)

    for dataset_id, dataset_type in dataset_ids_dict.items():
        dataset = openml.datasets.get_dataset(dataset_id)
        desc = format_string(dataset.description)
        name = format_string(dataset.name)
        print(f"Creating notebook for dataset {dataset_id}: {dataset.name}")

        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        f"# OpenML dataset: **_{name}_**",
                        "## **Description**",
                        desc,
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import openml\n\n",
                        f"dataset = openml.datasets.get_dataset({dataset_id})\n",
                        "X, y, cat_atr_mask, names = dataset.get_data(target=dataset.default_target_attribute, dataset_format='dataframe')",
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": ["X.head()"],
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": ["y.head()"],
                },
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "codemirror_mode": {"name": "ipython", "version": 3},
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.5",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 4,
        }

        safe_name = "".join(
            c if c.isalnum() or c in " ._-()" else "_" for c in dataset.name
        )
        file_name = os.path.join(output_dir, f"{safe_name}.ipynb")

        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(notebook_content, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    openml_ids_reg = [
        42726,
        44957,
        44958,
        531,
        44994,
        42727,
        44959,
        44978,
        44960,
        44965,
        44973,
        42563,
        44980,
        42570,
        43071,
        41021,
        44981,
        44970,
        550,
        41980,
        546,
        541,
        507,
        44967,
        505,
        422,
        42730,
        416,
    ]
    openml_ids_clf = [
        41156,
        40981,
        1464,
        40975,
        40701,
        23,
        31,
        40670,
        188,
        1475,
        4538,
        41143,
        1067,
        3,
        41144,
        12,
        1487,
        1049,
        41145,
        1489,
        1494,
        40900,
        40984,
        40982,
        41146,
        54,
        40983,
        40498,
        181,
    ]
    openml_ids = {id: "regression" for id in openml_ids_reg}
    openml_ids.update({id: "classification" for id in openml_ids_clf})

    create_ipynb_files(dataset_ids_dict=openml_ids, output_dir="..\\notebooks\\openml")
    print("Notebooks created successfully")
