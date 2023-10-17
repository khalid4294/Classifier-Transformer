import numpy as np


def corpus_to_documents(corpus, labels, document_lengths):
    """
    Split corpus into documents based on document lengths
    """
    start_idx = 0
    documents = []

    for idx, length in enumerate(document_lengths):
        end_idx = start_idx + length
        start_idx = end_idx
        labels[idx] = 13 if labels[idx] == 14 else labels[idx]
        documents.append((labels[idx], corpus[start_idx - length : end_idx]))

    return documents


def documents_to_blocks(documents, block_size):
    """
    Split documents into blocks of size block_size
    """
    documents_blocks = []
    for label, document in documents:
        if len(document) > block_size:
            blocks = [
                (label, document[i : i + block_size])
                for i in range(0, len(document), block_size)
            ]
            blocks = blocks[:-1] if len(blocks[-1]) < block_size else blocks
            documents_blocks.extend(blocks)

    return documents_blocks


def load_data(split, block_size):
    """
    Load data from binary files
    """
    corpus_filename, corpus_dtype = f"./data/{split}_corpus.bin", np.uint16
    document_labels_filename, document_labels_dtype = (
        f"./data/{split}_labels.bin",
        np.uint8,
    )
    documents_filename, documents_dtype = (
        f"./data/{split}_document_lengths.bin",
        np.uint16,
    )

    corpus = np.memmap(corpus_filename, dtype=corpus_dtype, mode="r")  # corpus
    labels = np.memmap(
        document_labels_filename, dtype=document_labels_dtype, mode="r"
    ).copy()  # labels
    document_lengths = np.memmap(
        documents_filename, dtype=documents_dtype, mode="r"
    )  # corpuses lengths

    documents = corpus_to_documents(corpus, labels, document_lengths)
    data = documents_to_blocks(documents, block_size)
    return data
