import numpy as np


def corpus_to_chats(corpus, labels, segment_lengths):
    start_idx = 0
    chats = []

    for idx, length in enumerate(segment_lengths):
        end_idx = start_idx + length
        start_idx = end_idx
        labels[idx] = 13 if labels[idx] == 14 else labels[idx]
        chats.append((labels[idx], corpus[start_idx - length : end_idx]))

    return chats


def chats_to_blocks(chats, block_size):
    chats_blocks = []
    for label, chat in chats:
        if len(chat) > block_size:
            blocks = [
                (label, chat[i : i + block_size])
                for i in range(0, len(chat), block_size)
            ]
            blocks = blocks[:-1] if len(blocks[-1]) < block_size else blocks
            chats_blocks.extend(blocks)

    return chats_blocks


def load_data(split, block_size):
    corpus_filename, corpus_dtype = f"./data/{split}_corpus.bin", np.uint16
    chat_levels_filename, chat_levels_dtype = (
        f"./data/{split}_chat_levels.bin",
        np.uint8,
    )
    segments_filename, segments_dtype = f"./data/{split}_segment_lengths.bin", np.uint16

    corpus = np.memmap(corpus_filename, dtype=corpus_dtype, mode="r")  # corpus
    labels = np.memmap(
        chat_levels_filename, dtype=chat_levels_dtype, mode="r"
    ).copy()  # labels
    segment_lengths = np.memmap(
        segments_filename, dtype=segments_dtype, mode="r"
    )  # corpuses lengths

    chats = corpus_to_chats(corpus, labels, segment_lengths)
    data = chats_to_blocks(chats, block_size)
    return data
