import random


def shuffle_lists(list1: list, list2: list) -> zip:
    combined = list(zip(list1, list2))
    random.shuffle(combined)
    return zip(*combined)
