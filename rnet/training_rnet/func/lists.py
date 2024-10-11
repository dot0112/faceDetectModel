import random


def shuffle_lists(lists: list[list]) -> tuple:
    list_combined = list(zip(*lists))
    random.shuffle(list_combined)
    return tuple(map(list, zip(*list_combined)))


def split_lists(
    lists: list[list], split_ratio: float = 0.5
) -> tuple[list, list, list, list, list, list]:
    train, val = zip(
        *[
            (lst[: int(len(lst) * split_ratio)], lst[int(len(lst) * split_ratio) :])
            for lst in lists
        ]
    )
    return (train[0], val[0], train[1], val[1], train[2], val[2])
