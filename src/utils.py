def merge_lists_alternating(list1: list, list2: list) -> list:
    """
    Helper function that merges 2 lists in an alternating fashion, i.e. it would merge
    ['a', 'c', 'd'] and ['b', 'e', 'f'] into ['a', 'b', 'c', 'd', 'e', 'f']. This is helpful with
    quick replacing named entities with special tokens
    """
    result = [None] * (len(list1) + len(list2))
    result[::2] = list1
    result[1::2] = list2
    return result
