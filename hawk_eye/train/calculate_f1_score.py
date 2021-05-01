def calculate_f1_score(
    true_positives, false_positives, false_negatives, beta: int
) -> int:

    """ This function finds the F1 score for a model.

    Examples::

        >>> calculate_f1_score(5, 2, 1, 2)
        .8064516
        >>> calculate_f1_score(10, 5, 5, 2)
        .666666

    """

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    f1_score = (1 + (beta ** 2)) * (
        (precision * recall) / (((beta ** 2) * precision) + recall)
    )

    return f1_score
