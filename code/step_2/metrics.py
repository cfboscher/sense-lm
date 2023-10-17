def get_prec(pred, true):
    pred_words = str(pred).split()
    true_words = str(true).split()

    if len(pred_words) == 0:
        return 0
    return len([x for x in pred_words if x in true_words]) / len(pred_words)


def get_rec(pred, true):
    pred_words = str(pred).split()
    true_words = str(true).split()
    if len(true_words) == 0:
        return 0
    return len([x for x in true_words if x in pred_words]) / len(true_words)