def val(tp, fn, fp, tn):
    from math import sqrt
    # --- Recall (senstivity) and Specificity
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)

    # --- Precision
    precision = tp / (tp + fp)

    # --- Accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # --- F-1 Score
    a = 1 / precision
    b = 1 / recall
    f1 = 2 / (a + b)

    # --- mcc
    x = (tp * tn) - (fp * fn)
    y = sqrt((tp + fp)*(tp + fn)*(tn + fp) * (tn + fn))
    mcc = x / y

    # --- Youdens J statistic
    j = (recall + specificity) - 1  # J = recall + specificity -1
    return recall, specificity, accuracy, mcc, f1, j