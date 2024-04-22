def get_coefficient(df):
    true_label, false_label = 0, 0
    for i in df['label']:
        if i == 1:
            true_label += 1
        elif i == 0:
            false_label += 1
    return pow(true_label / false_label, 2)


def get_threshold_values(pred, label_test, df):
    pred_true, pred_false, num_true, num_false = [], [], [], []

    for i in range(len(pred)):
        if label_test[i] == 1:
            pred_true.append(pred[i])
        elif label_test[i] == 0:
            pred_false.append(pred[i])
    for i in range(len(pred_true)):
        num_true.append(i)
    for i in range(len(pred_false)):
        num_false.append(i)

    pred_true.sort()
    pred_false.sort()

    coefficient = get_coefficient(df)

    true_values, false_values = [], []
    for i in range(100):
        false_list = [x for x in pred_false if x > i / 100]
        true_list = [x for x in pred_true if x > i / 100]
        if len(true_list) / (len(true_list) + len(false_list) + 1) > 0.6 and len(true_list) > len(pred_true) / coefficient:
            true_values.append([len(true_list) / (len(true_list) + len(false_list) + 1), len(false_list), len(true_list), i / 100])

    for i in range(100):
        false_list = [x for x in pred_false if x < i / 100]
        true_list = [x for x in pred_true if x < i / 100]
        if len(false_list) / (len(true_list) + len(false_list) + 1) > 0.6 and len(false_list) > len(pred_false) / coefficient:
            false_values.append([len(false_list) / (len(true_list) + len(false_list) + 1), len(false_list), len(true_list), i / 100])

    k1, k2, value = 0.5, 0.5, 0

    for i in range(len(false_values)):
        for j in range(len(true_values)):
            acc = (false_values[i][0] + true_values[j][0]) / 2
            if acc > value:
                value = acc
                k1 = false_values[i][3]
                k2 = true_values[j][3]

    return k1, k2


