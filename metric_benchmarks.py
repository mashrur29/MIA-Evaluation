import numpy as np
from termcolor import colored
from privacy_risk_score import calculate_risk_score


def get_correctness(shadow_train_performance, target_train_performance, shadow_test_performance,
                    target_test_performance):
    shadow_train_o, shadow_train_y = shadow_train_performance
    shadow_test_o, shadow_test_y = shadow_test_performance
    target_train_o, target_train_y = target_train_performance
    target_test_o, target_test_y = target_test_performance

    train_corr = (np.argmax(target_train_o, axis=1) == target_train_y).astype(int)
    test_corr = (np.argmax(target_test_o, axis=1) == target_test_y).astype(int)

    train_acc = float(np.sum(train_corr) / len(train_corr))
    test_acc = float(np.sum(test_corr) / len(test_corr))
    mia_score = 0.5 * (train_acc + 1 - test_acc)

    print('Train acc {}, Test acc {}, MIA Correctness {}'.format(train_acc, test_acc, mia_score))


def get_threshold(train_val, test_val):
    # print(train_val, test_val)
    lst = np.concatenate((train_val, test_val))
    threshold = -1
    maxi = -1

    for i in lst:
        train_r = float(np.sum(train_val >= i) / float(len(train_val)))
        test_r = float(np.sum(test_val < i) / float(len(test_val)))
        acc = 0.5 * (train_r + test_r)
        if acc > maxi:
            maxi = acc
            threshold = i
    return threshold


def get_confidence(shadow_train_performance, target_train_performance, shadow_test_performance,
                   target_test_performance, num_classes=100):
    shadow_train_o, shadow_train_y = shadow_train_performance
    shadow_test_o, shadow_test_y = shadow_test_performance
    target_train_o, target_train_y = target_train_performance
    target_test_o, target_test_y = target_test_performance

    shadow_train_conf = np.array([shadow_train_o[i, shadow_train_y[i]] for i in range(len(shadow_train_y))])
    shadow_test_conf = np.array([shadow_test_o[i, shadow_test_y[i]] for i in range(len(shadow_test_y))])
    target_train_conf = np.array([target_train_o[i, target_train_y[i]] for i in range(len(target_train_y))])
    target_test_conf = np.array([target_test_o[i, target_test_y[i]] for i in range(len(target_test_y))])

    member = 0
    nonmember = 0

    for i in range(num_classes):
        threshold = get_threshold(shadow_train_conf[shadow_train_y == i], shadow_test_conf[shadow_test_y == i])
        member += np.sum(target_train_conf[target_train_y == i] >= threshold)
        nonmember += np.sum(target_test_conf[target_test_y == i] < threshold)

    mia_score = 0.5 * ((member / float(len(target_train_y))) + (nonmember / float(len(target_test_y))))

    print('MIA Confidence: {}'.format(mia_score))


def get_entropy_vals(p):
    log = lambda x: -np.log(np.maximum(x, 1e-18))
    ret = np.sum(np.multiply(p, log(p)), axis=1)
    return ret


def get_entropy(shadow_train_performance, target_train_performance, shadow_test_performance,
                target_test_performance, num_classes=100):
    shadow_train_o, shadow_train_y = shadow_train_performance
    shadow_test_o, shadow_test_y = shadow_test_performance
    target_train_o, target_train_y = target_train_performance
    target_test_o, target_test_y = target_test_performance

    shadow_train_conf = get_entropy_vals(shadow_train_o)
    shadow_test_conf = get_entropy_vals(shadow_test_o)
    target_train_conf = get_entropy_vals(target_train_o)
    target_test_conf = get_entropy_vals(target_test_o)

    member = 0
    nonmember = 0

    for i in range(num_classes):
        threshold = get_threshold(shadow_train_conf[shadow_train_y == i], shadow_test_conf[shadow_test_y == i])
        member += np.sum(target_train_conf[target_train_y == i] >= threshold)
        nonmember += np.sum(target_test_conf[target_test_y == i] < threshold)

    mia_score = 0.5 * (member / (len(target_train_y)) + nonmember / (len(target_test_y)))

    print('MIA Entropy: {}'.format(mia_score))


def get_mentropy_vals(train_o, train_y):
    log = lambda x: -np.log(np.maximum(x, 1e-18))
    p1 = log(train_o)
    p = 1 - log(train_o)
    rp = log(p)

    modified_probs = np.copy(train_o)

    modified_probs[range(train_y.size), train_y] = p[range(train_y.size), train_y]
    modified_log_probs = np.copy(rp)
    modified_log_probs[range(train_y.size), train_y] = p1[range(train_y.size), train_y]
    return np.sum(np.multiply(modified_probs, modified_log_probs), axis=1)


def get_modified_entropy(shadow_train_performance, target_train_performance, shadow_test_performance,
                         target_test_performance, num_classes=100):
    shadow_train_o, shadow_train_y = shadow_train_performance
    shadow_test_o, shadow_test_y = shadow_test_performance
    target_train_o, target_train_y = target_train_performance
    target_test_o, target_test_y = target_test_performance

    shadow_train_conf = get_mentropy_vals(shadow_train_o, shadow_train_y)
    shadow_test_conf = get_mentropy_vals(shadow_test_o, shadow_test_y)
    target_train_conf = get_mentropy_vals(target_train_o, target_train_y)
    target_test_conf = get_mentropy_vals(target_test_o, target_test_y)

    member = 0
    nonmember = 0

    for i in range(num_classes):
        threshold = get_threshold(shadow_train_conf[shadow_train_y == i], shadow_test_conf[shadow_test_y == i])
        member += np.sum(target_train_conf[target_train_y == i] >= threshold)
        nonmember += np.sum(target_test_conf[target_test_y == i] < threshold)

    mia_score = 0.5 * (member / (len(target_train_y)) + nonmember / (len(target_test_y)))

    print('MIA Entropy: {}'.format(mia_score))


def get_privacy_risk_score(shadow_train_performance, target_train_performance, shadow_test_performance,
                           target_test_performance, num_classes=100):
    shadow_train_o, shadow_train_y = shadow_train_performance
    shadow_test_o, shadow_test_y = shadow_test_performance
    target_train_o, target_train_y = target_train_performance
    target_test_o, target_test_y = target_test_performance

    shadow_train_conf = get_mentropy_vals(shadow_train_o, shadow_train_y)
    shadow_test_conf = get_mentropy_vals(shadow_test_o, shadow_test_y)
    target_train_conf = get_mentropy_vals(target_train_o, target_train_y)

    scores = calculate_risk_score(shadow_train_conf, shadow_test_conf, shadow_train_y, shadow_test_y, target_train_conf,
                                  target_train_y)

    print('Privacy Risk Score: {}'.format(np.mean(scores)))
