import os
from collections import defaultdict
from typing import Sequence, Callable

import numpy as np
from tqdm import tqdm
from skimage.exposure import match_histograms

from dpipe.im.metrics import dice_score
from dpipe.commands import load_from_folder
from dpipe.io import save_json
from dpipe.itertools import zip_equal


def aggregate_metric_probably_with_ids(xs, ys, ids, metric, aggregate_fn=np.mean):
    """Aggregate a `metric` computed on pairs from `xs` and `ys`"""
    try:
        return aggregate_fn([metric(x, y, i) for x, y, i in zip_equal(xs, ys, ids)])
    except TypeError:
        return aggregate_fn([metric(x, y) for x, y in zip_equal(xs, ys)])


def evaluate_with_ids(y_true: Sequence, y_pred: Sequence, ids: Sequence[str], metrics: dict) -> dict:
    return {name: metric(y_true, y_pred, ids) for name, metric in metrics.items()}


def compute_metrics_probably_with_ids(predict: Callable, load_x: Callable, load_y: Callable, ids: Sequence[str],
                                      metrics: dict):
    return evaluate_with_ids(list(map(load_y, ids)), [predict(load_x(i)) for i in ids], ids, metrics)


def compute_metrics_probably_with_ids_spottune(predict, load_x, load_y, ids, metrics, architecture_main):
    architecture_main.val_flag = True  # counting validation stats ON
    old_metrics = evaluate_with_ids(list(map(load_y, ids)), [predict(load_x(i)) for i in ids], ids, metrics)
    spottune_stats_val = architecture_main.get_val_stats()
    spottune_stats_train = architecture_main.get_train_stats()
    architecture_main.val_flag = False  # counting validation stats OFF
    return {**old_metrics, **spottune_stats_val, **spottune_stats_train}


def evaluate_individual_metrics_probably_with_ids(load_y_true, metrics: dict, predictions_path, results_path,
                                                  exist_ok=False):
    assert len(metrics) > 0, 'No metric provided'
    os.makedirs(results_path, exist_ok=exist_ok)

    results = defaultdict(dict)
    for identifier, prediction in tqdm(load_from_folder(predictions_path)):
        target = load_y_true(identifier)

        for metric_name, metric in metrics.items():
            try:
                results[metric_name][identifier] = metric(target, prediction, identifier)
            except TypeError:
                results[metric_name][identifier] = metric(target, prediction)

    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)


def multichannel_pred_metric_reduction(true3d, pred4d, metric=dice_score, pred_reduction=None, score_reduction=np.mean):
    if pred_reduction is not None:
        pred3d = pred_reduction(pred4d)
        scores = [metric(true3d, pred3d)]
    else:
        scores = [metric(true3d, p) for p in pred4d]

    return score_reduction(scores)


def evaluate_individual_metrics_probably_with_ids_no_pred(load_y, load_x, predict, metrics: dict, test_ids,
                                                          results_path, exist_ok=False):
    assert len(metrics) > 0, 'No metric provided'
    os.makedirs(results_path, exist_ok=exist_ok)

    results = defaultdict(dict)
    for _id in tqdm(test_ids):
        target = load_y(_id)
        prediction = predict(load_x(_id))

        for metric_name, metric in metrics.items():
            try:
                results[metric_name][_id] = metric(target, prediction, _id)
            except TypeError:
                results[metric_name][_id] = metric(target, prediction)

    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)


def evaluate_individual_metrics_with_hm(load_y, load_x, predict, metrics: dict, test_ids, train_ids,
                                        results_path, exist_ok=False):
    assert len(metrics) > 0, 'No metric provided'
    os.makedirs(results_path, exist_ok=exist_ok)

    all_train_img = np.float16([])
    for _id in train_ids:
        all_train_img = np.concatenate((all_train_img, np.float16(load_x(_id)).ravel()))

    results = defaultdict(dict)
    for _id in tqdm(test_ids):
        target = load_y(_id)
        image = load_x(_id)
        prediction = predict(np.float32(np.reshape(match_histograms(image.ravel(), all_train_img), image.shape)))

        for metric_name, metric in metrics.items():
            try:
                results[metric_name][_id] = metric(target, prediction, _id)
            except TypeError:
                results[metric_name][_id] = metric(target, prediction)

    for metric_name, result in results.items():
        save_json(result, os.path.join(results_path, metric_name + '.json'), indent=0)
