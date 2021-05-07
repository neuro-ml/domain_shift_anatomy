import random
from copy import deepcopy

from dpipe.split import train_val_test_split, train_test_split


def one2all(df, val_size=2, seed=0xBadCafe):
    """Train on 1 domain, test on (n - 1) domains."""
    folds = sorted(df.fold.unique())
    split = []
    for f in folds:
        idx_b = df[df.fold == f].index.tolist()
        test_ids = df[df.fold != f].index.tolist()
        train_ids, val_ids = train_test_split(idx_b, test_size=val_size, random_state=seed)
        split.append([train_ids, val_ids, test_ids])
    return split


def single_cv(df, n_splits=3, val_size=2, seed=0xBadCafe):
    """Cross-validation inside every domain."""
    folds = sorted(df.fold.unique())
    split = []
    for f in folds:
        idx_b = df[df.fold == f].index.tolist()
        cv_b = train_val_test_split(idx_b, val_size=val_size, n_splits=n_splits, random_state=seed)
        for cv in cv_b:
            split.append(cv)
    return split


def one2one(df, val_size=2, n_add_ids=5, train_on_add_only=False, seed=0xBadCafe, train_on_source_only=False):
    random.seed(seed)
    folds = sorted(df.fold.unique())
    split = []
    for fb in folds:
        folds_o = set(folds) - {fb}
        ids_train, ids_val = train_test_split(df[df.fold == fb].index.tolist(), test_size=val_size, random_state=seed)
        if train_on_add_only:
            ids_train = []

        for fo in folds_o:
            ids_test, ids_train_add = train_test_split(df[df.fold == fo].index.tolist(), test_size=n_add_ids,
                                                       random_state=seed) if n_add_ids > 0 else \
                                          (df[df.fold == fo].index.tolist(), [])

            split.append([deepcopy(ids_train), random.sample(ids_test, val_size), ids_test]
                         if (not train_on_add_only) and train_on_source_only else
                         [deepcopy(ids_train) + ids_train_add, random.sample(ids_test, val_size), ids_test])
    return split
