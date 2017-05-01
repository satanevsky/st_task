"""Ranking model evaluation."""
import constants
import collections
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import base as sklearn_base


EvaluationResults = collections.namedtuple('EvaluationResults', ['scores', 'mean', 'std'])


_SCORE = 'score'
_CLICK = 'click'


def _to_input_df(df, product_category, product_cpc):
    """Converts (product_id, user_clicked) pairs DataFrame into default input format.

    Args:
        df: DataFrame to convert.
        product_category: Series of product category for product id.
        product_cpc: Series or product CPC for product id.
    Returns:
        Converted DataFrame.
    """
    df = df.groupby(constants.PRODUCT_ID).agg({
        _CLICK: {
            constants.VIEWS: 'count',
            constants.CLICKS: 'sum'}
    }).reset_index()

    df[constants.VIEWS] = df[_CLICK][constants.VIEWS]
    df[constants.CLICKS] = df[_CLICK][constants.CLICKS]
    del df[_CLICK]
    df[constants.CATEGORY_ID] = df.product_id.map(product_category)
    df[constants.CPC] = df.product_id.map(product_cpc)
    return df


def _get_dcg(item_scores):
    """Calculates DCG function for provided scores.

    Args:
        item_scores: Items scores in order they've been ranked by algorithm.
    Returns:
        DCG
    """
    return (item_scores / np.log2(np.arange(2, len(item_scores) + 2))).sum()


def _get_ndcg(item_scores):
    """Gets nDCG for provided scores.

    Args:
        item_scores: Items scores in order they've been ranked by algorithm.
    Returns:
        nDCG.
    """
    return _get_dcg(item_scores) / _get_dcg(np.sort(item_scores)[::-1])


def _get_metric(test_views_df, scores):
    """Calculates metric value for test items DataFrame and evaluated scores.

    Args:
        test_views_df: Test DataFrame.
        scores: Ranking algorithm scores.
    Returns:
        nDCG value
    """
    test_views_df = test_views_df.copy()
    test_views_df[_SCORE] = scores
    test_views_df = test_views_df.sort_values(by=_SCORE, ascending=False)
    return _get_ndcg(test_views_df.clicks / test_views_df.views * test_views_df.cpc)


def _get_views_df(dataset):
    """Gets DataFrame containing individual product views.

    Args:
        dataset: Source data containing views and clicked counters.
    Returns:
        DataFrame fro individual views.
    """
    views_df_items = {
        constants.PRODUCT_ID: list(),
        _CLICK: list(),
    }
    for product_data in dataset.itertuples():
        views_df_items[constants.PRODUCT_ID].extend([product_data.product_id] * product_data.views)
        views_df_items[_CLICK].extend([True] * product_data.clicks)
        views_df_items[_CLICK].extend([False] * (product_data.views - product_data.clicks))

    return pd.DataFrame.from_dict(views_df_items)


def _calculate_folds_scores(model, views_df, product_category, product_cpc, n_splits, random_state):
    """Calculates metric function for all splits.

    Applies data aggregation to put it in source format.

    Args:
        model: Model to evaluate.
        views_df: DataFrame containing individual products views.
        product_category: Series of product category for product id.
        product_cpc: Series or product CPC for product id.
        n_splits: Splits count.
        random_state: Random state to use while shuffling.
    Returns:
        Numpy array with score for each split.
    """
    scores = list()

    for train_index, test_index in model_selection.KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
    ).split(views_df):
        train_df = _to_input_df(views_df.ix[train_index], product_category=product_category, product_cpc=product_cpc)
        test_df = _to_input_df(views_df.ix[test_index], product_category=product_category, product_cpc=product_cpc)

        product_scores = sklearn_base.clone(model).fit(train_df).predict(
            test_df[[constants.PRODUCT_ID, constants.CATEGORY_ID, constants.CPC]])

        scores.append(_get_metric(test_df, product_scores))

    return np.array(scores)


def evaluate(model, dataset, n_splits=10, random_state=42):
    """Evaluates model on dataset doing cross-validation.

    Args:
        model: Ranking model.
        dataset: DataFrame with products information.
        n_splits: Count of folds in cross-validation.
        random_state: Random state to use when shuffling data.
    Returns:
        EvaluationResults with evaluation results.
    """
    product_category = dataset.set_index(constants.PRODUCT_ID).category_id
    product_cpc = dataset.set_index(constants.PRODUCT_ID).cpc

    views_df = _get_views_df(dataset)

    scores = _calculate_folds_scores(
        model=model,
        views_df=views_df,
        product_category=product_category,
        product_cpc=product_cpc,
        n_splits=n_splits,
        random_state=random_state,
    )

    return EvaluationResults(scores=scores, mean=np.mean(scores), std=np.std(scores))
