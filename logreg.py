from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import PCA


def logreg_grid_search_cv(X, y, C_grid, n_comp_grid=None):
    """Perform a grid search + CV on logistic regression models. A PCA is plugged before the logreg
    if `n_com_grid` is not None.
    Args:
        - X : 2D np.array input data
        - y : 1D np.array labels
    Returns:
        - best_model : Pipeline which contains the best model.
    """
    # Build the pipe
    pipe = [
        ("normalizer", Normalizer()),
        ("logistic", LogisticRegression(max_iter=1000)),
    ]
    if n_comp_grid is not None:
        pipe = [("pca", PCA())] + pipe
    pipeline = Pipeline(steps=pipe)

    # Parameter grid
    param_grid = {"logistic__C": C_grid}
    if n_comp_grid is not None:
        param_grid["pca__n_components"] = n_comp_grid

    # Grid search
    search = GridSearchCV(pipeline, param_grid, n_jobs=-1)
    search.fit(X, y)

    return search


if __name__ == "__main__":
    from sklearn.datasets import load_digits
    import numpy as np

    X, y = load_digits(return_X_y=True)
    C_grid = np.logspace(-4, 4, 20)
    n_comp_grid = [5, 15, 30, 45, 64]

    search = logreg_grid_search_cv(X, y, C_grid, n_comp_grid)

    print(search.best_score_)
    print(search.best_estimator_)
