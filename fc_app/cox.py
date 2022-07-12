from matplotlib.figure import Figure
from lifelines.utils import inv_normal_cdf, check_nans_or_infs
import numpy as np
from flask import current_app
from .util import redis_set, redis_get
import pandas as pd
import time


def create_figure(params_, standard_errors_):
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis = plot(params_, standard_errors_, ax=axis)
    fig.tight_layout()
    return fig


def plot(params_, standard_errors_, ax=None, **errorbar_kwargs):
    """
    Produces a visual representation of the coefficients (i.e. log hazard ratios), including their standard errors and magnitudes.

    Parameters
    ----------
    columns : list, optional
        specify a subset of the columns to plot
    hazard_ratios: bool, optional
        by default, ``plot`` will present the log-hazard ratios (the coefficients). However, by turning this flag to True, the hazard ratios are presented instead.
    errorbar_kwargs:
        pass in additional plotting commands to matplotlib errorbar command

    Returns
    -------
    ax: matplotlib axis
        the matplotlib axis that be edited.

    """
    from matplotlib import pyplot as plt

    if ax is None:
        ax = plt.gca()

    alpha = 0.05

    errorbar_kwargs.setdefault("c", "k")
    errorbar_kwargs.setdefault("fmt", "s")
    errorbar_kwargs.setdefault("markerfacecolor", "white")
    errorbar_kwargs.setdefault("markeredgewidth", 1.25)
    errorbar_kwargs.setdefault("elinewidth", 1.25)
    errorbar_kwargs.setdefault("capsize", 3)

    z = inv_normal_cdf(1 - alpha / 2)
    user_supplied_columns = True

    user_supplied_columns = False
    columns = params_.index

    yaxis_locations = list(range(len(columns)))
    log_hazards = params_.loc[columns].values.copy()

    order = list(range(len(columns) - 1, -1, -1)) if user_supplied_columns else np.argsort(log_hazards)

    symmetric_errors = z * standard_errors_[columns].values
    ax.errorbar(log_hazards[order], yaxis_locations, xerr=symmetric_errors[order], **errorbar_kwargs)
    ax.set_xlabel("log(HR) (%g%% CI)" % ((1 - alpha) * 100))

    best_ylim = ax.get_ylim()
    ax.vlines(0, -2, len(columns) + 1, linestyles="dashed", linewidths=1, alpha=0.65)
    ax.set_ylim(best_ylim)

    tick_labels = [columns[i] for i in order]

    ax.set_yticks(yaxis_locations)
    ax.set_yticklabels(tick_labels)
    ax.set_title("Visual representation of the coefficients of the federated model")

    return ax


def get_local_covariates():
    """
    get the name of the covariates of the local dataframe and send to the master
    """
    current_app.logger.info('[API] run get_local_covariates')

    X = redis_get('X')
    duration_col = redis_get('duration_col')
    event_col = redis_get('event_col')
    covariates = X.columns.values
    index = np.argwhere(covariates == duration_col)
    covariates = np.delete(covariates, index)
    index = np.argwhere(covariates == event_col)
    covariates = np.delete(covariates, index)
    redis_set('covariates', pd.Series(covariates))
    # current_app.logger.info(f'[API] covariates: {covariates}')
    if redis_get('master'):
        global_cov = redis_get('global_cov')
        global_cov.append(pd.Series(covariates))
        redis_set('global_cov', global_cov)
        global_covariates_intersection()

    else:
        redis_set('available', True)


def preprocessing():
    """
    Preprocesses the dataframe of the clients. Save X, T and E of the dataset.
    """
    current_app.logger.info('[API] run preprocessing')
    localdata = redis_get('data')
    duration_col = redis_get('duration_col')
    event_col = redis_get('event_col')

    if localdata is None:
        current_app.logger.info('[API] Data is None')
        return None
    elif duration_col is None:
        current_app.logger.info('[API] Duration column is None')
        return None
    elif event_col is None:
        current_app.logger.info('[API] Event column is None')
        return None
    else:
        sort_by = [duration_col, event_col] if event_col else [duration_col]
        data = localdata.sort_values(by=sort_by)
        T = data.pop(duration_col)
        E = data.pop(event_col)

        X = data.astype(float)
        T = T.astype(float)

        check_nans_or_infs(E)
        E = E.astype(bool)

        redis_set('X', X)
        redis_set('T', T)
        redis_set('E', E)

        current_app.logger.info('[API] Preprocessing is done (X,T,E saved)')

        redis_set('step', 'find_intersection')

        get_local_covariates()


def global_covariates_intersection():
    current_app.logger.info('[API] run global_covariates_intersection')
    global_cov = redis_get('global_cov')
    nr_clients = redis_get('nr_clients')
    np.set_printoptions(precision=30)
    if len(global_cov) == nr_clients:
        current_app.logger.info('[API] The data of all clients has arrived')

        start_time = time.time()  # start of runtime check
        redis_set('start_time', start_time)
        redis_set('global_time', start_time)

        intersect_covariates = pd.Series()

        for cov in global_cov:
            if intersect_covariates.empty:
                intersect_covariates = cov
            else:
                intersect_covariates = pd.Series(np.intersect1d(intersect_covariates, cov))

        # current_app.logger.info( f'[API] intersection of covariates: {intersect_covariates}' )
        redis_set('intersected_covariates', intersect_covariates)
        redis_set('available', True)  # send the intersected covariates to slaves

    else:
        current_app.logger.info('[API] Not the data of all clients has been send to the master')
