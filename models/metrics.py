import numpy as np

def calc_ccc(x, y):
    """
    Calculates Lin's Concordance Correlation Coefficient (CCC)
    ----------------------------------------------------------
    Parameters:
        x : First set of measurements.
        y : Second set of measurements.

    ----------------------------------------------------------
    Returns:
        CCC value

    ----------------------------------------------------------
    Examples:
    >>> x = np.linspace(0, 1, 20)
    >>> y = x + (np.random.rand(20) - 0.5) * 0.1
    >>> ccc(x, y)
    0.9946307958924353
    """

    x, y = np.array(x), np.array(y)
    assert len(x) == len(y), "Input arrays must have the same length."

    vx, cov_xy, _, vy = np.cov(x, y, bias=False).flat
    mx, my = x.mean(), y.mean()
    return 2 * cov_xy / (vx + vy + (mx - my) ** 2)


def calc_accuracy_at_std(gt, pred, std=1):
    assert len(gt) == len(
        pred
    ), "Ground truth and predictions must have the same length."
    # Compute the accuracy of predictions within one standard deviation
    gt = np.array(gt)
    pred = np.array(pred)
    return sum(np.abs(gt - pred) <= std) / len(gt)

def calc_aphr_away(gt, pred, away=1):
    bins = [0, 1_000_000, 3_000_000, 4_000_000, 7_000_000, 10_000_000]
    
    # convert gt, pred to labels by bins
    assert len(gt) == len(
        pred
    ), "Ground truth and predictions must have the same length."

    # use boxcox_bins
    gt_labels = np.digitize(gt, bins)
    pred_labels = np.digitize(pred, bins)
    return sum(np.abs(gt_labels - pred_labels) <= away) / len(gt)