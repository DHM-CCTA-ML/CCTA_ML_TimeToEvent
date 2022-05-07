#%%

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_array
from sklearn.model_selection._split import _RepeatedSplits

#%%
class StratifiedKFold_surv(StratifiedKFold):
    def split(self, X, y, groups=None):
        y=y['e']
        y = check_array(y, ensure_2d=False, dtype=None)
        return super().split(X, y, groups)

#%%
class RepeatedStratifiedKFold_surv(_RepeatedSplits):
    def __init__(self, *, n_splits, n_repeats, random_state=None):
        super().__init__(
            StratifiedKFold_surv, n_repeats=n_repeats, random_state=random_state,
            n_splits=n_splits)
            