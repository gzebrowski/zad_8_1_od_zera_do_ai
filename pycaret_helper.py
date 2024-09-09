from pathlib import Path
from typing import Optional

from pandas import DataFrame
from pycaret.regression import (compare_models, finalize_model, load_model,
                                save_model, setup)


def prepare_regression_model(
        df: DataFrame,
        model_name: str,
        target: str,
        session_id: Optional[int] = None,
        ignore_features: Optional[list[str]] = None,
        **kwargs,
):
    model_f_name = f'{model_name}.pkl'
    if not Path(model_f_name).exists():
        for kw_key, kw_val in [('ignore_features', ignore_features)]:
            if kw_val is not None:
                setup_kws = {kw_key: kw_val}
        setup(df, target=target, session_id=session_id or 123, **setup_kws, **kwargs)
        best_model = compare_models()
        final_model = finalize_model(best_model)
        save_model(final_model, model_name)
    else:
        final_model = load_model(model_name)
    return final_model
