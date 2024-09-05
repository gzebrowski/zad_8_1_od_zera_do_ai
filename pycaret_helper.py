from pathlib import Path

from pycaret.regression import (compare_models, finalize_model, load_model,
                                save_model, setup)


def prepare_regression_model(df, model_name, target, session_id=None):
    model_f_name = f'{model_name}.pkl'
    if not Path(model_f_name).exists():
        setup(df, target=target, session_id=session_id or 123)
        best_model = compare_models()
        final_model = finalize_model(best_model)
        save_model(final_model, model_name)
    else:
        final_model = load_model(model_name)
    return final_model
