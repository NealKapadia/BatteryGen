"""BatteryGen predictive + inverse-design pipeline.

Configure everything in ``batterygen.predictive.target`` (the one config block), then run the
stages in order:

    python -m batterygen.predictive.features   # 1. featurize the dataset
    python -m batterygen.predictive.select     # 2. shortlist features (RFE-CV)
    python -m batterygen.predictive.tune       # 3. (optional) Optuna hyper-parameters
    python -m batterygen.predictive.train      # 4. fit & lock the predictive model
    python -m batterygen.predictive.design     # 5. generate -> predict -> LLM/RAG -> ranked hits
"""
