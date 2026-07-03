"""MolForge predictive + inverse-design pipeline.

Configure everything in ``molforge.predictive.target`` (the one config block), then run the
stages in order:

    python -m molforge.predictive.features   # 1. featurize the dataset
    python -m molforge.predictive.select     # 2. shortlist features (RFE-CV)
    python -m molforge.predictive.tune       # 3. (optional) Optuna hyper-parameters
    python -m molforge.predictive.train      # 4. fit & lock the predictive model
    python -m molforge.predictive.design     # 5. generate -> predict -> LLM/RAG -> ranked hits
"""
