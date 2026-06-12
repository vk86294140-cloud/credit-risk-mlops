"""credit_risk: an end-to-end, reproducible ML pipeline for credit-default risk.

Modules:
    config    - paths and reproducibility settings
    data      - reproducible dataset generation / loading
    features  - scikit-learn preprocessing pipeline
    train     - train, evaluate, and version a model artifact
    model     - load a trained artifact and score requests
    schema    - pydantic request/response contracts
    api       - FastAPI scoring service
"""

__version__ = "0.1.0"
