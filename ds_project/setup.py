from setuptools import setup, find_packages

setup(
    name="ds_project",
    version="1.0.0",
    description="Regression on 20-feature noisy dataset (Spark + MLflow + Databricks)",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.11",
        "scikit-learn>=1.3",
        "xgboost>=2.0",
        "lightgbm>=4.1",
        "catboost>=1.2",
    ],
    extras_require={
        "spark": ["pyspark>=3.5", "mlflow>=2.10"],
        "dev":   ["pytest>=7.4", "pytest-cov>=4.1"],
    },
)
