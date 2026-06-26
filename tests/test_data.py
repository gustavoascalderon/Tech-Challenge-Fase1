"""
Testes automatizados — Etapa 1.
Inclui: smoke test, schema test e testes de dados.
Valores baseados nos resultados reais da EDA (01_EDA.ipynb).
"""
import logging
import pandas as pd
import numpy as np
import pytest

logger = logging.getLogger('churn_tests')

DATA_PATH   = 'data/processed/telco_churn_clean.csv'
RAW_PATH    = 'data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv'

# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope='module')
def df():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        pytest.skip('Dataset não encontrado — execute 01_EDA.ipynb primeiro')


@pytest.fixture(scope='module')
def X_sample(df):
    """Amostra de 5 linhas para smoke tests."""
    TARGET    = 'Churn'
    DROP_COLS = ['customerID', 'TotalCharges']
    return df.drop(columns=[TARGET] + DROP_COLS).head(5)


# ─────────────────────────────────────────────────────────────────────────────
# 1. SMOKE TESTS — o pipeline carrega e faz predição sem explodir
# ─────────────────────────────────────────────────────────────────────────────
def test_smoke_imports():
    """Smoke test: imports críticos do projeto funcionam."""
    import sklearn
    import lightgbm
    import mlflow
    import fairlearn
    logger.info('Smoke test imports OK — sklearn %s | lgbm %s',
                sklearn.__version__, lightgbm.__version__)


def test_smoke_pipeline_fit(X_sample, df):
    """Smoke test: pipeline básico consegue fit + predict sem erros."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.dummy import DummyClassifier

    TARGET    = 'Churn'
    DROP_COLS = ['customerID', 'TotalCharges']
    X = df.drop(columns=[TARGET] + DROP_COLS)
    y = df[TARGET]

    num_features = ['tenure', 'MonthlyCharges', 'SeniorCitizen']
    cat_features = [c for c in X.columns if c not in num_features]

    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')),
                          ('sc',  StandardScaler())]), num_features),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                          ('ohe', OneHotEncoder(handle_unknown='ignore',
                                                sparse_output=False))]), cat_features),
    ])

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier',   DummyClassifier(strategy='most_frequent', random_state=42)),
    ])

    pipe.fit(X.head(100), y.head(100))
    preds = pipe.predict(X.head(10))

    assert len(preds) == 10, 'Deve retornar 10 predições'
    assert set(preds).issubset({0, 1}), 'Predições devem ser 0 ou 1'
    logger.info('Smoke test pipeline OK')


def test_smoke_mlflow_connection():
    """Smoke test: MLflow consegue criar experimento localmente."""
    import mlflow
    mlflow.set_tracking_uri('file:./mlruns_test')
    exp = mlflow.set_experiment('test_smoke')
    assert exp is not None
    logger.info('Smoke test MLflow OK')


# ─────────────────────────────────────────────────────────────────────────────
# 2. SCHEMA TESTS — estrutura e tipos dos dados
# ─────────────────────────────────────────────────────────────────────────────
SCHEMA_ESPERADO = {
    'customerID'      : 'object',
    'gender'          : 'object',
    'SeniorCitizen'   : ['int64', 'int32', 'float64'],
    'Partner'         : 'object',
    'Dependents'      : 'object',
    'tenure'          : ['int64', 'float64'],
    'PhoneService'    : 'object',
    'MonthlyCharges'  : 'float64',
    'TotalCharges'    : 'float64',
    'Churn'           : ['int64', 'int32', 'float64'],
}

def test_schema_colunas_obrigatorias(df):
    """Schema: todas as colunas obrigatórias presentes."""
    faltando = [c for c in SCHEMA_ESPERADO if c not in df.columns]
    assert not faltando, f'Colunas ausentes: {faltando}'


def test_schema_tipos(df):
    """Schema: tipos de dados corretos após limpeza da EDA."""
    for col, dtype_esperado in SCHEMA_ESPERADO.items():
        if col not in df.columns:
            continue
        dtype_real = str(df[col].dtype)
        if isinstance(dtype_esperado, list):
            assert dtype_real in dtype_esperado, (
                f'{col}: dtype {dtype_real!r} não está em {dtype_esperado}')
        else:
            assert dtype_real == dtype_esperado, (
                f'{col}: dtype {dtype_real!r} != {dtype_esperado!r}')


def test_schema_target_binario(df):
    """Schema: target Churn é binário (0 ou 1)."""
    valores = set(df['Churn'].dropna().unique())
    assert valores.issubset({0, 1}), f'Target deve ser {{0,1}}, encontrado: {valores}'


def test_schema_sem_duplicatas(df):
    """Schema: zero registros duplicados."""
    n = df.duplicated().sum()
    assert n == 0, f'{n} duplicatas encontradas'


def test_schema_shape(df):
    """Schema: dataset tem o shape esperado (7043 × 21)."""
    assert df.shape == (7043, 21), f'Shape inesperado: {df.shape}'


def test_schema_churn_rate(df):
    """Schema: taxa de churn entre 24-29% conforme EDA real."""
    rate = df['Churn'].mean()
    assert 0.24 <= rate <= 0.29, f'Churn rate fora da faixa: {rate:.3f}'


def test_schema_nulos_aceitaveis(df):
    """Schema: nulos apenas em TotalCharges (≤11), zero nas demais."""
    nulos = df.isnull().sum()
    cols_problemas = nulos[(nulos > 0) & (nulos.index != 'TotalCharges')]
    assert len(cols_problemas) == 0, (
        f'Nulos inesperados em: {cols_problemas.to_dict()}')
    if 'TotalCharges' in nulos.index:
        assert nulos['TotalCharges'] <= 11, (
            f'TotalCharges tem {nulos["TotalCharges"]} nulos (esperado ≤11)')


# ─────────────────────────────────────────────────────────────────────────────
# 3. API TESTS — placeholder (implementação completa na Etapa 3)
# ─────────────────────────────────────────────────────────────────────────────
def test_api_placeholder_health():
    """
    API test placeholder — /health endpoint.
    Implementação completa na Etapa 3 com FastAPI + httpx.

    Quando implementado, este teste deverá:
        response = client.get('/health')
        assert response.status_code == 200
        assert response.json()['status'] == 'ok'
    """
    pytest.skip('API ainda não implementada — será feita na Etapa 3')


def test_api_placeholder_predict():
    """
    API test placeholder — /predict endpoint.
    Implementação completa na Etapa 3 com FastAPI + httpx.

    Quando implementado, este teste deverá:
        payload = {'tenure': 12, 'MonthlyCharges': 65.0, ...}
        response = client.post('/predict', json=payload)
        assert response.status_code == 200
        assert 'churn_probability' in response.json()
        assert 0 <= response.json()['churn_probability'] <= 1
    """
    pytest.skip('API ainda não implementada — será feita na Etapa 3')


def test_api_placeholder_schema_validation():
    """
    API test placeholder — validação de schema via Pydantic.
    Implementação completa na Etapa 3.

    Quando implementado:
        response = client.post('/predict', json={'campo_invalido': 'x'})
        assert response.status_code == 422
    """
    pytest.skip('API ainda não implementada — será feita na Etapa 3')
