Vamos fazer a analise de de redes neurais, onde posso selecionar varios X e 1 Y, sem interações

Estava pensando em um modulo exclusivo de machine learning para redes neurais e arvore de decisão, onde eu conseguisse, analisar por exemplo 1 arquivo hoje, fazer a analise, e guardar a analise, e outro dia eu analisar outro arquivo e usar essa rede para analisar esses dados.

Pensando em dados de histórico sabe, tenho centenas de linhas que foi lido hoje por uma maquina para ver defeitos em peças, para esse modelo aprender para ser usado em analises futuras sabe.
OU você acha válido, se for um histórico, pegar tudo em um unico arquivo e fazer somente 1 rede neural

hoje está dessa forma, mas pensei nessa possibilidade de ter uma rede salva e usar ela em futuras analises. eu teria dois modelos holdout e kfold

meu back end por enquanto 

from collections import OrderedDict, defaultdict
import re
from typing import Any, List
import numpy as np
import pandas as pd
from sklearn.calibration import label_binarize
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    confusion_matrix,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score
)
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelBinarizer
from sklearn.neural_network import MLPClassifier, MLPRegressor

def transform_columns(X: pd.DataFrame, categorical_cols: List[str], numeric_cols: List[str]):
    transformers = []

    if len(categorical_cols) >0:
        label_enc_cols = [col for col in categorical_cols]

        if label_enc_cols:
            transformers.append(('onehot', OneHotEncoder(), label_enc_cols)) 
    
    return transformers

# Calcula métricas
def calculate_metrics(
    df: pd.DataFrame,
    X_train, X_test,
    y_train, y_test,
    model,
    y_pred_train,
    y_pred_test,
    models: dict[str, dict]
    ):
    if df.iloc[:, -1].dtype == 'object':
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        train_precision = precision_score(y_train, y_pred_train, average='weighted')
        test_precision = precision_score(y_test, y_pred_test, average='weighted')
        train_recall = recall_score(y_train, y_pred_train, average='weighted')
        test_recall = recall_score(y_test, y_pred_test, average='weighted')
        train_f1 = f1_score(y_train, y_pred_train, average='weighted')
        test_f1 = f1_score(y_test, y_pred_test, average='weighted')
        confusion_matrix_test = confusion_matrix(y_test, y_pred_test)
        confusion_matrix_train = confusion_matrix(y_train, y_pred_train)

        unique_classes = np.unique(y_test)
        if len(unique_classes) == 2:
            pos_label = unique_classes[0]
            train_roc_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
            test_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            fpr_train, tpr_train, _ = roc_curve(y_train, model.predict_proba(X_train)[:, 1], pos_label=pos_label)
            fpr_test, tpr_test, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1], pos_label=pos_label)

            models["training"]["rocCurve"] = generate_roc_curve(fpr_train, tpr_train)
            models["validation"]["rocCurve"] = generate_roc_curve(fpr_test, tpr_test)
        else:
            train_roc_auc = roc_auc_score(y_train, model.predict_proba(X_train), multi_class='ovr')
            test_roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')
            fpr_train, tpr_train, _ = roc_curve(y_train, model.predict_proba(X_train)[:, 1], pos_label=model.classes_[1])
            fpr_test, tpr_test, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1], pos_label=model.classes_[1])

            models["training"]["rocCurve"] = generate_roc_curve(fpr_train, tpr_train)
            models["validation"]["rocCurve"] = generate_roc_curve(fpr_test, tpr_test)

        models["training"]["accuracy"] = np.round(train_accuracy, 5)
        models["training"]["precision"] = np.round(train_precision, 5)
        models["training"]["recall"] = np.round(train_recall, 5)
        models["training"]["fScore"] = np.round(train_f1, 5)
        models["training"]["rocAuc"] = np.round(train_roc_auc, 5)
        models["training"]["mean"] = 0
        models["training"]["stdDev"] = 0
        models["validation"]["accuracy"] = np.round(test_accuracy, 5)
        models["validation"]["precision"] = np.round(test_precision, 5)
        models["validation"]["recall"] = np.round(test_recall, 5)
        models["validation"]["fScore"] = np.round(test_f1, 5)
        models["validation"]["rocAuc"] = np.round(test_roc_auc, 5)

        # Adiciona nomes das classes à matriz de confusão
        models["validation"]["confusionMatrix"] = {
            class_name: confusion_matrix_test.tolist()[i]
            for i, class_name in enumerate(unique_classes)
        }
        models["training"]["confusionMatrix"] = {
            class_name: confusion_matrix_train.tolist()[i]
            for i, class_name in enumerate(unique_classes)
        }

        models["validation"]["mean"] = 0
        models["validation"]["stdDev"] = 0

    else:
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)

        models["training"]["meanSquaredError"] = np.round(train_mse, 5)
        models["training"]["r2"] = np.round(train_r2, 5)
        models["training"]["mean"] = np.round(np.mean(y_pred_train), 5)
        models["training"]["stdDev"] = np.round(np.std(y_pred_train), 5)
        models["validation"]["meanSquaredError"] = np.round(test_mse, 5)
        models["validation"]["r2"] = np.round(test_r2, 5)
        models["validation"]["mean"] = np.round(np.mean(y_pred_test), 5)
        models["validation"]["stdDev"] = np.round(np.std(y_pred_test), 5)
        models["training"]["rocCurve"] = []
        models["validation"]["rocCurve"] = []

# Cria Paths das redes neurais
def create_paths(model: MLPClassifier | MLPRegressor):
    nodes = []
    edges = []

    layer_sizes = [model.hidden_layer_sizes] if isinstance(model.hidden_layer_sizes, int) else model.hidden_layer_sizes
    layer_sizes = [model.n_features_in_] + list(layer_sizes) + [model.n_outputs_]
    n_layers = len(layer_sizes)

    v_spacing = 1. / (max(layer_sizes) + 1)
    h_spacing = 1. / (n_layers + 1)

    # Nodes
    for i, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing * (layer_size - 1) / 2 + v_spacing / 2
        for j in range(layer_size):
            node = {
                'id': f'layer{i}_node{j}',
                'type': None,
                'data': {
                    'label': f"Layer {i}"
                },
                'position': (i * h_spacing, layer_top - j * v_spacing)
            }
            nodes.append(node)

    # Edges
    for i, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        for j in range(layer_size_a):
            for k in range(layer_size_b):
                edge = {
                    'id': f'layer{i}_node{j}',
                    'source': f'layer{i}_node{j}',
                    'target': f'layer{i+1}_node{k}',
                    'type': "straight",
                    'weight': model.coefs_[i][j, k]
                }
                edges.append(edge)

    return nodes, edges

# Gera colunas importantes
def create_feature_importance(transformed_feature_names, num_features: int, X_test, y_test, model):
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=num_features * 20, random_state=0, n_jobs=-1)
    feature_importance = {name: np.round(score, 5) for name, score in zip(transformed_feature_names, perm_importance.importances_mean)}

    # quero adicionar os coeficientes de cada coluna também além do importance best_model.coefs_

    # Adiciona intercept da ultima camada nas features importantes
    if hasattr(model, 'intercepts_') and model.intercepts_:
        feature_importance['Intercept'] = np.round(np.mean(model.intercepts_[-1]), 5)

    # ordena feature importantes
    feature_importance_sorted = OrderedDict(
            sorted(feature_importance.items(), key=lambda item: item[0] != 'Intercept')
    )
    return feature_importance_sorted

# Gera curva ROC
def generate_roc_curve(fpr, tpr):
    return [{"x": round(x, 3), "y": round(y, 3)} for x, y in zip(fpr, tpr)]

# Calcula média das métricas para o K-Fold
def calculate_average_metrics(metrics:dict[str, Any]):
    averaged_metrics = {}
    for k, v in metrics.items():
        if k == "confusionMatrices":
            if isinstance(v, list) and len(v) > 0:
                averaged_metrics[k] = np.round(np.mean(v), 5)
            else:
                averaged_metrics[k] = v
        elif k == "rocCurve":
            averaged_metrics[k] = v
        else:
            if isinstance(v, list) and len(v) >0:
                averaged_metrics[k] = np.round(np.mean(v), 5)
            else:
                averaged_metrics[k] = v

    return averaged_metrics

# Calcula métricas do K-Fold
def calculate_kfold_metrics(is_categoric: bool, models: dict[str, dict[str, list]], X_train, X_test, y_train, y_test, model, y_train_pred, y_test_pred):
    if is_categoric:
        unique_classes = np.unique(y_test)
        if len(unique_classes) == 2:
            models["training"]["rocAuc"].append(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
            fpr_train, tpr_train, _ = roc_curve(y_train, model.predict_proba(X_train)[:, 1], pos_label=unique_classes[0])
            models["training"]["rocCurve"] = generate_roc_curve(fpr_train, tpr_train)

            models["validation"]["rocAuc"].append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
            fpr_test, tpr_test, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1], pos_label=unique_classes[0])
            models["validation"]["rocCurve"] = generate_roc_curve(fpr_test, tpr_test)

        else:
            models["training"]["rocAuc"].append(roc_auc_score(y_train, model.predict_proba(X_train), multi_class="ovr"))
            models["validation"]["rocAuc"].append(roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr"))

            y_test_bin = label_binarize(y_test, classes=model.classes_)
            y_train_bin = label_binarize(y_train, classes=model.classes_)

            fpr_test = {}
            tpr_test = {}

            fpr_train = {}
            tpr_train = {}

                    # Calcula a curva ROC para cada classe
            for i in range(model.classes_.shape[0]):
                fpr_train[i], tpr_train[i], _ = roc_curve(y_train_bin[:, i], model.predict_proba(X_train)[:, i])
                fpr_test[i], tpr_test[i], _ = roc_curve(y_test_bin[:, i], model.predict_proba(X_test)[:, i])

            models["training"]["rocCurve"] = generate_roc_curve(fpr_train, tpr_train)
            models["validation"]["rocCurve"] = generate_roc_curve(fpr_test, tpr_test)

        models["training"]["accuracy"].append(accuracy_score(y_train, y_train_pred))
        models["training"]["precision"].append(precision_score(y_train, y_train_pred, average='weighted'))
        models["training"]["recall"].append(recall_score(y_train, y_train_pred, average='weighted'))
        models["training"]["fScore"].append(f1_score(y_train, y_train_pred, average='weighted'))

        models["validation"]["accuracy"].append(accuracy_score(y_test, y_test_pred))
        models["validation"]["precision"].append(precision_score(y_test, y_test_pred, average='weighted'))
        models["validation"]["recall"].append(recall_score(y_test, y_test_pred, average='weighted'))
        models["validation"]["fScore"].append(f1_score(y_test, y_test_pred, average='weighted'))

        models["training"]["mean"].append(0)
        models["training"]["stdDev"].append(0)
        models["validation"]["mean"].append(0)
        models["validation"]["stdDev"].append(0)

        confusion_matrix_test = confusion_matrix(y_test, y_test_pred)
        confusion_matrix_train = confusion_matrix(y_train, y_train_pred)

        # Adiciona nomes das classes à matriz de confusão
        models["validation"]["confusionMatrix"] = {
            class_name: confusion_matrix_test.tolist()[i]
            for i, class_name in enumerate(unique_classes)
        }
        models["training"]["confusionMatrix"] = {
            class_name: confusion_matrix_train.tolist()[i]
            for i, class_name in enumerate(unique_classes)
        }

    else:
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mean = np.mean(y_train_pred)
        train_std = np.std(y_train_pred)

        test_mse = mean_squared_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        test_mean = np.mean(y_test_pred)
        test_std = np.std(y_test_pred)

        models["training"]["meanSquaredError"].append(np.round(train_mse, 5))
        models["training"]["r2"].append(np.round(train_r2, 5))
        models["training"]["mean"].append(np.round(train_mean, 5))
        models["training"]["stdDev"].append(np.round(train_std, 5))

        models["validation"]["meanSquaredError"].append(np.round(test_mse, 5))
        models["validation"]["r2"].append(np.round(test_r2, 5))
        models["validation"]["mean"].append(np.round(test_mean, 5))
        models["validation"]["stdDev"].append(np.round(test_std, 5))

# Gera curva de aprendizado
def generate_learning_curve(model, X: pd.DataFrame, y: pd.Series):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    learning_curve_data = {
        "trainSizes": train_sizes.tolist(),
        "trainScoresMean": train_scores_mean.tolist(),
        "testScoresMean": test_scores_mean.tolist()
    }

    return learning_curve_data

def generate_forward_pass_equation(model, X_transformed: pd.DataFrame, categorical_cols: list, activation: str):
    """
    Gera as equações do forward pass da rede neural e retorna um dicionário.
    As chaves são os nomes dos neurônios e os valores são as equações.
    """
    feature_names = X_transformed.columns  # Usar os nomes reais das colunas transformadas

    coefs = model.coefs_  # Pesos das camadas
    intercepts = model.intercepts_  # Bias das camadas

    equations_dict = {}  # Dicionário que armazenará as equações

    # Mapeamento de colunas categóricas para suas colunas One-Hot
    categorical_mapping = {
        col: [onehot_col for onehot_col in feature_names if onehot_col.startswith(f"onehot__{col}_")]
        for col in categorical_cols
    }

    # Lista de colunas de entrada ajustadas
    input_symbols = list(feature_names)  # Começa com as colunas já transformadas

    for layer_idx, (coef_layer, intercept_layer) in enumerate(zip(coefs, intercepts)):
        output_symbols = []  # Armazena os neurônios da camada atual

        for neuron_idx in range(coef_layer.shape[1]):
            terms = []

            for feature_idx, feature_name in enumerate(input_symbols):
                weight = coef_layer[feature_idx, neuron_idx]

                # Se a coluna for uma das categóricas One-Hot, reescreve o nome da equação
                for original_col, onehot_cols in categorical_mapping.items():
                    if feature_name in onehot_cols:
                        category_label = feature_name.replace(f"onehot__{original_col}_", "")
                        feature_name = f"{original_col}[{category_label}]"

                terms.append(f"({weight:.4f} * {feature_name})")

            # Adiciona o bias (intercepto)
            equation = " + ".join(terms) + f" + {intercept_layer[neuron_idx]:.4f}"
            equation = equation.replace("remainder__", "") 
            
            # Aplicação da função de ativação
            if activation == "relu":
                equation = f"max(0, {equation})"
            elif activation == "logistic":
                equation = f"(1 / (1 + exp({equation})))"
            elif activation == "identity":
                equation = f"{equation}" 
            else:
                equation = f"tanh(0.5 * ({equation}))"

            # Nome do neurônio na camada
            neuron_name = f"h{layer_idx+1}_{neuron_idx+1}"

            # Armazena a equação
            equations_dict[neuron_name] = equation
            output_symbols.append(neuron_name)

        # Atualiza os símbolos para a próxima camada
        input_symbols = output_symbols

    return equations_dict

def get_optimal_folds(n_samples: int):
    """Determina automaticamente o número de folds baseado no tamanho dos dados."""
    if n_samples < 50:
        return min(n_samples, 5)
    elif 50 <= n_samples < 500:
        return 5
    elif 500 <= n_samples < 5000:
        return 10
    else:
        return 20

def get_y_y_predicted_values(df: pd.DataFrame):
    df_sorted = df.sort_values(by="Y Predicted")

    colunas = df_sorted.columns.tolist()
    indice_y_pred = colunas.index("Y Predicted")
    coluna_anterior = colunas[indice_y_pred - 1] if indice_y_pred > 0 else None
    y = []
    
    if coluna_anterior:
        y = df_sorted[coluna_anterior].tolist()
        
    y_predicted = df_sorted["Y Predicted"].tolist()

    return y, y_predicted

def get_classes(y_values: list, output_neurons: list):
    lb = LabelBinarizer()
    lb.fit(y_values)

    classes = lb.classes_.tolist()

    mapping = dict(zip(output_neurons, classes))
    return mapping

def get_categoric_mean(df: pd.DataFrame, categorical_cols: list[str], is_categoric: bool):
    categoric_mean: dict = {}

    # Pega a coluna de resposta (última coluna)
    y_col = df.columns[-1]

    df = df.copy()

    # Se a resposta for categórica, transformamos em numérico com encoding
    if is_categoric:
        df[y_col], _ = pd.factorize(df[y_col])

    # Calcula a média de Y para cada categoria das colunas X
    for col in categorical_cols:
        for category in df[col].dropna().unique():
            filtered_df = df[df[col] == category]
            mean_y = filtered_df[y_col].mean()
            key = f"{col}[{category}]"
            categoric_mean[key] = mean_y

    for category in df[y_col].unique():
        mean_y_cat = df[df[y_col] == category][y_col].mean()
        key = f"{y_col}[{category}]"
        categoric_mean[key] = mean_y_cat

    return categoric_mean

def is_categorical_y(y):
    y_series = pd.Series(y)
    
    if y_series.dtype == 'O' or y_series.dtype.name == 'category':
        return True 
    
    unique_values = y_series.unique()
    
    if len(unique_values) <= 10 and all(isinstance(v, (str, int)) for v in unique_values):
        return True 
    
    return False 

def create_neuron_diagramm(equations: dict):
    layer_neurons = defaultdict(list)
    for neuron_key in equations:
        match = re.match(r"h(\d+)_(\d+)", neuron_key)
        if match:
            layer = int(match.group(1))
            neuron = int(match.group(2))
            layer_neurons[layer].append({
                    "id": neuron_key,
                    "label": f"H{layer}",  # nome da camada
                    "neuronIndex": neuron  # número do neurônio na camada
                })
        else:
                # Se for a camada de entrada (ex: Age, BMI, etc), pule
            continue

        # Criar lista de camadas com seus neurônios
    diagram_layers = []

        # Camada de entrada (pode ser inferida com base nas variáveis da primeira equação)
    input_features = sorted(list(set(
            re.findall(r"[A-Za-z_]+", equations[list(equations.keys())[0]])
        )))
    diagram_layers.append({
            "layerIndex": 0,
            "type": "input",
            "label": "Input Layer",
            "neurons": [{"id": name, "label": "Input", "neuronIndex": idx + 1} for idx, name in enumerate(input_features)]
        })

        # Camadas ocultas e possivelmente saída
    for layer_idx in sorted(layer_neurons.keys()):
        diagram_layers.append({
                "layerIndex": layer_idx,
                "type": "hidden" if layer_idx != max(layer_neurons.keys()) else "output",
                "label": f"Hidden Layer {layer_idx}" if layer_idx != max(layer_neurons.keys()) else "Output Layer",
                "neurons": sorted(layer_neurons[layer_idx], key=lambda x: x["neuronIndex"])
            })
        
    return diagram_layers

import gc
from typing import List, Union

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor

from utils.neural_networks.neural_networks_calculate import (
    calculate_metrics,
    create_feature_importance,
    create_neuron_diagramm,
    generate_forward_pass_equation,
    generate_learning_curve,
    get_categoric_mean,
    get_classes,
    get_y_y_predicted_values,
    is_categorical_y,
    transform_columns,
)
from utils.profiler.profiler_calculate import expand_equation

holdout_router = APIRouter()


class HodoutSettings(BaseModel):
    testSize: float


class HodoutNeuralNetwork(BaseModel):
    inputData: dict[str, List[Union[str, float]]]
    settings: HodoutSettings
    type: str
    categoricColumns: List[str]


@holdout_router.post(
    "/holdout/calculate",
    tags=["Rede Neural"],
    description="""Calcula rede neural utilizando método hodout. 
                            inputData: DataFrame com várias colunas de features (X) e uma coluna de resposta (Y);
                            settings: objeto com (solver: str, hiddenLayer: int, testSize: float, learningRate: str) para definir a configuração da rede neural;
                            type: tipo de ativação da rede neural (relu, logistic, tahn, identity)
                          """,
    response_model=object,
)
def calculate_hodout_neural_networks(body: HodoutNeuralNetwork):
    df = pd.DataFrame(body.inputData)

    try:
        setting = body.settings
        test_size = setting.testSize
        activation = body.type
        categoric_columns = body.categoricColumns

        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        for col in categoric_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        is_categoric = is_categorical_y(y)

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # transforma colunas
        transformers = transform_columns(X, categorical_cols, numeric_cols)

        preprocessor = ColumnTransformer(transformers, remainder="passthrough")

        X = preprocessor.fit_transform(X)
        transformed_feature_names = preprocessor.get_feature_names_out()

        X_transformed = pd.DataFrame(X, columns=transformed_feature_names)

        if hasattr(X, "toarray"):
            X = X.toarray()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=0
        )

        num_features = X.shape[1]

        if is_categoric:
            best_model = MLPClassifier(random_state=0, max_iter=300)
            scoring_metric = "accuracy"
        else:
            best_model = MLPRegressor(random_state=0, max_iter=300)
            scoring_metric = "r2"

        param_grid = {
            "hidden_layer_sizes": [
                # 1 camada (máx 5 neurônios)
                (1,),
                (2,),
                (3,),
                # 2 camadas
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 5),
                (5, 4),
                # 3 camadas
                (3, 2, 1),
                (4, 3, 2),
                (5, 4, 3),
                (5, 5, 5),
                (2, 2, 2),
                (3, 3, 3),
                # 4 camadas
                (2, 2, 2, 2),
                (3, 2, 2, 1),
                (4, 3, 2, 1),
                (5, 4, 3, 2),
                # 5 camadas
                (2, 2, 2, 2, 2),
                (3, 2, 2, 2, 1),
                (4, 3, 2, 1, 1),
                # 6 camadas
                (2, 2, 2, 2, 2, 2),
                (4, 3, 2, 2, 2, 1),
            ],
            "activation": [activation],
            "solver": ["adam", "sgd"],
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "alpha": [0.001, 0.01, 0.1],
        }

        grid_search = GridSearchCV(
            best_model,
            param_grid,
            scoring=scoring_metric,
            cv=[(slice(None), slice(None))],
            n_jobs=1,
            pre_dispatch="2*n_jobs",
        )

        grid_search.fit(X_transformed, y)

        model = grid_search.best_estimator_

        # Apagar o objeto pesado
        del grid_search

        # Forçar a liberação de memória
        gc.collect()

        # Treina o modelo
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        models = {
            "training": {},
            "validation": {},
        }

        # Calcula métricas
        calculate_metrics(
            df,
            X_train,
            X_test,
            y_train,
            y_test,
            model,
            y_pred_train,
            y_pred_test,
            models,
        )

        # Calcular a importância das features usando permutation_importance
        feature_importance_sorted = create_feature_importance(
            transformed_feature_names, num_features, X_test, y_test, model
        )

        # Gera a curva de aprendizado
        learning_curve_data = generate_learning_curve(model, X_transformed, y)

        # Gera equação
        equations = generate_forward_pass_equation(
            model, X_transformed, categorical_cols, activation
        )

        df["Y Predicted"] = model.predict(X_transformed)

        # Calcula Y predito
        y_values, y_predicted = get_y_y_predicted_values(df)

        layer_prefixes = [key.split("_")[0] for key in equations.keys() if "_" in key]
        last_layer_prefix = sorted(set(layer_prefixes), key=lambda x: int(x[1:]))[-1]
        last_key = last_layer_prefix + "_"

        final_equation = expand_equation(equations, last_key)
        output_neurons = [k for k in equations.keys() if k.startswith(last_key)]
        output_neurons.sort()

        mapping = get_classes(y_values, output_neurons) if is_categoric else {}

        categoric_mean = get_categoric_mean(df, categorical_cols, is_categoric)

        diagram_layers = create_neuron_diagramm(equations)

        # Montar o resultado final
        diagram_data = {"layers": diagram_layers}

        return {
            "models": models,
            "featureImportance": feature_importance_sorted,
            "isCategoric": is_categoric,
            "learningCurve": learning_curve_data,
            "equations": equations,
            "yValues": y_values,
            "yPredictedValues": y_predicted,
            "uniqueEquation": final_equation,
            "categoricNeurons": mapping if is_categoric else {},
            "categoricColumns": categorical_cols,
            "categoricMean": categoric_mean,
            "diagramData": diagram_data,
        }

    except Exception as e:
        print("error", e)
        raise HTTPException(status_code=500, detail=f"neuralNetworkError: {e}")

import gc
from typing import List, Union

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sklearn.calibration import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor

from utils.neural_networks.neural_networks_calculate import (
    calculate_average_metrics,
    calculate_kfold_metrics,
    create_feature_importance,
    create_neuron_diagramm,
    generate_forward_pass_equation,
    generate_learning_curve,
    get_categoric_mean,
    get_classes,
    get_optimal_folds,
    get_y_y_predicted_values,
    is_categorical_y,
    transform_columns,
)
from utils.profiler.profiler_calculate import expand_equation

kfold_router = APIRouter()


class KfoldSettings(BaseModel):
    testSize: float
    fold: int


class KfoldNeuralNetwork(BaseModel):
    inputData: dict[str, List[Union[str, float]]]
    settings: KfoldSettings
    type: str
    categoricColumns: List[str]


@kfold_router.post(
    "/kfold/calculate",
    tags=["Rede Neural"],
    description="""Calcula rede neural utilizando método k-fold.
                            inputData: DataFrame com várias colunas de features (X) e uma coluna de resposta (Y);
                            settings: objeto com (solver: str, hiddenLayer: int, testSize: float, learningRate: str) para definir a configuração da rede neural;
                            type: tipo de ativação da rede neural (relu, logistic, tanh, identity)
                          """,
    response_model=object,
)
def calculate_kfold_neural_networks(body: KfoldNeuralNetwork):
    df = pd.DataFrame(body.inputData)

    try:
        setting = body.settings
        fold = setting.fold
        activation = body.type
        categoric_columns = body.categoricColumns

        categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

        for col in categoric_columns:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        is_categoric = is_categorical_y(y)

        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        # Transforma colunas
        transformers = transform_columns(X, categorical_cols, numeric_cols)
        preprocessor = ColumnTransformer(transformers, remainder="passthrough")

        # Transforma X em X_transformed usando o ColumnTransformer
        X_transformed = preprocessor.fit_transform(X)
        transformed_feature_names = preprocessor.get_feature_names_out()

        # Converte X_transformed de volta para DataFrame para acessar as colunas
        X_transformed = pd.DataFrame(X_transformed, columns=transformed_feature_names)

        models = {
            "training": {
                "accuracy": [],
                "rocAuc": [],
                "confusionMatrix": [],
                "meanSquaredError": [],
                "r2": [],
                "mean": [],
                "stdDev": [],
                "precision": [],
                "recall": [],
                "fScore": [],
                "rocCurve": [],
            },
            "validation": {
                "accuracy": [],
                "rocAuc": [],
                "confusionMatrix": [],
                "meanSquaredError": [],
                "r2": [],
                "mean": [],
                "stdDev": [],
                "precision": [],
                "recall": [],
                "fScore": [],
                "rocCurve": [],
            },
        }

        if is_categoric:
            base_model = MLPClassifier(random_state=0, max_iter=300)
            scoring_metric = "accuracy"
        else:
            base_model = MLPRegressor(random_state=0, max_iter=300)
            scoring_metric = "r2"

        n_samples = len(X)
        folds = fold if fold > 1 else get_optimal_folds(n_samples)

        if is_categoric:
            min_class_size = y.value_counts().min()
            if folds > min_class_size:
                folds = min_class_size
            kf = StratifiedKFold(n_splits=folds)
        else:
            kf = KFold(n_splits=folds)

        # Define a grade de hiperparâmetros para o Grid Search
        param_grid = {
            "hidden_layer_sizes": [
                # 1 camada (máx 5 neurônios)
                (1,),
                (2,),
                (3,),
                # 2 camadas
                (1, 1),
                (2, 2),
                (3, 3),
                (4, 4),
                (5, 5),
                (5, 4),
                # 3 camadas
                (3, 2, 1),
                (4, 3, 2),
                (5, 4, 3),
                (5, 5, 5),
                (2, 2, 2),
                (3, 3, 3),
                # 4 camadas
                (2, 2, 2, 2),
                (3, 2, 2, 1),
                (4, 3, 2, 1),
                (5, 4, 3, 2),
                # 5 camadas
                (2, 2, 2, 2, 2),
                (3, 2, 2, 2, 1),
                (4, 3, 2, 1, 1),
                # 6 camadas
                (2, 2, 2, 2, 2, 2),
                (4, 3, 2, 2, 2, 1),
            ],
            "activation": [activation],
            "solver": ["adam", "sgd"],
            "learning_rate": ["constant", "invscaling", "adaptive"],
            "alpha": [0.001, 0.01, 0.1],
        }

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=kf,
            scoring=scoring_metric,
            n_jobs=1,
            pre_dispatch="2*n_jobs",
        )

        grid_search.fit(X_transformed, y)

        best_model = grid_search.best_estimator_

        # Apagar o objeto pesado
        del grid_search

        # Forçar a liberação de memória
        gc.collect()

        for train_index, test_index in kf.split(X_transformed, y):
            X_train, X_test = (
                X_transformed.iloc[train_index],
                X_transformed.iloc[test_index],
            )
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Treina o modelo com os melhores hiperparâmetros
            best_model.fit(X_train, y_train)

            # Calcula métricas (para treinamento e validação)
            calculate_kfold_metrics(
                is_categoric,
                models,
                X_train,
                X_test,
                y_train,
                y_test,
                best_model,
                best_model.predict(X_train),
                best_model.predict(X_test),
            )

        models["training"] = calculate_average_metrics(models["training"])
        models["validation"] = calculate_average_metrics(models["validation"])

        # Cria features importantes
        feature_importance_sorted = create_feature_importance(
            transformed_feature_names, len(X.columns), X_test, y_test, best_model
        )

        # Gera a curva de aprendizado
        learning_curve_data = generate_learning_curve(best_model, X_transformed, y)

        # Gera a equação
        equations = generate_forward_pass_equation(
            best_model, X_transformed, categorical_cols, activation
        )

        df["Y Predicted"] = best_model.predict(X_transformed)

        # Ordenar pelo Y Predicted
        y_values, y_predicted = get_y_y_predicted_values(df)

        layer_prefixes = [key.split("_")[0] for key in equations.keys() if "_" in key]
        last_layer_prefix = sorted(set(layer_prefixes), key=lambda x: int(x[1:]))[-1]
        last_key = last_layer_prefix + "_"

        final_equation = expand_equation(equations, last_key)
        output_neurons = [k for k in equations.keys() if k.startswith(last_key)]
        output_neurons.sort()

        mapping = get_classes(y_values, output_neurons) if is_categoric else {}
        categoric_mean = get_categoric_mean(df, categorical_cols, is_categoric)

        diagram_layers = create_neuron_diagramm(equations)

        # Montar o resultado final
        diagram_data = {"layers": diagram_layers}

        print("final_equation", final_equation)

        return {
            "models": models,
            "featureImportance": feature_importance_sorted,
            "isCategoric": is_categoric,
            "learningCurve": learning_curve_data,
            "equations": equations,
            "yValues": y_values,
            "yPredictedValues": y_predicted,
            "uniqueEquation": final_equation,
            "categoricNeurons": mapping if is_categoric else {},
            "categoricMean": categoric_mean,
            "categoricColumns": categorical_cols,
            "diagramData": diagram_data,
        }

    except Exception as e:
        print("error", e)
        raise HTTPException(status_code=500, detail=f"neuralNetworkError: {e}")


meu front end

import React, { useEffect, useState } from "react";
import { useTranslation } from "next-i18next";
import { Button, Row, Col, Select, Tooltip } from "antd";
import Transfer from "shared/transfer";
import { useRouter } from "next/router";
import { AiOutlineInfoCircle, AiOutlineSetting } from "react-icons/ai";
import NeuralNetworksSettingModal from "./neuralNetworksSetting";
import { NeuralNetworksSettings } from "components/neuralNetworks/interface";
import { transformDataToTargetKeys } from "utils/core";
import { RecordType } from "components/insertData/inteface";
import Modal from "shared/modal";
import { useAuth } from "hooks/useAuth";
import axios from "axios";

const fetcher = axios.create({
  baseURL: "/api",
});

const InsertOptionsNeuralNetworks = (props: {
  visibleOption: boolean;
  data: any[];
  xVariables: RecordType[];
  yVariables: RecordType[];
  onModalClose: () => void;
}) => {
  const { visibleOption, data, xVariables, yVariables, onModalClose } = props;

  const { t: commonT } = useTranslation("common");
  const router = useRouter();

  const { user } = useAuth();

  const [loading, setLoading] = useState(false);

  const [calculationMethod, setCalculationMethod] = useState("holdout");

  const [type, setType] = useState("relu");
  const [settingModel, setSettingModel] = useState<NeuralNetworksSettings>({
    testSize: 0.3,
  });

  const [categoricColumns, setCategoricColumns] = useState<string[]>([]);
  const [categoricColumnsOptions, setCategoricColumnsOptions] = useState<any[]>(
    []
  );

  const [neuralNetworkSettingModal, setNeuralNetworkSettingModal] =
    useState(false);

  const calculattionMethodOptions = [
    {
      label: commonT("neuralNetworks.hodout"),
      value: "holdout",
    },
    {
      label: commonT("neuralNetworks.kfold"),
      value: "kFold",
    },
  ];

  const hodoutOptions = [
    {
      label: commonT("neuralNetworks.relu"),
      value: "relu",
    },
    {
      label: commonT("neuralNetworks.tanh"),
      value: "tanh",
    },
    {
      label: commonT("neuralNetworks.logistic"),
      value: "logistic",
    },
    {
      label: commonT("neuralNetworks.identity"),
      value: "identity",
    },
  ];

  const [neuralNetworksTargetKeys, setNeuralNetworksTargetKeys] = useState<
    string[]
  >([]);
  const [neuralNetworksSelectedKeys, setNeuralNetworksSelectedKeys] = useState<
    string[]
  >([]);
  const [
    neuralNetworksTargetKeysResponse,
    setNeuralNetworksTargetKeysResponse,
  ] = useState<string[]>([]);
  const [
    neuralNetworksSelectedKeysResponse,
    setNeuralNetworksSelectedKeysResponse,
  ] = useState<string[]>([]);

  const neuralNetworksOnChange = (nextTargetKeys: string[]) => {
    setNeuralNetworksTargetKeys(nextTargetKeys);
  };

  const neuralNetworksOnSelectChange = (
    sourceSelectedKeys: string[],
    targetSelectedKeys: string[]
  ) => {
    setNeuralNetworksSelectedKeys([
      ...sourceSelectedKeys,
      ...targetSelectedKeys,
    ]);
  };

  const neuralNetworksOnChangeResponse = (nextTargetKeys: string[]) => {
    setNeuralNetworksTargetKeysResponse(nextTargetKeys);
  };

  const neuralNetworksOnSelectChangeResponse = (
    sourceSelectedKeys: string[],
    targetSelectedKeys: string[]
  ) => {
    setNeuralNetworksSelectedKeysResponse([
      ...sourceSelectedKeys,
      ...targetSelectedKeys,
    ]);
  };

  useEffect(() => {
    const createCategoricColumnsOptions = () => {
      if (neuralNetworksTargetKeys.length > 0) {
        const options = [];
        neuralNetworksTargetKeys.forEach((el) => {
          options.push({ value: el, label: el });
        });
        setCategoricColumnsOptions(options);
      } else {
        setCategoricColumnsOptions([]);
      }
    };
    createCategoricColumnsOptions();
  }, [neuralNetworksTargetKeys]);

  useEffect(() => {
    if (calculationMethod === "kFold") {
      setSettingModel({
        testSize: 0.3,
        fold: 5,
      });
    } else {
      setSettingModel({
        testSize: 0.3,
      });
    }
  }, [calculationMethod]);

  const neuralNetworksHandleOk = async () => {
    setLoading(true);

    try {
      const targetKeysConcat = neuralNetworksTargetKeys.concat(
        neuralNetworksTargetKeysResponse
      );
      const targetKeysData = transformDataToTargetKeys(
        data,
        targetKeysConcat,
        categoricColumns
      );

      const dataToSend = prepareDataToSend(targetKeysData);

      const newWindow = await handleCalculationMethod(
        calculationMethod,
        dataToSend
      );

      if (newWindow) newWindow.opener = null;

      clearVariables();
      onModalClose();
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  function prepareDataToSend(targetKeysData: any[]) {
    const variables = Object.keys(targetKeysData[0]);
    const dataToSend: Record<string, any[]> = {};

    variables.forEach((variable) => {
      dataToSend[variable] = targetKeysData.map((item) => item[variable]);
    });

    return dataToSend;
  }

  async function handleCalculationMethod(
    method: string,
    dataToSend: Record<string, any[]>
  ) {
    const baseUrl = `${window.location.origin}/${router.locale}`;

    await postQueueItem(dataToSend, method);

    return window.open(
      `${baseUrl}/analysisManagement`,
      "_blank",
      "noopener,noreferrer"
    );
  }

  async function postQueueItem(
    inputData: Record<string, any[]>,
    neuralNetworkType: string
  ) {
    await fetcher.post("neuralNetworks/mongodb/postQueueItem", {
      data: {
        status: "pending",
        payload: {
          inputData,
          type,
          settings: settingModel,
          categoricColumns,
          neuralNetworkType,
        },
        created_at: new Date(),
        user: user.username,
        visible: true,
        tool: "neuralNetwork",
      },
    });
  }

  const clearVariables = () => {
    setType("relu");
    setCalculationMethod("holdout");
    setNeuralNetworksTargetKeys([]);
    setNeuralNetworksSelectedKeys([]);
    setNeuralNetworksTargetKeysResponse([]);
    setNeuralNetworksSelectedKeysResponse([]);
  };

  const handleCancel = () => {
    clearVariables();
    onModalClose();
  };

  const handleSettings = (settings: NeuralNetworksSettings) => {
    setSettingModel(settings);
  };

  return (
    <>
      <Modal
        title={commonT("neuralNetworks.title")}
        open={visibleOption}
        onOk={neuralNetworksHandleOk}
        onCancel={handleCancel}
        footer={[
          <Button key="back" onClick={handleCancel}>
            {commonT("buttons.back")}
          </Button>,
          <Button
            key="submit"
            type="primary"
            loading={loading}
            onClick={neuralNetworksHandleOk}
            disabled={
              neuralNetworksTargetKeys.length <= 0 ||
              neuralNetworksTargetKeysResponse.length !== 1
            }
          >
            <a target="_blank" rel="noopener noreferrer">
              {commonT("buttons.submit")}
            </a>
          </Button>,
        ]}
        modalSize={{
          maxHeight: "70vh",
          overflow: "auto",
          overflowY: "auto"
        }}
      >
        <Row style={{ fontWeight: "bolder" }}>
          <Col span={12}> {commonT("neuralNetworks.calculationMethod")}</Col>
          <Col offset={1} span={11}>
            {commonT("neuralNetworks.type")}
          </Col>
        </Row>
        <Row style={{ fontWeight: "bolder", marginBottom: 20 }}>
          <Col span={11}>
            <div style={{ display: "flex", gap: 10 }}>
              <Select
                options={calculattionMethodOptions}
                value={calculationMethod}
                onChange={setCalculationMethod}
                style={{ width: "80%" }}
              />

              <Tooltip
                placement="top"
                title={
                  calculationMethod === "kFold" ? (
                    <>{commonT("neuralNetworks.infoType.kFold")}</>
                  ) : (
                    <>{commonT("neuralNetworks.infoType.holdout")}</>
                  )
                }
              >
                <Button
                  icon={
                    <AiOutlineInfoCircle
                      style={{ transform: "scale(1.3)", color: "darkblue" }}
                    />
                  }
                />
              </Tooltip>

              <Tooltip
                placement="top"
                title={commonT("neuralNetworks.settings")}
              >
                <Button
                  onClick={() => setNeuralNetworkSettingModal(true)}
                  icon={
                    <AiOutlineSetting style={{ transform: "scale(1.3)" }} />
                  }
                />
              </Tooltip>
            </div>
          </Col>
          <Col offset={2} span={11}>
            <div style={{ display: "flex", gap: 10 }}>
              <Select
                options={hodoutOptions}
                value={type}
                onChange={setType}
                style={{ width: "95%" }}
              />
              <Tooltip
                placement="top"
                title={
                  type === "relu"
                    ? commonT("neuralNetworks.infoType.relu")
                    : type === "tahn"
                    ? commonT("neuralNetworks.infoType.tahn")
                    : type === "logistic"
                    ? commonT("neuralNetworks.infoType.logistic")
                    : type === "softmax"
                    ? commonT("neuralNetworks.infoType.softmax")
                    : commonT("neuralNetworks.infoType.identity")
                }
              >
                <Button
                  icon={
                    <AiOutlineInfoCircle
                      style={{ transform: "scale(1.3)", color: "darkblue" }}
                    />
                  }
                />
              </Tooltip>
            </div>
          </Col>
        </Row>

        <Row style={{ fontWeight: "bolder" }}>
          <Col span={11}>{commonT("multipleRegression.categoricColumns")}</Col>
        </Row>

        <Row>
          <Col span={11}>
            <Select
              value={categoricColumns}
              mode="multiple"
              style={{ width: "100%" }}
              onChange={(value) => setCategoricColumns(value)}
              options={categoricColumnsOptions}
              maxTagCount={2}
            />
          </Col>
        </Row>

        <Transfer
          mandatory={true}
          dataSource={xVariables}
          titles={[
            commonT("neuralNetworks.xSelect"),
            commonT("neuralNetworks.xSelected"),
          ]}
          targetKeys={neuralNetworksTargetKeys}
          selectedKeys={neuralNetworksSelectedKeys}
          onChange={neuralNetworksOnChange}
          onSelectChange={neuralNetworksOnSelectChange}
        />
        <Transfer
          mandatory={true}
          dataSource={yVariables}
          titles={[
            commonT("neuralNetworks.variableResponse"),
            commonT("neuralNetworks.variableSelectedResponse"),
          ]}
          targetKeys={neuralNetworksTargetKeysResponse}
          selectedKeys={neuralNetworksSelectedKeysResponse}
          onChange={neuralNetworksOnChangeResponse}
          onSelectChange={neuralNetworksOnSelectChangeResponse}
        />
      </Modal>

      <NeuralNetworksSettingModal
        showOpenModal={neuralNetworkSettingModal}
        onModalClose={() => setNeuralNetworkSettingModal(false)}
        onSave={handleSettings}
        neuralNetworkType={calculationMethod}
      />
    </>
  );
};

export default InsertOptionsNeuralNetworks;

import React, { useEffect, useState } from "react";
import {
  Button,
  Checkbox,
  Col,
  Form,
  InputNumber,
  Row,
  Tooltip,
} from "antd";
import { useTranslation } from "next-i18next";
import { AiOutlineInfoCircle } from "react-icons/ai";
import { NeuralNetworksSettings } from "components/neuralNetworks/interface";
import Modal from "shared/modal";

const NeuralNetworksSettingModal = (props: {
  showOpenModal: boolean;
  onModalClose: () => void;
  neuralNetworkType: string;
  onSave: (settingsModel: NeuralNetworksSettings) => void;
}) => {
  const { showOpenModal, neuralNetworkType, onModalClose, onSave } = props;
  const { t: commonT } = useTranslation("common");

  const [openModal, setOpenModal] = useState(showOpenModal);
  const [testSize, setTestSize] = useState(0.3);
  const [folds, setFolds] = useState(5);
  const [enableAutoFolds, setEnableAutoFolds] = useState(false);

  useEffect(() => {
    setOpenModal(showOpenModal);
  }, [showOpenModal]);

  const handleCancel = () => {
    onModalClose();
  };

  const neuralNetworksSettingsHandleOk = () => {
    const model: NeuralNetworksSettings = {
      testSize: testSize,
      fold: enableAutoFolds ? 0 : folds,
    };
    onSave(model);
    setOpenModal(false);
  };

  return (
    <Modal
      key="settingModal"
      title={commonT("neuralNetworks.settingsModal.title")}
      open={openModal}
      onOk={neuralNetworksSettingsHandleOk}
      onCancel={handleCancel}
      width={300}
      footer={[
        <Button key="back" onClick={handleCancel}>
          {commonT("buttons.back")}
        </Button>,
        <Button
          key="submit"
          type="primary"
          onClick={neuralNetworksSettingsHandleOk}
        >
          {commonT("buttons.submit")}
        </Button>,
      ]}
    >
      <Form layout="horizontal">
        <Row>
          <Col span={20}>
            <Form.Item label={commonT("neuralNetworks.settingsModal.testSize")}>
              <InputNumber
                value={testSize}
                min={0.1}
                max={1}
                onChange={(value) => setTestSize(Number(value))}
              />
            </Form.Item>
          </Col>
          <Col offset={1} span={3}>
            <Tooltip
              title={commonT(
                "neuralNetworks.settingsModal.testeSizeDescription"
              )}
            >
              <Button
                icon={<AiOutlineInfoCircle style={{ color: "darkblue" }} />}
              />
            </Tooltip>
          </Col>
        </Row>
        {neuralNetworkType !== "holdout" && (
          <>
            <Row>
              <Col span={24}>
                <Form.Item
                  label={commonT(
                    "neuralNetworks.settingsModal.enableAutoFolds"
                  )}
                >
                  <Checkbox
                    checked={enableAutoFolds}
                    onChange={(e) => setEnableAutoFolds(e.target.checked)}
                  />
                </Form.Item>
              </Col>
            </Row>
            <Row>
              <Col span={20}>
                <Form.Item
                  label={commonT("neuralNetworks.settingsModal.folds")}
                >
                  <InputNumber
                    value={folds}
                    min={1}
                    max={10}
                    disabled={enableAutoFolds}
                    onChange={(value) => setFolds(Number(value))}
                  />
                </Form.Item>
              </Col>

              <Col offset={1} span={3}>
                <Tooltip
                  title={commonT(
                    "neuralNetworks.settingsModal.foldsDescription"
                  )}
                >
                  <Button
                    icon={<AiOutlineInfoCircle style={{ color: "darkblue" }} />}
                  />
                </Tooltip>
              </Col>
            </Row>
          </>
        )}
      </Form>
    </Modal>
  );
};

export default NeuralNetworksSettingModal;


a analise no front

import React, { useEffect, useRef, useState } from "react";
import axios from "axios";
import { useTranslation } from "next-i18next";
import { useAuth } from "hooks/useAuth";
import ContentHeader from "shared/contentHeader";
import { Spin } from "shared/spin";
import { Col, Row, message, notification } from "antd";
import {
  DataToCompile,
  NeuralNetworkDiagramLayer,
  NeuralNetworkHoldoutData,
  validateRSquare,
} from "../interface";
import Table from "shared/table";
import { formatNumber } from "utils/formatting";
import {
  createConfusionMatrix,
  createParameterEstimatesTable,
  createSummaryTable,
} from "../table";
import { dataSortOrder } from "utils/core";
import { useRouter } from "next/router";
import dynamic from "next/dynamic";
import InformationTable from "../informationTable";
import Equation from "shared/equation";
import {
  HighChartCustomSeries,
  HighChartTemplate,
} from "shared/widget/chartHub/interface";
import {
  UniqueAnalysisManagementResponse,
} from "components/analysisManagement/interface";

const ChartHub = dynamic(() => import("shared/widget/chartHub"), {
  ssr: false,
  loading: () => <Spin />,
});

const Profiler = dynamic(() => import("shared/profiler"), {
  ssr: false,
  loading: () => <Spin />,
});

const fetcher = axios.create({
  baseURL: "/api",
});

export const NeuralNetworksHoldout: React.FC = () => {
  const { t: commonT } = useTranslation("common", { useSuspense: false });
  const { user } = useAuth();
  const router = useRouter();

  const [loadingPage, setLoadingPage] = useState(true);
  const [loadingSummary, setLoadingSummary] = useState(true);
  const [loadingParameterEstimates, setLoadingParameterEstimates] =
    useState(true);

  const [chartData, setChartData] = useState<HighChartTemplate>({});
  const [isCategoric, setIsCategoric] = useState(false);
  const [equation, setEquation] = useState<string>("");

  const summaryTable = useRef<any>({});
  const parameterEstimates = useRef<any>({});
  const confusionMatrixTraining = useRef<any>({});
  const confusionMatrixValidation = useRef<any>({});

  const [profilerData, setProfilerData] = useState<Record<string, any>>({});
  const [colNumber, setColNumber] = useState(3);
  const [loadingProfiler, setLoadingProfiler] = useState(true);
  const [neuralDiagram, setNeuralDiagram] = useState<HighChartTemplate>({});

  const [rSquare, setRSquare] = useState<validateRSquare>({
    r2: undefined,
    r2Validation: undefined,
  });

  useEffect(() => {
    const getData = async () => {
      const parsedUrlQuery = router?.query;
      if (Object.keys(parsedUrlQuery).length > 0) {
        const id = parsedUrlQuery.id as string;

        if (id) {
          try {
            const { data: analysisManagementResponse } =
              await fetcher.post<UniqueAnalysisManagementResponse>(
                "neuralNetworks/mongodb/getQueueId",
                {
                  id,
                }
              );

            const data = analysisManagementResponse.data
              .result as NeuralNetworkHoldoutData;
            const payload = analysisManagementResponse.data.payload;

            const variables = Object.keys(payload.inputData);
            const dataToSend: DataToCompile = {
              obj: payload.inputData,
              itens: variables,
            };

            const responseColumnVariable = variables.at(-1);

            setColNumber(Object.keys(dataToSend.obj).length - 1);
            setIsCategoric(data.isCategoric);

            setRSquare({
              r2: data.isCategoric
                ? data.models.training.accuracy
                : data.models.training.r2,
              r2Validation: data.isCategoric
                ? data.models.validation.accuracy
                : data.models.validation.r2,
            });

            const summaryTranslate = {
              term: commonT("neuralNetworks.terms"),
              accuracy: commonT("neuralNetworks.accuracy"),
              fScore: commonT("neuralNetworks.fScore"),
              mean: commonT("neuralNetworks.mean"),
              precision: commonT("neuralNetworks.precision"),
              recall: commonT("neuralNetworks.recall"),
              rocAuc: commonT("neuralNetworks.rocAuc"),
              stdDev: commonT("neuralNetworks.stdDev"),
              r2: commonT("neuralNetworks.r2"),
              rmse: commonT("neuralNetworks.rmse"),
              training: commonT("neuralNetworks.training"),
              validation: commonT("neuralNetworks.validation"),
            };

            summaryTable.current = createSummaryTable(
              data.models,
              summaryTranslate,
              data.isCategoric
            );
            setLoadingSummary(false);

            const parameterEstimateTransalte = {
              importance: commonT("neuralNetworks.importance"),
              term: commonT("neuralNetworks.terms"),
            };

            parameterEstimates.current = createParameterEstimatesTable(
              data.featureImportance,
              parameterEstimateTransalte
            );

            const paretoPlotData = data.featureImportance;
            delete paretoPlotData["Intercept"];

            setLoadingParameterEstimates(false);

            const estimatesList = Object.entries(paretoPlotData).reduce(
              (acc: Record<string, number>, [key, value]) => {
                acc[
                  key
                    .replace("num__", "")
                    .replace("onehot__", "")
                    .replace("remainder__", "")
                ] = value;
                return acc;
              },
              {}
            );

            const sortedList: { [key: string]: number } = dataSortOrder(
              estimatesList,
              "desc",
              true
            );

            const columnList = Object.entries(sortedList).map(([key]) => key);

            const sortedEstimates: number[] = Object.entries(sortedList)
              .map(([, value]) => value)
              .sort((a, b) => a - b);

            if (data.isCategoric) {
              const confusionMatrixTranslate = {
                term: commonT("neuralNetworks.terms"),
              };

              confusionMatrixTraining.current = createConfusionMatrix(
                data.models.training.confusionMatrix,
                confusionMatrixTranslate
              );

              confusionMatrixValidation.current = createConfusionMatrix(
                data.models.validation.confusionMatrix,
                confusionMatrixTranslate
              );
            }

            // equaçao
            const lastKey = Object.keys(data.equations).pop();
            const lastValue = lastKey ? data.equations[lastKey] : undefined;
            setEquation(`Y = ${lastValue}`);

            const profilerObj: Record<string, any> = {};

            if (data.uniqueEquation) {
              for (const key of Object.keys(data?.uniqueEquation)) {
                const equationNeuron = data.uniqueEquation[key];

                profilerObj[key] = [];
                profilerObj[key]["maxValue"] = data.isCategoric
                  ? 1
                  : Math.max(...data.yPredictedValues);
                profilerObj[key]["minValue"] = data.isCategoric
                  ? 0
                  : Math.min(...data.yPredictedValues);

                profilerObj[key]["equation"] = equationNeuron;
                profilerObj[key]["equationOrigin"] = equationNeuron;
                profilerObj[key]["responseTitle"] = "Y"; // pegar a ultima coluna do inputData

                profilerObj[key]["data"] = {
                  [key]: payload.inputData, //dataToSend.obj,
                };

                profilerObj[key]["categoricNeurons"] = data.categoricNeurons;
                profilerObj[key]["categoricColumns"] = data.categoricColumns;

                profilerObj[key]["bounds"] = {};

                const categoricMean = data.categoricMean;

                for (const colKey of Object.keys(dataToSend.obj).filter(
                  (el) => el !== responseColumnVariable
                )) {
                  const items = dataToSend.obj[colKey];

                  if (typeof items[0] === "string") {
                    for (const meanKey of Object.keys(categoricMean)) {
                      if (meanKey.startsWith(`${colKey}[`)) {
                        profilerObj[key]["bounds"][meanKey] = [0, 1];
                      }
                    }
                  } else {
                    const min = Math.min(...items);
                    const max = Math.max(...items);
                    profilerObj[key]["bounds"][colKey] = [min, max];
                  }
                }
              }
            }

            setProfilerData(profilerObj);

            setLoadingProfiler(false);

            notification.success({
              message: commonT("neuralNetworks.proccesSuccess"),
            });

            generateChart(
              data,
              columnList,
              sortedEstimates,
              responseColumnVariable
            );

            setLoadingPage(false);
          } catch (error) {
            console.error(error);
            message.error({
              content: commonT("error.general.unexpectedMsg"),
            });
          }
        }
      }
    };
    getData().catch(console.error);
  }, [router.isReady, router.query]);

  useEffect(() => {
    const countAccess = async () => {
      if (user) {
        fetcher.post("access_counter", {
          user: user?.username,
          operation: "neuralNetworks",
        });
      }
    };
    countAccess().catch(console.error);
  }, []);

  function generateChart(
    data: NeuralNetworkHoldoutData,
    columnList: string[],
    sortedEstimates: number[],
    responseColumnVariable: string
  ) {
    const trainingRocCurve = data.models.training.rocCurve;
    const validationRocCurve = data.models.validation.rocCurve;

    const trainingLinePoints = Array.from(
      { length: trainingRocCurve.length + 1 },
      (_, index) => {
        return index / trainingRocCurve.length;
      }
    );

    const validationLinePoints = Array.from(
      { length: validationRocCurve.length + 1 },
      (_, index) => {
        return index / validationRocCurve.length;
      }
    );

    const yValuesTraining = data.models.training.rocCurve.map((el) => el.y);
    const yValuesValidation = data.models.validation.rocCurve.map((el) => el.y);
    const testScoresMean = data.learningCurve.testScoresMean;
    const trainScoresMean = data.learningCurve.trainScoresMean;

    const rocAuc = data.models.training.rocAuc;
    const rocAucValidation = data.models.validation.rocAuc;

    const chartsForVariable = buildChart(
      columnList,
      sortedEstimates,
      yValuesTraining,
      trainingLinePoints,
      yValuesValidation,
      validationLinePoints,
      testScoresMean,
      trainScoresMean,
      data.yValues,
      data.yPredictedValues,
      rocAuc,
      rocAucValidation,
      responseColumnVariable
    );

    setChartData(chartsForVariable);

    const neuralDiagramData = buildNeuralDiagram(data.diagramData.layers);
    setNeuralDiagram(neuralDiagramData);
  }

  function buildNeuralDiagram(layers: NeuralNetworkDiagramLayer[]) {
    const chartsForVariable: HighChartTemplate = {};

    const diagramSeries: HighChartCustomSeries = {
      type: "neuralNetworkDiagram",
      name: commonT("neuralNetworks.neuralDiagramTitle"),
      layers: layers,
      data: [],
    };

    chartsForVariable["neural"] = {
      type: "neuralNetworkDiagram",
      seriesData: [diagramSeries],
      options: {
        chartName: "Y",
        title: commonT("neuralNetworks.neuralDiagramTitle"),
        xAxisTitle: commonT("neuralNetworks.layer"),
        yAxisTitle: commonT("neuralNetworks.neurons"),
      },
    };

    return chartsForVariable;
  }

  function buildChart(
    columnList: string[],
    sortedEstimates: number[],
    yValuesTraining: any,
    trainingLinePoints: any,
    yValuesValidation: any,
    validationLinePoints: any,
    testScoresMean: any,
    trainScoresMean: any,
    yValues: any,
    yPredictedValues: any,
    rocAuc: any,
    rocAucValidation: any,
    responseColumn: string
  ) {
    const chartsForVariable: HighChartTemplate = {};

    chartsForVariable["estimates"] = {
      seriesData: [
        {
          data: sortedEstimates,
          name: commonT("neuralNetworks.chart.dataLegend"),
          type: "bar",
        },
      ],
      options: {
        title: commonT("neuralNetworks.chart.titleChart"),
        titleFontSize: 24,
        chartName: responseColumn,
        legendEnabled: true,
        useVerticalLimits: false,
        categories: columnList,
        barIsVertical: false,
      },
      type: "bar",
      displayName: commonT("neuralNetworks.chart.titleChart"),
    };

    chartsForVariable["roc"] = {
      seriesData: [
        {
          data: yValuesTraining,
          name: commonT("neuralNetworks.chart.dataLegendTraining"),
          type: "line",
        },
        {
          data: trainingLinePoints,
          name: `${commonT(
            "neuralNetworks.chart.rocAucTraining"
          )} - ${formatNumber(rocAuc)}`,
          type: "line",
          color: "orange",
        },
        {
          data: yValuesValidation,
          name: commonT("neuralNetworks.chart.dataLegendValidation"),
          type: "line",
          color: "black",
        },

        {
          data: validationLinePoints,
          name: `${commonT(
            "neuralNetworks.chart.rocAucValidation"
          )} - ${formatNumber(rocAucValidation)}`,
          type: "line",
          color: "red",
        },
      ],

      options: {
        title: commonT("neuralNetworks.chart.rocTitle"),
        xAxisTitle: commonT("neuralNetworks.chart.axisX"),
        yAxisTitle: commonT("neuralNetworks.chart.axisY"),
        titleFontSize: 24,
        chartName: responseColumn,
        legendEnabled: true,
      },
      type: "line",
      displayName: commonT("neuralNetworks.chart.rocTitle"),
    };

    chartsForVariable["learningCurve"] = {
      seriesData: [
        {
          data: testScoresMean,
          name: commonT("neuralNetworks.chart.testScoresMean"),
          type: "line",
        },
        {
          data: trainScoresMean,
          name: commonT("neuralNetworks.chart.trainScoresMean"),
          type: "line",
        },
      ],
      options: {
        title: commonT("neuralNetworks.chart.learningCurveTitle"),
        xAxisTitle: commonT("neuralNetworks.chart.trainSize"),
        titleFontSize: 24,
        chartName: responseColumn,
        legendEnabled: true,
      },

      type: "line",
      displayName: commonT("neuralNetworks.chart.learningCurveTitle"),
    };

    chartsForVariable["overlay"] = {
      seriesData: [
        {
          data: yValues,
          name: commonT("widget.chartType.y"),
          type: "scatter",
        },
        {
          data: yPredictedValues,
          name: commonT("widget.chartType.yPredicted"),
          type: "scatter",
        },
      ],
      options: {
        title: commonT("widget.chartType.overlayPlot"),
        xAxisTitle: commonT("widget.chartType.rowNumber"),
        yAxisTitle: commonT("widget.chartType.y"),
        titleFontSize: 24,
        chartName: responseColumn,
        legendEnabled: true,
      },
      type: "scatter",
      displayName: commonT("widget.chartType.overlayPlot"),
    };

    return chartsForVariable;
  }

  return (
    <>
      <ContentHeader
        title={`${commonT("neuralNetworks.title")} - ${commonT(
          "neuralNetworks.hodout"
        )}`}
        tool={"neuralNetworks"}
      />

      {loadingPage ? (
        <Spin />
      ) : (
        <div id="content">
          <Row>
            <Col span={12}>
              <div style={{ marginBottom: 20 }}>
                <ChartHub
                  chartConfigurations={chartData}
                  tool="neuralNetworks"
                  showLimits
                />
              </div>

              <div style={{ marginBottom: 20 }}>
                <ChartHub
                  chartConfigurations={neuralDiagram}
                  tool="neuralNetworks"
                />
              </div>

              {isCategoric && (
                <>
                  <div style={{ marginBottom: 20 }}>
                    <Row>
                      <Col span={12}>
                        <Table
                          dataSource={
                            confusionMatrixTraining.current.datasource
                          }
                          columns={confusionMatrixTraining.current.columns}
                          loading={false}
                          title={commonT(
                            "neuralNetworks.confusionMatrixTraining"
                          )}
                        />
                      </Col>
                      <Col offset={1} span={11}>
                        <Table
                          dataSource={
                            confusionMatrixValidation.current.datasource
                          }
                          columns={confusionMatrixValidation.current.columns}
                          loading={false}
                          title={commonT(
                            "neuralNetworks.confusionMatrixValidation"
                          )}
                        />
                      </Col>
                    </Row>
                  </div>
                </>
              )}
              <div style={{ marginBottom: 20 }}>
                <InformationTable
                  r2Train={rSquare.r2}
                  r2Validation={rSquare.r2Validation}
                />
              </div>
            </Col>
            <Col offset={1} span={11}>
              <div style={{ marginBottom: 20 }}>
                <Table
                  dataSource={summaryTable.current.datasource}
                  columns={summaryTable.current.columns}
                  loading={loadingSummary}
                  title={commonT("neuralNetworks.summary")}
                />
              </div>
              <div style={{ marginBottom: 20 }}>
                <Table
                  dataSource={parameterEstimates.current.datasource}
                  columns={parameterEstimates.current.columns}
                  loading={loadingParameterEstimates}
                  title={commonT("neuralNetworks.parameterEstimates")}
                />
              </div>
            </Col>
          </Row>
          <Row style={{ marginBottom: 20 }}>
            <Col span={12}>
              <Equation
                tooltipTitle={equation}
                equation={equation}
                width={0}
                loading={false}
              />
            </Col>
          </Row>
          <Row>
            <Col span={24}>
              {loadingProfiler ? (
                <Spin />
              ) : (
                <Profiler
                  profilerData={profilerData}
                  optimizationData={profilerData}
                  loading={false}
                  type={"neuralNetwork"}
                  defaultColNumber={colNumber}
                />
              )}
            </Col>
          </Row>
        </div>
      )}
    </>
  );
};

import React, { useEffect, useRef, useState } from "react";
import { useAuth } from "hooks/useAuth";
import { useTranslation } from "next-i18next";
import axios from "axios";
import ContentHeader from "shared/contentHeader";
import { Spin } from "shared/spin";
import { Col, Row, message, notification } from "antd";
import {
  DataToCompile,
  NeuralNetworkDiagramLayer,
  NeuralNetworkKFoldData,
  validateRSquare,
} from "../interface";
import Table from "shared/table";
import { formatNumber } from "utils/formatting";
import {
  createConfusionMatrix,
  createParameterEstimatesTable,
  createSummaryTable,
} from "../table";
import { dataSortOrder } from "utils/core";
import { useRouter } from "next/router";
import dynamic from "next/dynamic";
import InformationTable from "../informationTable";
import Equation from "shared/equation";
import {
  HighChartCustomSeries,
  HighChartTemplate,
} from "shared/widget/chartHub/interface";
import { UniqueAnalysisManagementResponse } from "components/analysisManagement/interface";

const ChartHub = dynamic(() => import("shared/widget/chartHub"), {
  ssr: false,
  loading: () => <Spin />,
});

const Profiler = dynamic(() => import("shared/profiler"), {
  ssr: false,
  loading: () => <Spin />,
});

const fetcher = axios.create({
  baseURL: "/api",
});

export const NeuralNetworksKfold: React.FC = () => {
  const { t: commonT } = useTranslation("common", { useSuspense: false });
  const { user } = useAuth();
  const router = useRouter();

  const [loadingPage, setLoadingPage] = useState(false);
  const [loadingSummary, setLoadingSummary] = useState(true);
  const [loadingParameterEstimates, setLoadingParameterEstimates] =
    useState(true);
  const [loadingEquation, setLoadingEquation] = useState(true);
  const [loadingProfiler, setLoadingProfiler] = useState(true);

  const [isCategoric, setIsCategoric] = useState(false);
  const [equation, setEquation] = useState<string>("");

  const [chartData, setChartData] = useState<HighChartTemplate>({});
  const [neuralDiagram, setNeuralDiagram] = useState<HighChartTemplate>({});

  const [profilerData, setProfilerData] = useState<Record<string, any>>({});

  const [colNumber, setColNumber] = useState(3);

  const [rSquare, setRSquare] = useState<validateRSquare>({
    r2: undefined,
    r2Validation: undefined,
  });

  const summaryTable = useRef<any>({});
  const parameterEstimates = useRef<any>({});
  const confusionMatrixTraining = useRef<any>({});
  const confusionMatrixValidation = useRef<any>({});

  useEffect(() => {
    const getData = async () => {
      const parsedUrlQuery = router?.query;
      if (Object.keys(parsedUrlQuery).length > 0) {
        const id = parsedUrlQuery.id as string;

        if (id) {
          try {
            const { data: analysisManagementResponse } =
              await fetcher.post<UniqueAnalysisManagementResponse>(
                "neuralNetworks/mongodb/getQueueId",
                { id }
              );
            setLoadingPage(false);

            const data = analysisManagementResponse.data
              .result as NeuralNetworkKFoldData;
            const payload = analysisManagementResponse.data.payload;

            const variables = Object.keys(payload.inputData);
            const dataToSend: DataToCompile = {
              obj: payload.inputData,
              itens: variables,
            };

            const responseColumnVariable = variables.at(-1);

            setIsCategoric(data.isCategoric);
            setColNumber(Object.keys(dataToSend.obj).length - 1);

            const summaryTranslate = {
              term: commonT("neuralNetworks.terms"),
              accuracy: commonT("neuralNetworks.accuracy"),
              fScore: commonT("neuralNetworks.fScore"),
              mean: commonT("neuralNetworks.mean"),
              precision: commonT("neuralNetworks.precision"),
              recall: commonT("neuralNetworks.recall"),
              rocAuc: commonT("neuralNetworks.rocAuc"),
              stdDev: commonT("neuralNetworks.stdDev"),
              r2: commonT("neuralNetworks.r2"),
              rmse: commonT("neuralNetworks.rmse"),
              training: commonT("neuralNetworks.training"),
              validation: commonT("neuralNetworks.validation"),
            };

            summaryTable.current = createSummaryTable(
              data.models,
              summaryTranslate,
              data.isCategoric
            );
            setLoadingSummary(false);

            const parameterEstimateTransalte = {
              importance: commonT("neuralNetworks.importance"),
              term: commonT("neuralNetworks.terms"),
            };

            parameterEstimates.current = createParameterEstimatesTable(
              data.featureImportance,
              parameterEstimateTransalte
            );

            const paretoPlotData = data.featureImportance;
            delete paretoPlotData["Intercept"];

            setRSquare({
              r2: data.isCategoric
                ? data.models.training.accuracy
                : data.models.training.r2,
              r2Validation: data.isCategoric
                ? data.models.validation.accuracy
                : data.models.validation.r2,
            });

            setLoadingParameterEstimates(false);

            const estimatesList = Object.entries(paretoPlotData).reduce(
              (acc: any, [key, value]) => {
                acc[
                  key
                    .replace("num__", "")
                    .replace("onehot__", "")
                    .replace("remainder__", "")
                ] = value;
                return acc;
              },
              {}
            );

            const sortedList: { [key: string]: number } = dataSortOrder(
              estimatesList,
              "desc",
              true
            );

            const columnList = Object.entries(sortedList).map(([key]) => key);

            const sortedEstimates: number[] = Object.entries(sortedList)
              .map(([, value]) => value)
              .sort((a, b) => a - b);

            if (data.isCategoric) {
              const confusionMatrixTranslate = {
                term: commonT("neuralNetworks.terms"),
              };

              confusionMatrixTraining.current = createConfusionMatrix(
                data.models.training.confusionMatrix,
                confusionMatrixTranslate
              );

              confusionMatrixValidation.current = createConfusionMatrix(
                data.models.validation.confusionMatrix,
                confusionMatrixTranslate
              );
            }

            // equaçao
            let newEquation = "";
            if (data.isCategoric) {
              const neuronToLabel = data.categoricNeurons;
              const neuronToEquation = data.uniqueEquation;

              const formattedEquations = Object.entries(neuronToLabel)
                .map(([neuron, label]) => {
                  const equation = neuronToEquation[neuron];
                  return `(${label}) => ${equation}`;
                })
                .join("\n");

              newEquation = formattedEquations;
            } else {
              const lastKey = Object.keys(data.equations).pop();
              const lastValue = lastKey ? data.equations[lastKey] : undefined;
              newEquation = lastValue;
            }

            setEquation(`Y = ${newEquation}`);
            setLoadingEquation(false);

            const profilerObj: Record<string, any> = {};

            for (const key of Object.keys(data.uniqueEquation)) {
              const equationNeuron = data.uniqueEquation[key];

              profilerObj[key] = [];
              profilerObj[key]["maxValue"] = data.isCategoric
                ? 1
                : Math.max(...data.yPredictedValues);
              profilerObj[key]["minValue"] = data.isCategoric
                ? 0
                : Math.min(...data.yPredictedValues);

              profilerObj[key]["equation"] = equationNeuron;
              profilerObj[key]["equationOrigin"] = equationNeuron;
              profilerObj[key]["responseTitle"] = responseColumnVariable;

              profilerObj[key]["data"] = {
                [key]: dataToSend.obj,
              };

              profilerObj[key]["categoricNeurons"] = data.categoricNeurons;
              profilerObj[key]["categoricColumns"] = data.categoricColumns;

              profilerObj[key]["bounds"] = {};

              const categoricMean = data.categoricMean;

              for (const colKey of Object.keys(dataToSend.obj).filter(
                (el) => el !== responseColumnVariable
              )) {
                const items = dataToSend.obj[colKey];

                if (typeof items[0] === "string") {
                  for (const meanKey of Object.keys(categoricMean)) {
                    if (meanKey.startsWith(`${colKey}[`)) {
                      profilerObj[key]["bounds"][meanKey] = [0, 1];
                    }
                  }
                } else {
                  const min = Math.min(...items);
                  const max = Math.max(...items);
                  profilerObj[key]["bounds"][colKey] = [min, max];
                }
              }
            }

            setProfilerData(profilerObj);

            setLoadingProfiler(false);

            notification.success({
              message: commonT("neuralNetworks.proccesSuccess"),
            });

            await generateChart(
              data,
              columnList,
              sortedEstimates,
              responseColumnVariable
            );
          } catch (error) {
            console.error(error);
            message.error({
              content: commonT("error.general.unexpectedMsg"),
            });
          }
        }
      }
    };
    getData().catch(console.error);
  }, [router.isReady, router.query]);

  useEffect(() => {
    const countAccess = async () => {
      if (user) {
        fetcher.post("access_counter", {
          user: user?.username,
          operation: "neuralNetworks",
        });
      }
    };
    countAccess().catch(console.error);
  }, []);

  async function generateChart(
    data: NeuralNetworkKFoldData,
    columnList: string[],
    sortedEstimates: number[],
    responseColumnVariable: string
  ) {
    const trainingRocCurve = data.models.training.rocCurve;
    const validationRocCurve = data.models.validation.rocCurve;

    const trainingLinePoints = Array.from(
      { length: trainingRocCurve.length + 1 },
      (_, index) => {
        return index / trainingRocCurve.length;
      }
    );

    const validationLinePoints = Array.from(
      { length: validationRocCurve.length + 1 },
      (_, index) => {
        return index / validationRocCurve.length;
      }
    );

    const yValuesTraining = data.models.training.rocCurve.map((el) => el.y);
    const yValuesValidation = data.models.validation.rocCurve.map((el) => el.y);
    const testScoresMean = data.learningCurve.testScoresMean;
    const trainScoresMean = data.learningCurve.trainScoresMean;

    const rocAuc = data.models.training.rocAuc;
    const rocAucValidation = data.models.validation.rocAuc;

    const chartForVariable = buildChart(
      columnList,
      sortedEstimates,
      yValuesTraining,
      trainingLinePoints,
      yValuesValidation,
      validationLinePoints,
      testScoresMean,
      trainScoresMean,
      data.yValues,
      data.yPredictedValues,
      rocAuc,
      rocAucValidation,
      responseColumnVariable
    );

    setChartData(chartForVariable);

    const neuralDiagramData = buildNeuralDiagram(data.diagramData.layers);
    setNeuralDiagram(neuralDiagramData);
  }

  function buildNeuralDiagram(layers: NeuralNetworkDiagramLayer[]) {
    const chartsForVariable: HighChartTemplate = {};

    const diagramSeries: HighChartCustomSeries = {
      type: "neuralNetworkDiagram",
      name: commonT("neuralNetworks.neuralDiagramTitle"),
      layers: layers,
      data: [],
    };

    chartsForVariable["neural"] = {
      type: "neuralNetworkDiagram",
      seriesData: [diagramSeries],
      options: {
        chartName: "Y",
        title: commonT("neuralNetworks.neuralDiagramTitle"),
        xAxisTitle: commonT("neuralNetworks.layer"),
        yAxisTitle: commonT("neuralNetworks.neurons"),
      },
    };

    return chartsForVariable;
  }

  function buildChart(
    columnList: string[],
    sortedEstimates: number[],
    yValuesTraining: any,
    trainingLinePoints: any,
    yValuesValidation: any,
    validationLinePoints: any,
    testScoresMean: any,
    trainScoresMean: any,
    yValues: any,
    yPredictedValues: any,
    rocAuc: any,
    rocAucValidation: any,
    responseColumn: string
  ) {
    const chartsForVariable: HighChartTemplate = {};

    chartsForVariable["estimates"] = {
      seriesData: [
        {
          data: sortedEstimates,
          name: commonT("neuralNetworks.chart.dataLegend"),
          type: "bar",
        },
      ],
      options: {
        title: commonT("neuralNetworks.chart.titleChart"),
        titleFontSize: 24,
        chartName: responseColumn,
        legendEnabled: true,
        useVerticalLimits: false,
        categories: columnList,
        barIsVertical: false,
      },
      type: "bar",
      displayName: commonT("neuralNetworks.chart.titleChart"),
    };

    chartsForVariable["roc"] = {
      seriesData: [
        {
          data: yValuesTraining,
          name: commonT("neuralNetworks.chart.dataLegendTraining"),
          type: "line",
        },
        {
          data: trainingLinePoints,
          name: `${commonT(
            "neuralNetworks.chart.rocAucTraining"
          )} - ${formatNumber(rocAuc)}`,
          type: "line",
          color: "orange",
        },
        {
          data: yValuesValidation,
          name: commonT("neuralNetworks.chart.dataLegendValidation"),
          type: "line",
          color: "black",
        },
        {
          data: validationLinePoints,
          name: `${commonT(
            "neuralNetworks.chart.rocAucValidation"
          )} - ${formatNumber(rocAucValidation)}`,
          type: "line",
          color: "red",
        },
      ],

      options: {
        title: commonT("neuralNetworks.chart.rocTitle"),
        xAxisTitle: commonT("neuralNetworks.chart.axisX"),
        yAxisTitle: commonT("neuralNetworks.chart.axisY"),
        titleFontSize: 24,
        chartName: responseColumn,
        legendEnabled: true,
      },
      type: "line",
      displayName: commonT("neuralNetworks.chart.rocTitle"),
    };

    chartsForVariable["learningCurve"] = {
      seriesData: [
        {
          data: testScoresMean,
          name: commonT("neuralNetworks.chart.testScoresMean"),
          type: "line",
        },
        {
          data: trainScoresMean,
          name: commonT("neuralNetworks.chart.trainScoresMean"),
          type: "line",
        },
      ],
      options: {
        title: commonT("neuralNetworks.chart.learningCurveTitle"),
        xAxisTitle: commonT("neuralNetworks.chart.trainSize"),
        titleFontSize: 24,
        chartName: responseColumn,
        legendEnabled: true,
      },

      type: "line",
      displayName: commonT("neuralNetworks.chart.learningCurveTitle"),
    };

    chartsForVariable["overlay"] = {
      seriesData: [
        {
          data: yValues,
          name: commonT("widget.chartType.y"),
          type: "scatter",
        },
        {
          data: yPredictedValues,
          name: commonT("widget.chartType.yPredicted"),
          type: "scatter",
        },
      ],
      options: {
        title: commonT("widget.chartType.overlayPlot"),
        xAxisTitle: commonT("widget.chartType.rowNumber"),
        yAxisTitle: commonT("widget.chartType.y"),
        titleFontSize: 24,
        chartName: responseColumn,
        legendEnabled: true,
      },
      type: "scatter",
      displayName: commonT("widget.chartType.overlayPlot"),
    };

    return chartsForVariable;
  }

  return (
    <>
      <ContentHeader
        title={`${commonT("neuralNetworks.title")} - ${commonT(
          "neuralNetworks.kfold"
        )}`}
        tool={"neuralNetworks"}
      />

      {loadingPage ? (
        <Spin />
      ) : (
        <div id="content">
          <Row>
            <Col xs={24} sm={24} md={24} lg={24} xl={24} xxl={12}>
              <div style={{ marginBottom: 20 }}>
                <ChartHub
                  chartConfigurations={chartData}
                  tool="neuralNetworks"
                  showLimits
                />
              </div>
            </Col>
            <Col
              xs={24}
              sm={24}
              md={24}
              lg={24}
              xl={24}
              xxl={{ offset: 1, span: 11 }}
            >
              <div style={{ marginBottom: 20 }}>
                <Table
                  dataSource={summaryTable.current.datasource}
                  columns={summaryTable.current.columns}
                  loading={loadingSummary}
                  title={commonT("neuralNetworks.summary")}
                />
              </div>
            </Col>
          </Row>

          <Row>
            <Col xs={24} sm={24} md={24} lg={24} xl={24} xxl={12}>
              <div style={{ marginBottom: 20 }}>
                <ChartHub
                  chartConfigurations={neuralDiagram}
                  tool="neuralNetworks"
                />
              </div>
            </Col>
            <Col
              xs={24}
              sm={24}
              md={24}
              lg={24}
              xl={24}
              xxl={{ offset: 1, span: 11 }}
            >
              <div style={{ marginBottom: 20 }}>
                <Table
                  dataSource={parameterEstimates.current.datasource}
                  columns={parameterEstimates.current.columns}
                  loading={loadingParameterEstimates}
                  title={commonT("neuralNetworks.parameterEstimates")}
                />
              </div>
            </Col>
          </Row>

          {isCategoric && (
            <>
              <div style={{ marginBottom: 20 }}>
                <Row>
                  <Col xs={24} sm={24} md={24} lg={24} xl={24} xxl={12}>
                    <Table
                      dataSource={confusionMatrixTraining.current.datasource}
                      columns={confusionMatrixTraining.current.columns}
                      loading={false}
                      title={commonT("neuralNetworks.confusionMatrixTraining")}
                    />
                  </Col>
                  <Col
                    xs={24}
                    sm={24}
                    md={24}
                    lg={24}
                    xl={24}
                    xxl={{ offset: 1, span: 11 }}
                  >
                    <Table
                      dataSource={confusionMatrixValidation.current.datasource}
                      columns={confusionMatrixValidation.current.columns}
                      loading={false}
                      title={commonT(
                        "neuralNetworks.confusionMatrixValidation"
                      )}
                    />
                  </Col>
                </Row>
              </div>
            </>
          )}
          <Row style={{ marginBottom: 20 }}>
            <Col xs={24} sm={24} md={24} lg={24} xl={24} xxl={12}>
              <div style={{ marginBottom: 20 }}>
                <InformationTable
                  r2Train={rSquare.r2}
                  r2Validation={rSquare.r2Validation}
                />
              </div>
            </Col>
            <Col
              xs={24}
              sm={24}
              md={24}
              lg={24}
              xl={24}
              xxl={{ offset: 1, span: 11 }}
            >
              <Equation
                tooltipTitle={equation}
                equation={equation}
                width={0}
                loading={loadingEquation}
              />
            </Col>
          </Row>
          <Row>
            <Col xs={24} sm={24} md={24} lg={24} xl={24} xxl={24}>
              {loadingProfiler ? (
                <Spin />
              ) : (
                <Profiler
                  profilerData={profilerData}
                  optimizationData={profilerData}
                  loading={false}
                  type={"neuralNetwork"}
                  defaultColNumber={colNumber}
                />
              )}
            </Col>
          </Row>
        </div>
      )}
    </>
  );
};

import { useTranslation } from "next-i18next";
import React from "react";
import Table from "shared/table";

interface R2TableProps {
  r2Train: number;
  r2Validation: number;
}

const InformationTable: React.FC<R2TableProps> = ({
  r2Train,
  r2Validation,
}) => {
  const { t: commonT } = useTranslation("common", { useSuspense: false });

  const getInterpretation = (r2: number) => {
    if (!r2) return "";
    if (r2 >= 0.8) return commonT("neuralNetworks.informationTable.veryGood");
    if (r2 >= 0.6) return commonT("neuralNetworks.informationTable.acceptable");
    if (r2 >= 0.4) return commonT("neuralNetworks.informationTable.bad");
    return commonT("neuralNetworks.informationTable.bad");
  };

  const getScenario = (r2Train: number, r2Validation: number) => {
    if (!r2Train || !r2Validation) return "";

    // Overfitting
    if (r2Train > 0.9 && r2Validation < 0.5)
      return commonT("neuralNetworks.informationTable.goodTrain");

    // Bom ajuste
    if (r2Train >= 0.7 && r2Validation >= 0.7)
      return commonT("neuralNetworks.informationTable.goodAdjust");

    // Não aprendeu bem
    if (r2Train <= 0.7 && r2Validation <= 0.7)
      return commonT("neuralNetworks.informationTable.notLearn");

    // Dados ruidosos
    if (r2Train <= 0.5 && r2Validation <= 0.5)
      return commonT("neuralNetworks.informationTable.noisy");

    return commonT("neuralNetworks.informationTable.noisy");
  };

  const dataSource = [
    {
      key: "1",
      type: commonT("neuralNetworks.informationTable.training"),
      r2: r2Train?.toFixed(3),
      interpretation: getInterpretation(r2Train),
      scenario: getScenario(r2Train, r2Validation),
    },
    {
      key: "2",
      type: commonT("neuralNetworks.informationTable.validation"),
      r2: r2Validation?.toFixed(3),
      interpretation: getInterpretation(r2Validation),
      scenario: getScenario(r2Train, r2Validation),
    },
  ];

  const columns = [
    {
      title: commonT("neuralNetworks.informationTable.type"),
      dataIndex: "type",
      key: "type",
    },
    {
      title: commonT("neuralNetworks.informationTable.scenario"),
      dataIndex: "scenario",
      key: "scenario",
    },
    {
      title: "R²",
      dataIndex: "r2",
      key: "r2",
    },
    {
      title: commonT("neuralNetworks.informationTable.interpretation"),
      dataIndex: "interpretation",
      key: "interpretation",
    },
  ];

  return (
    <Table
      dataSource={dataSource}
      columns={columns}
      title={commonT("neuralNetworks.informationTable.title")}
      loading={r2Train === undefined && r2Validation === undefined}
    />
  );
};

export default InformationTable;
import {
  NeuralNetworkModels,
  ParameterEstimateGridRow,
} from "components/neuralNetworks/interface";
import { P_VALUE_NOT_REPLICATED_LIMIT } from "utils/constant";

interface DataType {
  source: string;
  [key: string]: number | string;
}

export function createSummaryTable(
  models: NeuralNetworkModels,
  translate: any,
  isCategoric: boolean
) {
  const columns: any[] = [
    {
      title: translate.term,
      dataIndex: "source",
      key: "source",
    },
  ];

  if (isCategoric) {
    columns.push(
      {
        title: translate.accuracy,
        dataIndex: "accuracy",
        key: "accuracy",
      },
      {
        title: translate.fScore,
        dataIndex: "fScore",
        key: "fScore",
      },
      {
        title: translate.precision,
        dataIndex: "precision",
        key: "precision",
      },
      {
        title: translate.recall,
        dataIndex: "recall",
        key: "recall",
      },
      {
        title: translate.rocAuc,
        dataIndex: "rocAuc",
        key: "rocAuc",
      }
    );
  } else {
    columns.push(
      {
        title: translate.rmse,
        dataIndex: "meanSquaredError",
        key: "meanSquaredError",
      },
      {
        title: translate.r2,
        dataIndex: "r2",
        key: "r2",
      },
      {
        title: translate.mean,
        dataIndex: "mean",
        key: "mean",
      },
      {
        title: translate.stdDev,
        dataIndex: "stdDev",
        key: "stdDev",
      }
    );
  }

  const datasource = [];

  if (isCategoric) {
    datasource.push(
      {
        source: Object.keys(models)[0] === "training" ? translate.training : "",
        accuracy: models.training.accuracy,
        fScore: models.training.fScore,
        mean: models.training.mean,
        precision: models.training.precision,
        recall: models.training.recall,
        rocAuc: models.training.rocAuc,
        stdDev: models.training.stdDev,
      },
      {
        source:
          Object.keys(models)[1] === "validation" ? translate.validation : "",
        accuracy: models.validation.accuracy,
        fScore: models.validation.fScore,
        mean: models.validation.mean,
        precision: models.validation.precision,
        recall: models.validation.recall,
        rocAuc: models.validation.rocAuc,
        stdDev: models.validation.stdDev,
      }
    );
  } else {
    datasource.push(
      {
        source: Object.keys(models)[0] === "training" ? translate.training : "",
        mean: models.training.mean,
        stdDev: models.training.stdDev,
        r2: models.training.r2,
        meanSquaredError: models.training.meanSquaredError,
      },
      {
        source:
          Object.keys(models)[1] === "validation" ? translate.validation : "",
        mean: models.validation.mean,
        stdDev: models.validation.stdDev,
        r2: models.validation.r2,
        meanSquaredError: models.validation.meanSquaredError,
      }
    );
  }

  return { columns, datasource };
}

export const createParameterEstimatesTable = (
  featureImportance: Record<string, number>,
  translate: any
) => {
  const terms = Object.keys(featureImportance).map((el) =>
    el.replace("num__", "").replace("onehot__", "").replace("remainder__", "")
  );

  const columns = [
    {
      title: translate.term,
      dataIndex: "source",
      key: "source",
      filters: terms.map((term) => ({ text: term, value: term })),
    },
    {
      title: translate.importance,
      dataIndex: "importance",
      key: "importance",
      onCell: (item: any) => {
        return {
          ["style"]: {
            color:
              Math.abs(Number(item.importance)) > P_VALUE_NOT_REPLICATED_LIMIT
                ? "red"
                : "",
          },
        };
      },
    },
  ];

  const datasource: ParameterEstimateGridRow[] = [];

  Object.keys(featureImportance).map((key) => {
    const row: ParameterEstimateGridRow = {
      source: key
        .replace("num__", "")
        .replace("onehot__", "")
        .replace("remainder__", ""),
      importance: Math.abs(featureImportance[key]),
    };

    datasource.push(row);
  });

  return { columns, datasource };
};

export const createConfusionMatrix = (
  confusionMatrix: Record<string, number[]> | undefined,
  translate: any
) => {
  if (!confusionMatrix) return;
  const terms = Object.keys(confusionMatrix);

  const columns = [
    {
      title: translate.term,
      dataIndex: "source",
      key: "source",
    },
  ];

  terms.forEach((el) => {
    const row = {
      title: el,
      dataIndex: el,
      key: el,
      align: "center",
    };
    columns.push(row as any);
  });

  const datasource: DataType[] = terms.map((term) => {
    const rowData: DataType = { source: term };
    terms.forEach((el, i) => {
      rowData[el] = confusionMatrix[term][i];
    });
    return rowData;
  });

  return { columns, datasource };
};

import {
  WidgetLabelsDisabled,
  WidgetFontDisabled,
  WidgetGeneralDisabled,
  WidgetColorDisable,
} from "shared/widget/widget/interface";

export const widgeLabelDisabled: WidgetLabelsDisabled = {
  disabledAxisX: true,
  disabledAxisY: true,
  disabledTitle: true,
  disabledEnableAxisY: true,
  disabledEnableAxisX: true,
  disabledDataLabel: true,
};

export const widgetFontDisabled: WidgetFontDisabled = {
  disabledAxisX: true,
  disabledAxisY: true,
};

export const widgetGeneralDisabled: WidgetGeneralDisabled = {
  disabledMaxCharacterLimitLegend: true,
  disabledPointerStyle: true,
  disabledPointerStyleSize: true,
  disabledAxisBeginAtZero: true,
  disabledXMaxScale: true,
  disabledXMinScale: true,
  disabledYMaxScale: true,
  disabledYMinScale: true,
  disabledOrderColumn: true,
};

export const widgetColorDisabledChartBar: WidgetColorDisable = {
  modelLineColor: true,
  groupMeansColor: true,
  lieColor: true,
  lseColor: true,
  subGroupMeansColor: true,
  pseLineColor: true,
  stdErrorLineColor: true,
};
