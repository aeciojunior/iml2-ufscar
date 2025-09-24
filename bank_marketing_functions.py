"""
IML 1.2 - Funções Auxiliares para Análise de Marketing Bancário

Este módulo contém funções auxiliares para o projeto de classificação
de aquisição de produtos bancários.

"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_preprocess_data(file_path):
    """
    Carrega e faz pré-processamento básico dos dados
    
    Args:
        file_path (str): Caminho para o arquivo CSV
        
    Returns:
        tuple: (df_processed, feature_columns, label_encoders)
    """
    # Carregar dados
    df = pd.read_csv(file_path, sep=';')
    
    # Criar cópia para processamento
    df_processed = df.copy()
    
    # Tratamento de valores 'unknown'
    df_processed['default'] = df_processed['default'].replace('unknown', 'no')
    df_processed['education'] = df_processed['education'].replace('unknown', 'basic.4y')
    
    # Variáveis categóricas para encoding
    categorical_vars_to_encode = ['job', 'marital', 'education', 'contact', 'month', 'day_of_week', 'poutcome']
    
    # Aplicar Label Encoding
    label_encoders = {}
    for var in categorical_vars_to_encode:
        le = LabelEncoder()
        df_processed[var + '_encoded'] = le.fit_transform(df_processed[var])
        label_encoders[var] = le
    
    # Converter variável alvo para numérica
    df_processed['y_numeric'] = df_processed['y'].map({'yes': 1, 'no': 0})
    
    # Definir features
    numeric_vars = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
                    'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    binary_vars = ['default', 'housing', 'loan']
    
    feature_columns = (numeric_vars + 
                      [var + '_encoded' for var in categorical_vars_to_encode] + 
                      binary_vars)
    
    return df_processed, feature_columns, label_encoders


def evaluate_model_comprehensive(model, X_train, X_test, y_train, y_test, model_name):
    """
    Avalia um modelo com múltiplas métricas
    
    Args:
        model: Modelo treinado
        X_train, X_test: Dados de treino e teste
        y_train, y_test: Labels de treino e teste
        model_name (str): Nome do modelo
        
    Returns:
        dict: Dicionário com métricas
    """
    # Treinar modelo
    model.fit(X_train, y_train)
    
    # Previsões
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'Modelo': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_pred_proba),
        'AUC-PR': average_precision_score(y_test, y_pred_proba)
    }
    
    return metrics, y_pred, y_pred_proba


def plot_model_comparison(metrics_df):
    """
    Cria gráficos comparativos dos modelos
    
    Args:
        metrics_df (DataFrame): DataFrame com métricas dos modelos
    """
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'AUC-PR']
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'plum', 'orange']
    
    plt.figure(figsize=(15, 10))
    
    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 3, i+1)
        bars = plt.bar(metrics_df['Modelo'], metrics_df[metric], color=colors[i])
        plt.title(f'{metric}')
        plt.ylabel(metric)
        plt.xticks(rotation=45)
        
        # Adicionar valores nas barras
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def plot_roc_curves(models_data, y_test):
    """
    Plota curvas ROC comparativas
    
    Args:
        models_data (list): Lista de tuplas (y_pred_proba, model_name, auc_score)
        y_test: Labels de teste
    """
    plt.figure(figsize=(10, 8))
    
    for y_pred_proba, model_name, auc_score in models_data:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curvas ROC Comparativas')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def predict_new_user(user_data, model, feature_columns, label_encoders, scaler=None, use_scaled=False):
    """
    Faz previsão para um novo usuário
    
    Args:
        user_data (dict): Dados do usuário
        model: Modelo treinado
        feature_columns (list): Lista de features
        label_encoders (dict): Encoders das variáveis categóricas
        scaler: Scaler usado (se necessário)
        use_scaled (bool): Se deve usar dados normalizados
        
    Returns:
        tuple: (prediction, probability)
    """
    # Criar DataFrame com os dados do usuário
    user_df = pd.DataFrame([user_data])
    
    # Aplicar o mesmo tratamento dos dados originais
    if 'default' in user_df.columns and user_df['default'].iloc[0] == 'unknown':
        user_df['default'] = 'no'
    if 'education' in user_df.columns and user_df['education'].iloc[0] == 'unknown':
        user_df['education'] = 'basic.4y'
    
    # Encoding das variáveis categóricas
    categorical_vars_to_encode = ['job', 'marital', 'education', 'contact', 'month', 'day_of_week', 'poutcome']
    for var in categorical_vars_to_encode:
        if var in user_df.columns:
            user_df[var + '_encoded'] = label_encoders[var].transform(user_df[var])
    
    # Selecionar as features na mesma ordem
    user_features = user_df[feature_columns]
    
    # Normalizar se necessário
    if use_scaled:
        user_features = scaler.transform(user_features)
    
    # Fazer previsão
    prediction = model.predict(user_features)[0]
    probability = model.predict_proba(user_features)[0][1]
    
    return prediction, probability


def get_business_recommendation(probability, threshold=0.5):
    """
    Retorna recomendação de negócio baseada na probabilidade
    
    Args:
        probability (float): Probabilidade de subscrever
        threshold (float): Threshold para decisão
        
    Returns:
        str: Recomendação de negócio
    """
    if probability > threshold:
        return {
            'action': 'Priorizar cliente',
            'reason': 'Alta probabilidade de subscrever',
            'suggestion': 'Focar recursos de marketing neste cliente'
        }
    else:
        return {
            'action': 'Baixa prioridade',
            'reason': 'Baixa probabilidade de subscrever',
            'suggestion': 'Focar em outros clientes com maior potencial'
        }


# Exemplo de uso das funções
if __name__ == "__main__":
    print("Módulo de funções auxiliares para análise de marketing bancário")
    print("Use as funções importando este módulo no seu notebook ou script principal")
