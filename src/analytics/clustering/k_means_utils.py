"""
K-Means Clustering Utilities
Análise de agrupamento usando K-Means
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def calculate_elbow_method(X, max_k=10):
    """
    Calcula inércia para diferentes valores de K (método do cotovelo)
    
    Args:
        X: Dados para clustering (já normalizados)
        max_k: Número máximo de clusters a testar
    
    Returns:
        dict com inércias e silhouette scores por K
    """
    inertias = []
    silhouette_scores = []
    k_values = range(2, min(max_k + 1, len(X)))
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        
        # Silhouette score (quanto maior, melhor)
        if k > 1:
            sil_score = silhouette_score(X, labels)
            silhouette_scores.append(sil_score)
        else:
            silhouette_scores.append(0)
    
    return {
        'k_values': list(k_values),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores
    }


def suggest_optimal_k(elbow_data):
    """
    Sugere K ótimo baseado no método do cotovelo e silhouette
    
    Args:
        elbow_data: Resultado do calculate_elbow_method
    
    Returns:
        int: K sugerido
    """
    k_values = elbow_data['k_values']
    silhouette_scores = elbow_data['silhouette_scores']
    
    # Sugestão: K com maior silhouette score
    if silhouette_scores:
        best_idx = np.argmax(silhouette_scores)
        return k_values[best_idx]
    
    return 3  # Default


def perform_kmeans(data, feature_columns, n_clusters=3, scale=True):
    """
    Executa K-Means clustering
    
    Args:
        data: DataFrame com os dados
        feature_columns: Lista de colunas para usar no clustering
        n_clusters: Número de clusters
        scale: Se deve normalizar os dados
    
    Returns:
        dict com resultados do clustering
    """
    # Preparar dados
    X = data[feature_columns].values
    
    # Verificar dados válidos
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        raise ValueError("Dados contêm valores NaN ou infinitos. Remova ou trate esses valores.")
    
    # Normalizar se necessário
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # Executar K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Métricas de qualidade
    inertia = kmeans.inertia_
    
    metrics = {
        'inertia': inertia,
        'silhouette': None,
        'davies_bouldin': None,
        'calinski_harabasz': None
    }
    
    # Calcular métricas se houver mais de 1 cluster
    if n_clusters > 1 and len(np.unique(labels)) > 1:
        metrics['silhouette'] = silhouette_score(X_scaled, labels)
        metrics['davies_bouldin'] = davies_bouldin_score(X_scaled, labels)
        metrics['calinski_harabasz'] = calinski_harabasz_score(X_scaled, labels)
    
    # Centroides (nos dados originais se foram normalizados)
    centroids_scaled = kmeans.cluster_centers_
    if scale and scaler is not None:
        centroids = scaler.inverse_transform(centroids_scaled)
    else:
        centroids = centroids_scaled
    
    # Criar DataFrame com centroides
    centroids_df = pd.DataFrame(
        centroids,
        columns=feature_columns
    )
    centroids_df.insert(0, 'Cluster', range(n_clusters))
    
    # Contar pontos por cluster
    unique, counts = np.unique(labels, return_counts=True)
    cluster_sizes = dict(zip(unique, counts))
    
    # Adicionar coluna de tamanho ao DataFrame de centroides
    centroids_df['N_Points'] = [cluster_sizes.get(i, 0) for i in range(n_clusters)]
    
    # Calcular variância intra-cluster
    cluster_variances = []
    for i in range(n_clusters):
        cluster_points = X_scaled[labels == i]
        if len(cluster_points) > 1:
            variance = np.mean(np.var(cluster_points, axis=0))
        else:
            variance = 0
        cluster_variances.append(variance)
    
    centroids_df['Variance'] = cluster_variances
    
    # Preparar interpretações
    interpretations = generate_interpretations(metrics, n_clusters, cluster_sizes)
    
    return {
        'labels': labels,
        'centroids': centroids_df,
        'metrics': metrics,
        'interpretations': interpretations,
        'scaler': scaler,
        'kmeans_model': kmeans,
        'X_scaled': X_scaled,
        'cluster_sizes': cluster_sizes
    }


def generate_interpretations(metrics, n_clusters, cluster_sizes):
    """
    Gera interpretações dos resultados
    
    Args:
        metrics: Dicionário com métricas
        n_clusters: Número de clusters
        cluster_sizes: Dicionário com tamanho de cada cluster
    
    Returns:
        dict com interpretações
    """
    interpretations = {}
    
    # Silhouette Score
    if metrics['silhouette'] is not None:
        sil = metrics['silhouette']
        if sil > 0.7:
            interpretations['silhouette'] = {
                'status': 'Excelente',
                'color': 'green',
                'message': f'Silhouette Score = {sil:.3f} - Clusters bem definidos e separados'
            }
        elif sil > 0.5:
            interpretations['silhouette'] = {
                'status': 'Bom',
                'color': 'blue',
                'message': f'Silhouette Score = {sil:.3f} - Boa estrutura de clusters'
            }
        elif sil > 0.25:
            interpretations['silhouette'] = {
                'status': 'Razoável',
                'color': 'yellow',
                'message': f'Silhouette Score = {sil:.3f} - Clusters com sobreposição moderada'
            }
        else:
            interpretations['silhouette'] = {
                'status': 'Fraco',
                'color': 'red',
                'message': f'Silhouette Score = {sil:.3f} - Clusters mal definidos, considere mudar K'
            }
    
    # Davies-Bouldin Index (quanto menor, melhor)
    if metrics['davies_bouldin'] is not None:
        db = metrics['davies_bouldin']
        if db < 0.5:
            interpretations['davies_bouldin'] = {
                'status': 'Excelente',
                'color': 'green',
                'message': f'Davies-Bouldin = {db:.3f} - Clusters muito bem separados'
            }
        elif db < 1.0:
            interpretations['davies_bouldin'] = {
                'status': 'Bom',
                'color': 'blue',
                'message': f'Davies-Bouldin = {db:.3f} - Boa separação entre clusters'
            }
        else:
            interpretations['davies_bouldin'] = {
                'status': 'Moderado',
                'color': 'yellow',
                'message': f'Davies-Bouldin = {db:.3f} - Considere ajustar o número de clusters'
            }
    
    # Distribuição de pontos
    sizes = list(cluster_sizes.values())
    max_size = max(sizes)
    min_size = min(sizes)
    ratio = max_size / min_size if min_size > 0 else float('inf')
    
    if ratio < 2:
        interpretations['balance'] = {
            'status': 'Balanceado',
            'color': 'green',
            'message': f'Clusters com tamanhos similares (razão {ratio:.1f}:1)'
        }
    elif ratio < 5:
        interpretations['balance'] = {
            'status': 'Moderado',
            'color': 'yellow',
            'message': f'Algum desbalanceamento entre clusters (razão {ratio:.1f}:1)'
        }
    else:
        interpretations['balance'] = {
            'status': 'Desbalanceado',
            'color': 'red',
            'message': f'Clusters muito desbalanceados (razão {ratio:.1f}:1) - considere revisar K'
        }
    
    return interpretations
