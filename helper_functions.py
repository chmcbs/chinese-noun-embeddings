"""
Helper functions for embedding generation, clustering, and analysis.

Note: This code was generated with AI assistance
"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


def generate_embeddings(nouns_df, model_name, embeddings_file_path, embedding_extraction_fn=None, device=None):
    """
    Generate embeddings for nouns using a specified model.
    """
    # Check if embeddings already exist
    if os.path.exists(embeddings_file_path):
        print(f"Loading existing embeddings from {embeddings_file_path}")
        embeddings = np.load(embeddings_file_path)
        
        # Validate shape matches current dataframe
        if embeddings.shape[0] == len(nouns_df):
            print(f"Loaded embeddings shape: {embeddings.shape}")
            return embeddings
        else:
            print(f"Warning: Loaded embeddings ({embeddings.shape[0]}) don't match dataframe ({len(nouns_df)})")
            print("Regenerating embeddings...")
            os.remove(embeddings_file_path)
    
    # Generate new embeddings
    print(f"Generating new embeddings using {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    # Default embedding extraction (BERT-style)
    if embedding_extraction_fn is None:
        def default_extract(outputs):
            return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        embedding_extraction_fn = default_extract
    
    nouns = nouns_df['Noun'].tolist()
    embeddings = []
    
    for i, noun in enumerate(nouns):
        inputs = tokenizer(noun, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        emb = embedding_extraction_fn(outputs)
        embeddings.append(emb)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(nouns)} nouns...")
    
    embeddings = np.stack(embeddings)
    
    # Save embeddings for future use
    np.save(embeddings_file_path, embeddings)
    print(f"Saved embeddings to {embeddings_file_path}")
    print(f"Embeddings shape: {embeddings.shape}")
    
    return embeddings


def run_clustering(embeddings, n_clusters, method='agglomerative', return_silhouette=False, random_state=42, n_init=10):
    """
    Run clustering on embeddings.
    """
    # Normalize method name (allow 'agg' as shorthand)
    if method in ['agg', 'agglomerative']:
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clusterer.fit_predict(embeddings)
    elif method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
        labels = clusterer.fit_predict(embeddings)
    else:
        raise ValueError(f"method must be 'agglomerative'/'agg' or 'kmeans', got '{method}'")
    
    if return_silhouette:
        silhouette = silhouette_score(embeddings, labels)
        return labels, silhouette
    else:
        return labels


def plot_silhouette_comparison(cluster_range, scores_dict, optimal_dict, titles_dict, figsize=(16, 6)):
    """
    Plot silhouette scores for clustering methods.
    
    Parameters:
        cluster_range: range of cluster counts
        scores_dict: score lists
        optimal_dict: optimal cluster counts
        titles_dict: plot titles
        figsize: figure size tuple
    """
    n_plots = len(scores_dict)
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    colors = {'kmeans': 'steelblue', 'agg': 'orange'}
    
    for idx, (method, scores) in enumerate(scores_dict.items()):
        ax = axes[idx]
        ax.plot(cluster_range, scores, marker='o', linewidth=2, markersize=6, 
                color=colors.get(method, 'steelblue'))
        ax.axvline(x=optimal_dict[method], color='r', linestyle='--', alpha=0.7,
                   label=f'Optimal: {optimal_dict[method]}')
        ax.set_xlabel('Number of Clusters', fontsize=12)
        ax.set_ylabel('Silhouette Score', fontsize=12)
        ax.set_title(titles_dict[method], fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.show()


def run_clustering_grid_search(embeddings, cluster_range, method='both', random_state=42, n_init=10, plot=False, title=None, figsize=(16, 6)):
    """
    Run grid search over cluster counts for specified clustering method(s).
    
    Returns:
        If method is 'agg' or 'kmeans': returns tuple (optimal_clusters, optimal_labels)
        If method is 'both': returns dict with optimal clusters and labels for both methods
    """
    cluster_range_list = list(cluster_range)
    scores_dict = {}
    optimal_dict = {}
    titles_dict = {}
    
    if method in ['agglomerative', 'agg', 'both']:
        agg_scores = []
        agg_labels_dict = {}  # Store labels for each cluster count
        
        for n_clusters in cluster_range:
            agg_labels, agg_silhouette = run_clustering(
                embeddings, n_clusters, method='agg', 
                return_silhouette=True
            )
            agg_scores.append(agg_silhouette)
            agg_labels_dict[n_clusters] = agg_labels  # Cache labels
        
        optimal_agg_idx = np.argmax(agg_scores)
        optimal_agg_clusters = cluster_range_list[optimal_agg_idx]
        optimal_agg_labels = agg_labels_dict[optimal_agg_clusters]  # Get cached labels
        
        if method in ['agglomerative', 'agg']:
            if plot:
                plot_silhouette_comparison(
                    cluster_range,
                    {'agg': agg_scores},
                    {'agg': optimal_agg_clusters},
                    {'agg': title or 'Agglomerative Clustering'},
                    figsize
                )
            return optimal_agg_clusters, optimal_agg_labels  # Return both
        
        scores_dict['agg'] = agg_scores
        optimal_dict['agg'] = optimal_agg_clusters
        titles_dict['agg'] = title or 'Agglomerative Clustering'
    
    if method in ['kmeans', 'both']:
        kmeans_scores = []
        kmeans_labels_dict = {}  # Store labels for each cluster count
        
        for n_clusters in cluster_range:
            kmeans_labels, kmeans_silhouette = run_clustering(
                embeddings, n_clusters, method='kmeans', 
                random_state=random_state, n_init=n_init, 
                return_silhouette=True
            )
            kmeans_scores.append(kmeans_silhouette)
            kmeans_labels_dict[n_clusters] = kmeans_labels  # Cache labels
        
        optimal_kmeans_idx = np.argmax(kmeans_scores)
        optimal_kmeans_clusters = cluster_range_list[optimal_kmeans_idx]
        optimal_kmeans_labels = kmeans_labels_dict[optimal_kmeans_clusters]  # Get cached labels
        
        if method == 'kmeans':
            if plot:
                plot_silhouette_comparison(
                    cluster_range,
                    {'kmeans': kmeans_scores},
                    {'kmeans': optimal_kmeans_clusters},
                    {'kmeans': title or 'K-means Clustering'},
                    figsize
                )
            return optimal_kmeans_clusters, optimal_kmeans_labels  # Return both
        
        scores_dict['kmeans'] = kmeans_scores
        optimal_dict['kmeans'] = optimal_kmeans_clusters
        titles_dict['kmeans'] = title or 'K-means Clustering'
    
    # If method is 'both'
    if plot:
        plot_silhouette_comparison(cluster_range, scores_dict, optimal_dict, titles_dict, figsize)
    
    return {
        'optimal_agg_clusters': optimal_agg_clusters,
        'optimal_agg_labels': optimal_agg_labels,
        'optimal_kmeans_clusters': optimal_kmeans_clusters,
        'optimal_kmeans_labels': optimal_kmeans_labels
    }


def run_pca_grid_search(embeddings, pca_components, n_clusters=None, random_state=42, compute_silhouette=True):
    """
    Run PCA grid search across different component counts.
    """
    pca_models_dict = {}
    pca_embeddings_dict = {}
    pca_scores = []
    
    for n_components in pca_components:
        # Apply PCA
        pca_model = PCA(n_components=n_components, random_state=random_state)
        pca_embeddings = pca_model.fit_transform(embeddings)
        
        pca_models_dict[n_components] = pca_model
        pca_embeddings_dict[n_components] = pca_embeddings
        
        # Compute silhouette score if requested
        if compute_silhouette and n_clusters is not None:
            labels = run_clustering(pca_embeddings, n_clusters)
            silhouette = silhouette_score(pca_embeddings, labels)
            pca_scores.append(silhouette)
    
    return {
        'pca_models_dict': pca_models_dict,
        'pca_embeddings_dict': pca_embeddings_dict,
        'pca_scores': pca_scores if compute_silhouette else None,
        'pca_components': pca_components
    }


def compute_enhanced_metrics(embeddings_pca, embeddings_original, labels, n_clusters, baseline_centroids, pca_transformer=None):
    """
    Compute enhanced metrics for PCA clustering analysis.
    """
    metrics_list = []
    
    # Compute post-PCA centroids in original space
    pca_centroids_original = []
    for i in range(n_clusters):
        cluster_mask = labels == i
        centroid_original = embeddings_original[cluster_mask].mean(axis=0)
        pca_centroids_original.append(centroid_original)
    pca_centroids_original = np.array(pca_centroids_original)
    
    # Match post-PCA clusters to baseline clusters using Hungarian algorithm
    centroid_similarity_matrix = cosine_similarity(pca_centroids_original, baseline_centroids)
    row_ind, col_ind = linear_sum_assignment(-centroid_similarity_matrix)  # negative because it minimizes
    best_matches = {row_ind[i]: col_ind[i] for i in range(len(row_ind))}
    
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_members = embeddings_original[cluster_mask]
        n_members = cluster_members.shape[0]
        
        if n_members < 2:
            continue
        
        centroid_original = pca_centroids_original[cluster_id]
        
        # Intra-cluster coherence
        pairwise_sim = cosine_similarity(cluster_members)
        mask = ~np.eye(pairwise_sim.shape[0], dtype=bool)
        intra_coherence = pairwise_sim[mask].mean() if mask.sum() > 0 else 1.0
        
        # Inter-centroid separation
        other_centroids = np.delete(pca_centroids_original, cluster_id, axis=0)
        inter_separation = cosine_similarity(
            centroid_original.reshape(1, -1), 
            other_centroids
        ).mean()
        
        # Baseline similarity
        best_baseline_idx = best_matches[cluster_id]
        baseline_similarity = cosine_similarity(
            centroid_original.reshape(1, -1),
            baseline_centroids[best_baseline_idx].reshape(1, -1)
        )[0, 0]
        
        # Reconstruction fidelity
        if pca_transformer is not None:
            centroid_pca = embeddings_pca[cluster_mask].mean(axis=0)
            centroid_reconstructed = pca_transformer.inverse_transform(
                centroid_pca.reshape(1, -1)
            )[0]
            reconstruction_fidelity = cosine_similarity(
                centroid_original.reshape(1, -1),
                centroid_reconstructed.reshape(1, -1)
            )[0, 0]
        else:
            reconstruction_fidelity = 1.0
        
        metrics_list.append({
            'cluster_id': cluster_id,
            'n_members': n_members,
            'intra_coherence': intra_coherence,
            'inter_separation': inter_separation,
            'baseline_similarity': baseline_similarity,
            'reconstruction_fidelity': reconstruction_fidelity,
            'matched_baseline_cluster': best_baseline_idx
        })
    
    df = pd.DataFrame(metrics_list)
    
    # Compute composite score (normalized intra_coherence - normalized inter_separation)
    if len(df) > 0:
        intra_min, intra_max = df['intra_coherence'].min(), df['intra_coherence'].max()
        inter_min, inter_max = df['inter_separation'].min(), df['inter_separation'].max()
        
        # Avoid division by zero if all values are the same
        intra_range = intra_max - intra_min
        inter_range = inter_max - inter_min
        
        if intra_range > 0:
            intra_normalized = (df['intra_coherence'] - intra_min) / intra_range
        else:
            intra_normalized = 0
        
        if inter_range > 0:
            inter_normalized = (df['inter_separation'] - inter_min) / inter_range
        else:
            inter_normalized = 0
        
        df['composite_score'] = intra_normalized - inter_normalized
    
    return df


def plot_pca_enhanced_analysis(summary_metrics, figsize=(18, 10)):
    """
    Plot enhanced PCA analysis metrics in a 2x3 grid.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle('PCA Dimensionality Selection: Enhanced Analysis', fontsize=16, fontweight='bold', y=1.00)
    
    plot_data = summary_metrics.copy()
    plot_data['pca_numeric'] = plot_data['pca_components'].apply(lambda x: -10 if x == 'Baseline' else x)
    plot_data = plot_data.sort_values('pca_numeric')
    x_vals = range(len(plot_data))

    # Plot 1: Silhouette Score
    ax1 = axes[0, 0]
    ax1.plot(x_vals, plot_data['silhouette_score'], marker='o', linewidth=2, markersize=8, color='steelblue')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_xlabel('PCA Components', fontweight='bold')
    ax1.set_ylabel('Silhouette Score', fontweight='bold')
    ax1.set_title('Silhouette Score', fontweight='bold')
    ax1.set_xticks(x_vals)
    ax1.set_xticklabels(plot_data['pca_components'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Reconstruction Fidelity
    ax2 = axes[0, 1]
    ax2.plot(x_vals[1:], plot_data['mean_reconstruction_fidelity'].iloc[1:], marker='o', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('PCA Components', fontweight='bold')
    ax2.set_ylabel('Reconstruction Fidelity', fontweight='bold')
    ax2.set_title('Centroid Reconstruction Fidelity (↑ better)', fontweight='bold')
    ax2.set_xticks(x_vals[1:])
    ax2.set_xticklabels(plot_data['pca_components'].iloc[1:], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Baseline Similarity
    ax3 = axes[0, 2]
    ax3.plot(x_vals[1:], plot_data['mean_baseline_similarity'].iloc[1:], marker='o', linewidth=2, markersize=8, color='blue')
    ax3.set_xlabel('PCA Components', fontweight='bold')
    ax3.set_ylabel('Similarity to Baseline', fontweight='bold')
    ax3.set_title('Centroid Similarity to Baseline', fontweight='bold')
    ax3.set_xticks(x_vals[1:])
    ax3.set_xticklabels(plot_data['pca_components'].iloc[1:], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Intra-cluster Coherence
    ax4 = axes[1, 0]
    ax4.plot(x_vals, plot_data['mean_intra_coherence'], marker='o', linewidth=2, markersize=8, color='green')
    ax4.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax4.set_xlabel('PCA Components', fontweight='bold')
    ax4.set_ylabel('Intra-Cluster Coherence', fontweight='bold')
    ax4.set_title('Intra-Cluster Coherence (↑ better)', fontweight='bold')
    ax4.set_xticks(x_vals)
    ax4.set_xticklabels(plot_data['pca_components'], rotation=45, ha='right')
    ax4.grid(True, alpha=0.3)

    # Plot 5: Inter-Centroid Separation
    ax5 = axes[1, 1]
    ax5.plot(x_vals, plot_data['mean_inter_separation'], marker='o', linewidth=2, markersize=8, color='purple')
    ax5.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    ax5.set_xlabel('PCA Components', fontweight='bold')
    ax5.set_ylabel('Inter-Centroid Separation', fontweight='bold')
    ax5.set_title('Inter-Centroid Separation (↓ better)', fontweight='bold')
    ax5.set_xticks(x_vals)
    ax5.set_xticklabels(plot_data['pca_components'], rotation=45, ha='right')
    ax5.grid(True, alpha=0.3)

    # Plot 6: Composite Score
    ax6 = axes[1, 2]
    ax6.plot(x_vals, plot_data['mean_composite_score'], marker='o', linewidth=2, markersize=8, color='darkred')
    ax6.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    best_idx = plot_data['mean_composite_score'].idxmax()
    best_x = plot_data.index.get_loc(best_idx)
    ax6.axvline(x=best_x, color='green', linestyle='--', alpha=0.7, linewidth=2, label=f'Best: {plot_data.iloc[best_idx]["pca_components"]}')
    ax6.set_xlabel('PCA Components', fontweight='bold')
    ax6.set_ylabel('Composite Score', fontweight='bold')
    ax6.set_title('Composite Score (coherence - separation)', fontweight='bold')
    ax6.set_xticks(x_vals)
    ax6.set_xticklabels(plot_data['pca_components'], rotation=45, ha='right')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.tight_layout()
    plt.show()


def get_representative_nouns(embeddings, labels, nouns_df, n_clusters, n_representatives=25, return_dict=False):
    """
    Find the n closest nouns to each cluster centroid using KNN.
    """
    results_list = []
    
    for cluster_id in range(n_clusters):
        # Get all points in this cluster
        cluster_mask = labels == cluster_id
        cluster_embeddings = embeddings[cluster_mask]
        cluster_nouns = nouns_df[cluster_mask]
        
        if len(cluster_embeddings) == 0:
            results_list.append({
                'cluster_id': cluster_id,
                'size': 0,
                'representatives': []
            })
            continue
        
        # Calculate centroid
        centroid = cluster_embeddings.mean(axis=0).reshape(1, -1)
        
        # Use KNN to find closest nouns to centroid
        knn = NearestNeighbors(n_neighbors=min(n_representatives, len(cluster_embeddings)))
        knn.fit(cluster_embeddings)
        distances, indices = knn.kneighbors(centroid)
        
        # Get the representative nouns
        representative_nouns = cluster_nouns.iloc[indices[0]]
        
        results_list.append({
            'cluster_id': cluster_id,
            'size': cluster_mask.sum(),
            'representatives': representative_nouns[['Noun', 'English', 'Frequency']].to_dict('records')
        })
    
    if return_dict:
        # Convert to dict format for compatibility
        return {item['cluster_id']: item['representatives'] for item in results_list}
    else:
        return results_list


def print_representative_nouns(results_list, title=None, max_per_cluster=None, sort_by='cluster_id'):
    """
    Print representative nouns for each cluster.
    """
    if title:
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
    
    # Sort results if requested
    if sort_by == 'size':
        results_list = sorted(results_list, key=lambda x: x['size'], reverse=True)
    elif sort_by == 'cluster_id':
        results_list = sorted(results_list, key=lambda x: x['cluster_id'])
    # If sort_by is None or other value, don't sort
    
    for cluster_info in results_list:
        cluster_id = cluster_info['cluster_id']
        size = cluster_info['size']
        representatives = cluster_info['representatives']
        
        if max_per_cluster:
            representatives = representatives[:max_per_cluster]
        
        print(f"\nCluster {cluster_id} (size: {size}):")
        for i, rep in enumerate(representatives, 1):
            noun = rep.get('Noun', '')
            english = rep.get('English', '')
            freq = rep.get('Frequency', 0)
            print(f"  {i}. {noun:6s} - {english:20s} (freq: {freq:.0f})")
    
    if title:
        print("\n" + "=" * 70)


def display_clusters_side_by_side(group1_data, group2_data, group1_labels, group2_labels, n_clusters, group1_title, group2_title, n_shared_words=None, sort_by='cluster'):
    """
    Display two clusterings side-by-side with matched clusters.
    Shows all clusters from group1, with matched clusters first and unmatched at the bottom.
    """
    # Determine which words to use for matching
    if n_shared_words is None:
        n_shared_words = len(group1_labels)
    
    # Get actual cluster IDs that exist in the data dictionaries
    group1_cluster_ids = set(group1_data.keys())
    group2_cluster_ids = set(group2_data.keys())
    
    # Determine the maximum cluster ID needed for the overlap matrix
    max_cluster_id = max(max(group1_cluster_ids, default=0), max(group2_cluster_ids, default=0), n_clusters - 1)
    actual_n_clusters = max_cluster_id + 1
    
    # Match clusters based on shared words only
    overlap_matrix = np.zeros((actual_n_clusters, actual_n_clusters))
    for i in range(actual_n_clusters):
        for j in range(actual_n_clusters):
            # Count overlap in the shared words
            overlap = np.sum((group1_labels[:n_shared_words] == i) & 
                           (group2_labels[:n_shared_words] == j))
            overlap_matrix[i, j] = overlap
    
    # Use Hungarian algorithm to find best matching
    row_ind, col_ind = linear_sum_assignment(-overlap_matrix)
    group1_to_group2 = {int(row_ind[i]): int(col_ind[i]) for i in range(len(row_ind))}
    
    # Get cluster sizes from FULL label arrays - only for clusters that exist in group1_data
    group1_sizes = [(i, sum(group1_labels == i)) for i in group1_cluster_ids]
    
    # Separate matched and unmatched clusters
    matched_clusters = []
    unmatched_clusters = []
    
    for cluster_id, cluster_size in group1_sizes:
        matched_cluster_id = group1_to_group2.get(cluster_id)
        if matched_cluster_id is not None and matched_cluster_id in group2_data:
            matched_clusters.append((cluster_id, cluster_size, matched_cluster_id))
        else:
            unmatched_clusters.append((cluster_id, cluster_size))
    
    # Sort matched clusters based on sort_by parameter
    if sort_by in ['size', 'Size', 'SIZE']:
        matched_clusters.sort(key=lambda x: x[1], reverse=True)  # Sort by size, largest first
        unmatched_clusters.sort(key=lambda x: x[1], reverse=True)  # Sort unmatched by size too
    else:
        # Default to cluster number
        matched_clusters.sort(key=lambda x: x[0])  # Sort by cluster number
        unmatched_clusters.sort(key=lambda x: x[0])  # Sort unmatched by cluster number
    
    print("=" * 160)
    print(f"{group1_title:^80s}{group2_title:^80s}")
    print("=" * 160)
    print()
    
    # Display matched clusters first
    for group1_cluster_id, group1_size, group2_cluster_id in matched_clusters:
        group2_size = sum(group2_labels == group2_cluster_id)
        
        # Group 1 cluster
        words = group1_data[group1_cluster_id]
        group1_lines = [f"Cluster {group1_cluster_id:2d} (size: {group1_size:3d})"]
        group1_lines.append("─" * 78)
        for i, word in enumerate(words, 1):
            group1_lines.append(f"  {i:2d}. {word['Noun']:6s} - {word['English']:20s} (freq: {word['Frequency']:>6.0f})")
        
        # Group 2 cluster (matched)
        words = group2_data[group2_cluster_id]
        group2_lines = [f"Cluster {group2_cluster_id:2d} (size: {group2_size:3d})"]
        group2_lines.append("─" * 78)
        for i, word in enumerate(words, 1):
            group2_lines.append(f"  {i:2d}. {word['Noun']:6s} - {word['English']:20s} (freq: {word['Frequency']:>6.0f})")
        
        # Print side by side
        max_lines = max(len(group1_lines), len(group2_lines))
        for line_idx in range(max_lines):
            group1_line = group1_lines[line_idx] if line_idx < len(group1_lines) else ""
            group2_line = group2_lines[line_idx] if line_idx < len(group2_lines) else ""
            print(f"{group1_line:78s}  {group2_line:78s}")
        
        print()
    
    # Display unmatched clusters at the bottom
    for group1_cluster_id, group1_size in unmatched_clusters:
        # Group 1 cluster
        words = group1_data[group1_cluster_id]
        group1_lines = [f"Cluster {group1_cluster_id:2d} (size: {group1_size:3d})"]
        group1_lines.append("─" * 78)
        for i, word in enumerate(words, 1):
            group1_lines.append(f"  {i:2d}. {word['Noun']:6s} - {word['English']:20s} (freq: {word['Frequency']:>6.0f})")
        
        # Group 2 placeholder (no match)
        group2_lines = [f"No match"]
        group2_lines.append("─" * 78)
        group2_lines.append("  (No corresponding cluster)")
        
        # Print side by side
        max_lines = max(len(group1_lines), len(group2_lines))
        for line_idx in range(max_lines):
            group1_line = group1_lines[line_idx] if line_idx < len(group1_lines) else ""
            group2_line = group2_lines[line_idx] if line_idx < len(group2_lines) else ""
            print(f"{group1_line:78s}  {group2_line:78s}")
        
        print()


def create_3d_cluster_visualisation(clustering_data, cluster_labels, words_to_exclude=None, color_map=None, top_n_words_per_cluster=10, word_size_min=3, word_size_max=6, sphere_size_base=40, label_offset_z=0.4, word_distance_scale=0.3, renderer='browser', title=None, subtitle=None):
    """
    Create a 3D visualisation of clustered noun embeddings.
    """
    import numpy as np
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA
    from pypinyin import lazy_pinyin, Style
    import pickle
    
    # Load clustering data - handle both file path and direct data
    if isinstance(clustering_data, str):
        # It's a file path - load from pickle
        with open(clustering_data, 'rb') as f:
            clustering_data = pickle.load(f)
    elif not isinstance(clustering_data, dict):
        raise TypeError("clustering_data must be either a string (file path) or a dictionary")
    
    # Extract the saved data
    labels = clustering_data['labels']
    embeddings = clustering_data['embeddings']
    nouns_df = clustering_data['nouns_df'].copy()
    
    # Default words to exclude (empty list)
    if words_to_exclude is None:
        words_to_exclude = []
    
    # Filter the data
    if len(words_to_exclude) > 0:
        mask_to_keep = ~nouns_df['Noun'].isin(words_to_exclude)
        indices_to_keep = mask_to_keep.values
        
        nouns_df = nouns_df[mask_to_keep].reset_index(drop=True)
        embeddings = embeddings[indices_to_keep]
        labels = labels[indices_to_keep]
    
    # Add pinyin column
    nouns_df['Pinyin'] = nouns_df['Noun'].apply(lambda x: ' '.join(lazy_pinyin(x, style=Style.TONE)))
    
    # Reduce to 3D for visualisation
    pca_3d = PCA(n_components=3, random_state=42)
    embeddings_3d = pca_3d.fit_transform(embeddings)
    
    # Only display clusters that have labels
    display_clusters = np.array(list(cluster_labels.keys()))
    
    # Generate color map if not provided, or fill in missing colors
    if color_map is None:
        color_map = {}

    # Fill in any missing colors for clusters in cluster_labels
    if len(color_map) < len(display_clusters):
        # Use a set of highly distinct colors
        base_colors = [
            '#EF4444',  # Red
            '#10B981',  # Green
            '#3B82F6',  # Blue
            '#F59E0B',  # Orange
            '#8B5CF6',  # Purple
            '#06B6D4',  # Cyan
            '#EC4899',  # Pink
            '#EAB308',  # Yellow
            '#84CC16',  # Lime
            '#F97316',  # Orange-Red
            '#14B8A6',  # Teal
            '#6366F1',  # Indigo
            '#A855F7',  # Violet
            '#F43F5E',  # Rose
            '#22C55E',  # Emerald
            '#0EA5E9',  # Sky Blue
            '#A3E635',  # Light Green
            '#FB923C',  # Light Orange
        ]
        
        # Find clusters that need colors
        clusters_needing_colors = [cid for cid in display_clusters if cid not in color_map]
        
        # Assign colors to missing clusters, avoiding colors already used
        used_colors = set(color_map.values())
        available_colors = [c for c in base_colors if c not in used_colors]
        
        for i, cluster_id in enumerate(clusters_needing_colors):
            color_map[cluster_id] = available_colors[i % len(available_colors)]
    
    # Create lists to store cluster info and top words
    cluster_info = []
    all_top_words_data = []
    
    for cluster_id in display_clusters:
        cluster_mask = labels == cluster_id
        cluster_points_3d = embeddings_3d[cluster_mask]
        cluster_points_100d = embeddings[cluster_mask]
        cluster_words = nouns_df[cluster_mask].copy()
        
        if len(cluster_points_3d) == 0:
            continue
        
        # Calculate the centroid in each embedding space
        centroid_3d = cluster_points_3d.mean(axis=0)
        centroid_100d = cluster_points_100d.mean(axis=0)
        
        # Calculate each word's distance from the centroid in 100-dimensional space
        distances_100d = np.linalg.norm(cluster_points_100d - centroid_100d, axis=1)
        
        # Get n closest words
        closest_indices = np.argsort(distances_100d)[:top_n_words_per_cluster]
        
        # Get average distance for 10 closest words
        avg_distance_top10 = distances_100d[closest_indices].mean()
        
        cluster_words_with_index = cluster_words.reset_index(drop=True)
        
        for local_idx in closest_indices:
            word_row = cluster_words_with_index.iloc[local_idx]
            
            all_top_words_data.append({
                'cluster_id': cluster_id,
                'noun': word_row['Noun'],
                'english': word_row['English'],
                'pinyin': word_row['Pinyin'],
                'frequency': word_row['Frequency'],
                'position_3d': cluster_points_3d[local_idx],
                'centroid_3d': centroid_3d,
                'color': color_map[cluster_id]
            })
        
        cluster_info.append({
            'id': cluster_id,
            'centroid': centroid_3d,
            'size': cluster_mask.sum(),
            'avg_spread': avg_distance_top10,
            'label': cluster_labels[cluster_id],
            'color': color_map[cluster_id]
        })
    
    # Group words by cluster for per-cluster frequency scaling
    words_by_cluster = {}
    for word_data in all_top_words_data:
        cluster_id = word_data['cluster_id']
        if cluster_id not in words_by_cluster:
            words_by_cluster[cluster_id] = []
        words_by_cluster[cluster_id].append(word_data)
    
    # Calculate frequency range for each cluster
    cluster_freq_ranges = {}
    for cluster_id, words in words_by_cluster.items():
        freqs = [w['frequency'] for w in words]
        min_freq = min(freqs)
        max_freq = max(freqs)
        cluster_freq_ranges[cluster_id] = {
            'min': min_freq,
            'max': max_freq,
            'range': max_freq - min_freq if max_freq > min_freq else 1
        }
    
    # Calculate spread range across all clusters for sphere sizing
    all_spreads = [c['avg_spread'] for c in cluster_info]
    min_spread = min(all_spreads)
    max_spread = max(all_spreads)
    spread_range = max_spread - min_spread if max_spread > min_spread else 1
    
    # Create figure
    fig = go.Figure()
    
    # Add spheres for each cluster (sized by spread)
    for info in cluster_info:
        centroid = info['centroid']
        
        # Scale sphere size based on cluster spread
        normalised_spread = (info['avg_spread'] - min_spread) / spread_range
        sphere_size = sphere_size_base * (0.5 + normalised_spread * 1.0)
        
        # Add sphere
        fig.add_trace(go.Scatter3d(
            x=[centroid[0]],
            y=[centroid[1]],
            z=[centroid[2]],
            mode='markers',
            name=info['label'],
            marker=dict(
                size=sphere_size,
                color=info['color'],
                line=dict(width=2, color='white'),
                opacity=0.3
            ),
            hovertemplate=(
                f"<b>{info['label']}</b><br>"
                f"Size: {info['size']} words<br>"
                f"Spread: {info['avg_spread']:.3f}<br>"
                "<extra></extra>"
            ),
            showlegend=True,
            legendgroup=str(info['id'])
        ))
        
        # Add cluster label above sphere
        fig.add_trace(go.Scatter3d(
            x=[centroid[0]],
            y=[centroid[1]],
            z=[centroid[2] + label_offset_z],
            mode='text',
            text=[info['label']],
            textfont=dict(size=14, color='white', family='Arial Black'),
            textposition='middle center',
            showlegend=False,
            hoverinfo='skip',
            legendgroup=str(info['id'])
        ))
    
    # Add top words as sized points (with per-cluster frequency scaling)
    for word_data in all_top_words_data:
        cluster_id = word_data['cluster_id']
        freq_info = cluster_freq_ranges[cluster_id]
        
        # Scale word size based on frequency within the cluster
        normalised_freq = (word_data['frequency'] - freq_info['min']) / freq_info['range']
        word_size = word_size_min + (word_size_max - word_size_min) * normalised_freq
        
        pos = word_data['position_3d']
        centroid = word_data['centroid_3d']
        
        # Scale distance from centroid
        direction = pos - centroid
        scaled_pos = centroid + direction * word_distance_scale
        
        fig.add_trace(go.Scatter3d(
            x=[scaled_pos[0]],
            y=[scaled_pos[1]],
            z=[scaled_pos[2]],
            mode='markers',
            marker=dict(
                size=word_size,
                color=word_data['color'],
                line=dict(width=1, color='white'),
                opacity=0.9
            ),
            hovertemplate=(
                f"<b>{word_data['noun']}</b> <b>({word_data['pinyin']})</b><br>"
                f"{word_data['english']}<br>"
                "<extra></extra>"
            ),
            showlegend=False,
            legendgroup=str(word_data['cluster_id'])
        ))
    
    # Configure layout (hardcoded styling)
    title_config = None
    if title is not None:
        if subtitle:
            title_text = f'<b>{title}</b><br>{subtitle}'
        else:
            title_text = f'<b>{title}</b>'
        title_config = {
            'text': title_text,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': 'white'}
        }
    
    fig.update_layout(
        title=title_config,
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3',
            xaxis=dict(
                backgroundcolor='#1E1E1E',
                gridcolor='#404040',
                showbackground=True,
                zerolinecolor='#404040',
                title=dict(font=dict(color='white'))
            ),
            yaxis=dict(
                backgroundcolor='#1E1E1E',
                gridcolor='#404040',
                showbackground=True,
                zerolinecolor='#404040',
                title=dict(font=dict(color='white'))
            ),
            zaxis=dict(
                backgroundcolor='#1E1E1E',
                gridcolor='#404040',
                showbackground=True,
                zerolinecolor='#404040',
                title=dict(font=dict(color='white'))
            ),
            camera=dict(
                eye=dict(x=2.5, y=2.5, z=2.5)
            ),
            bgcolor='#1E1E1E'
        ),
        width=1600,
        height=1000,
        hoverlabel=dict(
            bgcolor="rgba(30,30,30,0.95)",
            font_size=12,
            font_family="Arial",
            font_color="white"
        ),
        paper_bgcolor='#1E1E1E',
        plot_bgcolor='#1E1E1E',
        hovermode='closest',
        showlegend=True,
        legend=dict(
            bgcolor='rgba(30,30,30,0.8)',
            font=dict(color='white', size=10),
            itemsizing='constant'
        )
    )
    
    if renderer:
        fig.show(renderer=renderer)
    
    return fig