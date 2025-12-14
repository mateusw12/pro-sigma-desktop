"""
Text Analysis Utilities
Functions for text mining, NLP, and visualization
"""
from typing import List, Dict, Tuple, Optional, Set
import re
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# Stopwords for different languages
STOPWORDS = {
    'portuguese': {
        'a', 'o', 'as', 'os', 'de', 'da', 'do', 'das', 'dos', 'em', 'no', 'na', 'nos', 'nas',
        'um', 'uma', 'uns', 'umas', 'para', 'com', 'sem', 'sob', 'sobre', 'por', 'ao', 'aos',
        'e', 'ou', 'mas', 'se', 'que', 'quando', 'onde', 'como', 'porque', 'porquê',
        'ele', 'ela', 'eles', 'elas', 'eu', 'tu', 'nós', 'vós', 'você', 'vocês',
        'meu', 'minha', 'meus', 'minhas', 'seu', 'sua', 'seus', 'suas', 'nosso', 'nossa',
        'este', 'esta', 'estes', 'estas', 'esse', 'essa', 'esses', 'essas', 'aquele', 'aquela',
        'sim', 'não', 'nem', 'já', 'mais', 'menos', 'muito', 'pouco', 'tanto', 'tão',
        'bem', 'mal', 'só', 'também', 'ainda', 'nunca', 'sempre', 'aqui', 'ali', 'lá',
        'ser', 'estar', 'ter', 'haver', 'fazer', 'ir', 'vir', 'dar', 'poder', 'dever',
        'é', 'são', 'está', 'estão', 'foi', 'foram', 'será', 'serão', 'era', 'eram',
        'tem', 'têm', 'tinha', 'tinham', 'há', 'faz', 'fez', 'pode', 'podem', 'deve', 'devem'
    },
    'english': {
        'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why', 'how',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'mine', 'yours', 'ours', 'theirs',
        'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'may', 'might',
        'can', 'must', 'shall', 'to', 'of', 'in', 'on', 'at', 'by', 'for', 'with', 'about',
        'as', 'from', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down',
        'yes', 'no', 'not', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some',
        'such', 'than', 'too', 'very', 'just', 'only', 'own', 'same', 'so', 'also'
    },
    'spanish': {
        'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'de', 'del', 'al', 'en', 'por',
        'para', 'con', 'sin', 'sobre', 'entre', 'y', 'o', 'pero', 'si', 'no', 'sí',
        'yo', 'tú', 'él', 'ella', 'nosotros', 'vosotros', 'ellos', 'ellas', 'me', 'te', 'se',
        'mi', 'tu', 'su', 'nuestro', 'vuestro', 'este', 'ese', 'aquel', 'esta', 'esa', 'aquella',
        'ser', 'estar', 'haber', 'tener', 'hacer', 'ir', 'dar', 'poder', 'deber', 'querer',
        'es', 'son', 'está', 'están', 'fue', 'fueron', 'será', 'ha', 'han', 'había',
        'más', 'menos', 'muy', 'poco', 'mucho', 'tanto', 'también', 'ya', 'aún', 'siempre'
    }
}


def preprocess_text(
    text: str,
    language: str = 'portuguese',
    custom_stopwords: Optional[Set[str]] = None,
    custom_ignore_chars: Optional[str] = None,
    min_word_length: int = 2,
    lowercase: bool = True
) -> str:
    """
    Preprocess text by removing stopwords, special characters, etc.
    
    Args:
        text: Input text
        language: Language for stopword removal
        custom_stopwords: Additional stopwords to remove
        custom_ignore_chars: Additional characters to ignore
        min_word_length: Minimum word length to keep
        lowercase: Convert to lowercase
        
    Returns:
        Preprocessed text
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase if requested
    if lowercase:
        text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Get stopwords for the language
    stopwords = STOPWORDS.get(language.lower(), set())
    if custom_stopwords:
        stopwords = stopwords.union(custom_stopwords)
    
    # Remove punctuation (keep only letters, numbers, and spaces)
    if custom_ignore_chars:
        for char in custom_ignore_chars:
            text = text.replace(char, ' ')
    
    # Remove stopwords and short words
    words = text.split()
    filtered_words = [
        word for word in words
        if word not in stopwords and len(word) >= min_word_length and word.isalpha()
    ]
    
    return ' '.join(filtered_words)


def extract_words(texts: List[str]) -> List[str]:
    """Extract all words from list of texts"""
    all_words = []
    for text in texts:
        if isinstance(text, str):
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
    return all_words


def extract_phrases(texts: List[str], n_gram: int = 2) -> List[str]:
    """
    Extract n-grams (phrases) from texts
    
    Args:
        texts: List of texts
        n_gram: Number of words per phrase (2 for bigrams, 3 for trigrams, etc.)
        
    Returns:
        List of phrases
    """
    all_phrases = []
    for text in texts:
        if isinstance(text, str):
            words = re.findall(r'\b\w+\b', text.lower())
            for i in range(len(words) - n_gram + 1):
                phrase = ' '.join(words[i:i + n_gram])
                all_phrases.append(phrase)
    return all_phrases


def count_words(words: List[str], top_n: int = 50) -> pd.DataFrame:
    """
    Count word frequencies
    
    Args:
        words: List of words
        top_n: Number of top words to return
        
    Returns:
        DataFrame with word counts
    """
    word_counts = Counter(words)
    most_common = word_counts.most_common(top_n)
    
    df = pd.DataFrame(most_common, columns=['Palavra', 'Frequência'])
    
    # Add percentage
    total = sum(word_counts.values())
    df['Percentual'] = (df['Frequência'] / total * 100).round(2)
    
    # Add rank
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    return df


def count_phrases(phrases: List[str], top_n: int = 30) -> pd.DataFrame:
    """
    Count phrase frequencies
    
    Args:
        phrases: List of phrases
        top_n: Number of top phrases to return
        
    Returns:
        DataFrame with phrase counts
    """
    phrase_counts = Counter(phrases)
    most_common = phrase_counts.most_common(top_n)
    
    df = pd.DataFrame(most_common, columns=['Frase', 'Frequência'])
    
    # Add percentage
    total = sum(phrase_counts.values())
    df['Percentual'] = (df['Frequência'] / total * 100).round(2)
    
    # Add rank
    df.insert(0, 'Rank', range(1, len(df) + 1))
    
    return df


def search_keyword(
    texts: List[str],
    keyword: str,
    case_sensitive: bool = False
) -> Dict[str, any]:
    """
    Search for a keyword or phrase in texts
    
    Args:
        texts: List of texts
        keyword: Keyword or phrase to search
        case_sensitive: Whether search is case-sensitive
        
    Returns:
        Dictionary with search results
    """
    results = {
        'total_occurrences': 0,
        'documents_with_keyword': 0,
        'occurrences_per_document': [],
        'contexts': []
    }
    
    for idx, text in enumerate(texts):
        if not isinstance(text, str):
            continue
        
        search_text = text if case_sensitive else text.lower()
        search_keyword = keyword if case_sensitive else keyword.lower()
        
        # Count occurrences
        count = search_text.count(search_keyword)
        
        if count > 0:
            results['total_occurrences'] += count
            results['documents_with_keyword'] += 1
            results['occurrences_per_document'].append({
                'document_index': idx,
                'count': count
            })
            
            # Extract context (surrounding text)
            pattern = re.compile(re.escape(search_keyword), re.IGNORECASE if not case_sensitive else 0)
            for match in pattern.finditer(text):
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                results['contexts'].append({
                    'document_index': idx,
                    'context': context,
                    'position': match.start()
                })
    
    return results


def create_wordcloud(
    words: List[str],
    max_words: int = 100,
    figsize: Tuple[int, int] = (12, 8),
    background_color: str = 'white',
    colormap: str = 'viridis'
) -> Figure:
    """
    Create word cloud visualization
    
    Args:
        words: List of words
        max_words: Maximum number of words in cloud
        figsize: Figure size
        background_color: Background color
        colormap: Color scheme
        
    Returns:
        Matplotlib Figure
    """
    try:
        from wordcloud import WordCloud
    except ImportError:
        # Fallback: create a bar chart instead
        return create_word_frequency_chart(words, top_n=max_words, figsize=figsize)
    
    # Create frequency dictionary
    word_freq = Counter(words)
    
    if not word_freq:
        # Return empty figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Nenhuma palavra encontrada', 
                ha='center', va='center', fontsize=16)
        ax.axis('off')
        return fig
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=1200,
        height=800,
        max_words=max_words,
        background_color=background_color,
        colormap=colormap,
        relative_scaling=0.5,
        min_font_size=10
    ).generate_from_frequencies(word_freq)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Nuvem de Palavras', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig


def create_word_frequency_chart(
    words: List[str],
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8),
    orientation: str = 'horizontal'
) -> Figure:
    """
    Create word frequency bar chart
    
    Args:
        words: List of words
        top_n: Number of top words to show
        figsize: Figure size
        orientation: 'horizontal' or 'vertical'
        
    Returns:
        Matplotlib Figure
    """
    word_counts = Counter(words)
    most_common = word_counts.most_common(top_n)
    
    if not most_common:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Nenhuma palavra encontrada', 
                ha='center', va='center', fontsize=16)
        ax.axis('off')
        return fig
    
    words_list, counts = zip(*most_common)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if orientation == 'horizontal':
        bars = ax.barh(range(len(words_list)), counts, color='steelblue', edgecolor='black', linewidth=0.5)
        ax.set_yticks(range(len(words_list)))
        ax.set_yticklabels(words_list)
        ax.set_xlabel('Frequência', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count, i, f' {count}', va='center', fontsize=10, fontweight='bold')
    else:
        bars = ax.bar(range(len(words_list)), counts, color='steelblue', edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(words_list)))
        ax.set_xticklabels(words_list, rotation=45, ha='right')
        ax.set_ylabel('Frequência', fontsize=12, fontweight='bold')
        
        # Add value labels
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(count)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_title(f'Top {top_n} Palavras Mais Frequentes', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='x' if orientation == 'horizontal' else 'y')
    
    plt.tight_layout()
    return fig


def create_phrase_frequency_chart(
    phrases: List[str],
    top_n: int = 15,
    figsize: Tuple[int, int] = (12, 8)
) -> Figure:
    """
    Create phrase frequency bar chart
    
    Args:
        phrases: List of phrases
        top_n: Number of top phrases to show
        figsize: Figure size
        
    Returns:
        Matplotlib Figure
    """
    phrase_counts = Counter(phrases)
    most_common = phrase_counts.most_common(top_n)
    
    if not most_common:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, 'Nenhuma frase encontrada', 
                ha='center', va='center', fontsize=16)
        ax.axis('off')
        return fig
    
    phrases_list, counts = zip(*most_common)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars = ax.barh(range(len(phrases_list)), counts, color='coral', edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(phrases_list)))
    ax.set_yticklabels(phrases_list)
    ax.set_xlabel('Frequência', fontsize=12, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count, i, f' {count}', va='center', fontsize=10, fontweight='bold')
    
    ax.set_title(f'Top {top_n} Frases Mais Frequentes', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, axis='x')
    
    plt.tight_layout()
    return fig


def calculate_text_statistics(texts: List[str]) -> pd.DataFrame:
    """
    Calculate various text statistics
    
    Args:
        texts: List of texts
        
    Returns:
        DataFrame with statistics
    """
    stats = {
        'Total de Documentos': len(texts),
        'Documentos Válidos': sum(1 for t in texts if isinstance(t, str) and t.strip()),
        'Total de Caracteres': sum(len(str(t)) for t in texts if isinstance(t, str)),
        'Média de Caracteres': np.mean([len(str(t)) for t in texts if isinstance(t, str)]),
        'Total de Palavras': sum(len(str(t).split()) for t in texts if isinstance(t, str)),
        'Média de Palavras': np.mean([len(str(t).split()) for t in texts if isinstance(t, str)]),
        'Palavras Únicas': len(set(extract_words(texts))),
    }
    
    df = pd.DataFrame(list(stats.items()), columns=['Métrica', 'Valor'])
    
    # Format values
    df['Valor'] = df['Valor'].apply(lambda x: f'{x:.2f}' if isinstance(x, float) else str(int(x)))
    
    return df
