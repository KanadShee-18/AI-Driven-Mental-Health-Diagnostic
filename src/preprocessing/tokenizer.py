import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class Tokenizer:
    
    def __init__(self, language='english'):
        self.language = language
        self.stop_words = None
        
        # Download NLTK data automatically
        self._download_nltk_data()
        
        # Load stop words
        self._load_stopwords()
    
    def _download_nltk_data(self):
        required_packages = ['punkt_tab', 'punkt', 'stopwords']
        
        for package in required_packages:
            try:
                nltk.download(package, quiet=True)
            except Exception as e:
                # Some packages might not exist, continue anyway
                pass
    
    def _load_stopwords(self):
        try:
            # Try to load from NLTK
            self.stop_words = set(stopwords.words(self.language))
        except Exception as e:
            # Fallback: use basic stop words list
            print("⚠️  Using basic stopwords list (NLTK data not fully available)")
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
                'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
                'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
                'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
                'against', 'between', 'into', 'through', 'during', 'before',
                'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
                'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then',
                'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
                'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
                'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'
            }
    
    def tokenize(self, text):
        try:
            # Use NLTK's word_tokenize (handles punctuation better)
            tokens = word_tokenize(text)
        except Exception as e:
            # Fallback: simple split by spaces
            tokens = text.split()
        
        return tokens
    
    def remove_stopwords(self, tokens):
        # Keep only tokens that are NOT in stop words
        filtered_tokens = [token for token in tokens if token not in self.stop_words]
        
        return filtered_tokens
    
    def process(self, text):
        # Step 1: Tokenize
        tokens = self.tokenize(text)
        original_count = len(tokens)
        
        # Step 2: Remove stop words
        filtered_tokens = self.remove_stopwords(tokens)
        filtered_count = len(filtered_tokens)
        
        # Step 3: Join back to text
        filtered_text = ' '.join(filtered_tokens)
        
        # Step 4: Calculate statistics
        removed_count = original_count - filtered_count
        
        return {
            'tokens': filtered_tokens,
            'text': filtered_text,
            'original_count': original_count,
            'filtered_count': filtered_count,
            'removed_count': removed_count
        }
    
    def process_batch(self, texts):
        return [self.process(text) for text in texts]


# ============= Helper Functions for DataFrames =============

def process_dataframe_tokens(df, text_column='cleaned_text'):
    # Create tokenizer instance
    tokenizer = Tokenizer()
    
    # Create copy to avoid modifying original
    df_copy = df.copy()
    
    # Process each text
    print("🔤 Tokenizing texts...")
    results = tokenizer.process_batch(df_copy[text_column].tolist())
    
    # Extract results into columns
    df_copy['tokens'] = [r['tokens'] for r in results]
    df_copy['filtered_text'] = [r['text'] for r in results]
    df_copy['stopwords_removed'] = [r['removed_count'] for r in results]
    df_copy['filtered_word_count'] = [r['filtered_count'] for r in results]
    
    return df_copy


def show_tokenization_examples(df, num_examples=5):
    print("\n" + "=" * 90)
    print(" " * 30 + "TOKENIZATION EXAMPLES")
    print("=" * 90)
    
    for i in range(min(num_examples, len(df))):
        print(f"\nExample {i+1}:")
        print(f"  Original     : {df.iloc[i]['text']}")
        print(f"  Cleaned      : {df.iloc[i]['cleaned_text']}")
        print(f"  Tokens       : {df.iloc[i]['tokens']}")
        print(f"  After Filter : {df.iloc[i]['filtered_text']}")
        print(f"  Words Removed: {df.iloc[i]['stopwords_removed']}")
        print(f"  Label        : {df.iloc[i]['label']}")
    
    print("\n" + "=" * 90)


def get_token_statistics(df):
    if 'word_count' in df.columns:
        avg_tokens_before = df['word_count'].mean()
    else:
        avg_tokens_before = df['filtered_word_count'].mean() + df['stopwords_removed'].mean()
    stats = {
        'total_samples': len(df),
        'avg_tokens_before': avg_tokens_before,
        'avg_tokens_after': df['filtered_word_count'].mean(),
        'total_stopwords_removed': df['stopwords_removed'].sum(),
        'avg_stopwords_per_text': df['stopwords_removed'].mean(),
        'min_tokens': df['filtered_word_count'].min(),
        'max_tokens': df['filtered_word_count'].max()
    }
    
    return stats


def print_token_stats(stats):
    print("\n" + "=" * 70)
    print(" " * 20 + "TOKENIZATION STATISTICS")
    print("=" * 70)
    
    print(f"\n📊 Total samples processed: {stats['total_samples']}")
    print(f"\n📝 Tokens per text:")
    print(f"   Before stop-word removal: {stats['avg_tokens_before']:.1f} words")
    print(f"   After stop-word removal : {stats['avg_tokens_after']:.1f} words")
    print(f"   Minimum tokens          : {stats['min_tokens']} words")
    print(f"   Maximum tokens          : {stats['max_tokens']} words")
    
    print(f"\n🚫 Stop words removed:")
    print(f"   Total removed           : {stats['total_stopwords_removed']:,}")
    print(f"   Average per text        : {stats['avg_stopwords_per_text']:.1f} words")
    
    if stats['avg_tokens_before'] > 0:
        reduction = ((stats['avg_tokens_before'] - stats['avg_tokens_after']) / stats['avg_tokens_before']) * 100
        print(f"   Reduction percentage    : {reduction:.1f}%")
    
    print("\n" + "=" * 70)