from src.data_loader.load_data import load_data, show_data_info

from src.preprocessing.text_cleaner import preprocess_dataframe, show_cleaning_comparison

from src.utils.data_comparison import compare_before_after, get_text_stats, print_text_stats

from src.preprocessing.tokenizer import (
    process_dataframe_tokens,
    show_tokenization_examples,
    get_token_statistics,
    print_token_stats
)

import os

def main():
    # Printing welcome message
    print("\n" + "🧠" * 30)
    print(" " * 20 + "MENTAL HEALTH ML PROJECT")
    print(" " * 25 + "Step 1: Data Loading")
    print("🧠" * 30 + "\n")

    data_path = 'data/sample1000.csv'

    if not os.path.exists(data_path):
        print(f"❌ Error: File not found at {data_path}")
        print("\n💡 Please create the CSV file first!")
        print("   Location: project/data/sample100.csv")
        return
    
    # Step 1: Loading the data:
    print("📂 Loading the data ...")
    print("-" * 70)
    try:
        df = load_data(data_path)
        print(f"✅ Loader {len(df)} samples")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Step 2: Clean Text
    print("\n\n🧹 Step 2: Cleaning Text ...")
    print("-" * 70)
    df_cleaned = preprocess_dataframe(df)
    print("✅ Text Cleaning Complete")

    # showing the cleaning examples:
    show_cleaning_comparison(df_cleaned, num_examples=3)

    # Step 3: TOKENIZATION
    print("\n\n🔤 STEP 3: Tokenization & Stop-word Removal...")
    print("-" * 70)

    df_tokenized = process_dataframe_tokens(df_cleaned)
    print(f"\n✅ Tokennization Completed!")

    show_tokenization_examples(df_tokenized, num_examples=5)

    # showing the comparison
    print("\n📊 Comparing: Cleaned Text vs Stop-words Removed")
    compare_before_after(df_tokenized, 'cleaned_text', 'filtered_text', num_samples=8)

    token_stats = get_token_statistics(df_tokenized)
    print_token_stats(token_stats)

    print("\n💡 Key Insights:")
    print("-" * 70)
    print(f"• Original text had average {token_stats['avg_tokens_before']:.1f} words per sample")
    print(f"• After removing stop words: {token_stats['avg_tokens_after']:.1f} words per sample")
    print(f"• We removed {token_stats['total_stopwords_removed']:,} stop words total")
    print(f"• This keeps only the meaningful words for mental health classification")

    # saving the processed data:
    output_path = 'data/processed/tokenized_data.csv'
    os.makedirs('data/processed', exist_ok=True)

    # Convert tokens list to string for CSV storage
    df_tokenized['tokens_str'] = df_tokenized['tokens'].apply(lambda x: ' '.join(x))
    df_tokenized.to_csv(output_path, index=False)
    print(f"\n💾 Tokenized data saved to: {output_path}")

    print("\n" + "=" * 70)
    print(" " * 20 + "✅ STEPS 1-3 (Loading, Cleaning, Tokenization) COMPLETE!")
    print("=" * 70)
    print("\n📝 Next Step: Feature Extraction (TF-IDF)")


if __name__ == "__main__":
    main()