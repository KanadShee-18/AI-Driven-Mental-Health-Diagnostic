from src.data_loader.load_data import load_data, show_data_info
from src.preprocessing.text_cleaner import preprocess_dataframe, show_cleaning_comparison
from src.utils.data_comparison import compare_before_after, get_text_stats, print_text_stats

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
    
    print("📂 Loading 1000 data from the CSV file ...")
    try:
        df = load_data(data_path)
        print("✅ Data loaded sucessfully!\n\n")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # showing the data
    show_data_info(df)

    # print a success message for our first step
    print("\n" + "=" * 60)
    print(" " * 20 + "✅ STEP 1 COMPLETE!")
    print("=" * 60)

    # Step: Text Preprocessing
    print("\n\n🧹 Step 2: Preprocessing Text Data ...")
    print("-" * 60)

    # cleaning the text in the dataframe
    df_cleaned = preprocess_dataframe(df)
    print("✅ Text Cleaning Complete!\n")

    # showing the cleaning examples:
    show_cleaning_comparison(df_cleaned, num_examples=5)

    # comparing before and after:
    compare_before_after(df_cleaned, 'text', 'cleaned_text', num_samples=10)

    # getting stats for original text
    print("\n📊 Original Text Statistics:")
    original_stats = get_text_stats(df_cleaned, 'text')
    print_text_stats(original_stats, "BEFORE CLEANING")

    # getting stats for cleaned text
    print("\n📊 Cleaned Text Statistics:")
    cleaned_stats = get_text_stats(df_cleaned, 'cleaned_text')
    print_text_stats(cleaned_stats, "BEFORE CLEANING")

    # showing the impact of text cleaning:
    print("\n💡 Cleaning Impact:")
    print("-" * 60)
    char_reduced = ((original_stats['avg_length'] - cleaned_stats['avg_length']))
    print(f"Average text reduced by: {char_reduced:.1f}%")
    print(f"Total Characters Removed: {original_stats['total_chars'] - cleaned_stats['total_chars']}")

    # storing the cleaned data
    output_path = 'data/processed/cleaned_data.csv'
    os.makedirs('data/processed', exist_ok=True)
    df_cleaned.to_csv(output_path, index=False)
    print(f"\n📉 Cleaned data saved to: {output_path}")

    # print a success message for our first step
    print("\n" + "=" * 60)
    print(" " * 20 + "✅ STEP 2: Preprocessing text COMPLETE!")
    print("=" * 60)



if __name__ == "__main__":
    main()