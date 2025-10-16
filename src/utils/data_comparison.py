import pandas as pd

# function for comparing two columns of the dataset
def compare_before_after(df, original_col, cleaned_col, num_samples=10):
    print("\n" + "=" * 100)
    print(" " * 40 + "BEFORE vs AFTER")
    print("=" * 100)
    print(f"\n{'ID':<5} {'BEFORE':<45} {'AFTER':<45}")
    print("-" * 100)

    # showing specific number of examples:
    for i in range(min(num_samples, len(df))):
        original = df.iloc[i][original_col][:42] + "..." if len(df.iloc[i][original_col]) > 45 else df.iloc[i][original_col]
        cleaned = df.iloc[i][cleaned_col][:42] + "..." if len(df.iloc[i][cleaned_col]) > 45 else df.iloc[i][cleaned_col]

        print(f"{i+1:<5} {original:<45} {cleaned:<45}")

    print("-" * 100)

# function to get the text stats:
def get_text_stats(df, text_col):
    stats = {
        'total_texts': len(df),
        'avg_length': df[text_col].str.len().mean(),
        'min_length': df[text_col].str.len().min(),
        'max_length': df[text_col].str.len().max(),
        'avg_words': df[text_col].str.split().str.len().mean(),
        'total_chars': df[text_col].str.len().sum()
    }

    return stats

# printing the above stats:
def print_text_stats(stats, title="Text Statistics"):
    print("\n" + "=" * 60)
    print(" " * 20 + title)
    print("=" * 60)

    print(f"\n📊 Total texts: {stats['total_texts']}")
    print(f"📏 Average length: {stats['avg_length']:.1f} characters")
    print(f"📏 Min length: {stats['min_length']} characters")
    print(f"📏 Max length: {stats['max_length']} characters")
    print(f"📝 Average words: {stats['avg_words']:.1f} words")
    print(f"📝 Total characters: {stats['total_chars']:,}")

    print("\n" + "=" * 60)

