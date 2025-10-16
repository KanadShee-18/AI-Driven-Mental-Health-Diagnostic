# Cleaning the user prompt

import re
import string

# converting into lowercase:
def to_lowercase(text):
    return text.lower()

# removing special characters
def remove_special_chars(text):
    """
    regex = [^a-zA-Z\s] -> anything that is not a letter or space
    """
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text);
    return cleaned_text

# removing numbers (unusual for mental health)
def remove_numbers(text):
    removed_nums = re.sub(r'\d+', '', text)
    return removed_nums

# Removing extra spaces
def remove_extra_spaces(text):
    space_removed = ' '.join(text.split())
    return space_removed

# Applying all above steps in the input
def cleaned_text(text):
    text = to_lowercase(text)
    text = remove_special_chars(text)
    text = remove_numbers(text)
    text = remove_extra_spaces(text)

    return text

# Cleaned dataframe:
def preprocess_dataframe(df, text_column='text'):
    # didn't manipulating the main data so making a copy
    df_copy = df.copy();

    # applying the cleaned_text() function in eacn row of the specific column
    df_copy['cleaned_text'] = df_copy[text_column].apply(cleaned_text)

    # Making some analysis for the cleaned text

    # length of cleaned text
    df_copy['text_length'] = df_copy['cleaned_text'].str.len()

    # counting number of words in the cleaned text
    df_copy['cleaned_word_count'] = df_copy['cleaned_text'].str.split().str.len()

    return df_copy

# Displaying the cleaning data by comparing with previous one:
def show_cleaning_comparison(df, num_examples=5):
    print("\n" + "=" * 80)
    print(" " * 30 + "CLEANING EXAMPLES")
    print("=" * 80)

    for i in range(min(num_examples, len(df))):
        print(f"\nExample {i + 1}")
        print(f"    Original : {df.iloc[i]['text']}")
        print(f"    Cleaned  : {df.iloc[i]['cleaned_text']}")
        print(f"    Label    : {df.iloc[i]['label']}")

    print("\n" + "=" * 80)