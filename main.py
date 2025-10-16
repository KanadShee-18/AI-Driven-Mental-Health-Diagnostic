from src.data_loader.load_data import load_data, show_data_info

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

if __name__ == "__main__":
    main()