import pandas as pd

# USE CASE FOR LOOP EMBEDDING QUESTIONS FOR TESTING
intrebari_test = []

def get_intrebari_test_from_csv():
    """
    Reads a CSV file containing legal questions and returns a list of questions.
    Returns:
        list: A list of legal questions.
    """
    df = pd.read_csv("questions_penal_code_ro.csv")

    for row in enumerate(df.values):
        intrebari_test.append(row)
        
    return intrebari_test




def write_in_csv(ragas_text_obj, filename="ragas.csv"):
    """
    Writes context into column B and GPT answer into column C of a CSV file,
    with NO headers. Column A is left empty.
    """
    df = pd.DataFrame([
        [str(ragas_text_obj)]   # A empty, B context, C answer
    ])

    df.to_csv(filename, index=False, header=False)# Example usage:
intrebari_test = get_intrebari_test_from_csv()
# print(intrebari_test)

# Example: write one row
# write_in_csv(["Ce faci ? "],["Art. 1", "Art. 2", "Art. 3"], "This is GPT's answer.")
