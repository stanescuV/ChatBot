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

    for ix, row in enumerate(df.values):

        intrebari_test.append(row[0])
        
        if ix == 0:
            break
    return intrebari_test

intrebari_test = get_intrebari_test_from_csv()
print(intrebari_test)

          