import csv
import os
import pandas as pd


def load_data(dataset):
    """
    Load a dataset from the list of available datasets and preprocess our data.
    Args:
        dataset: path of dataset
    """
    # Extract the directory path from the input file path
    input_path = os.path.dirname(dataset)

    # Create the output file path by combining the directory and the new filename
    output_path = os.path.join(input_path, 'reddit_posts_processed.csv')

    # open the input and output file
    with open(dataset, 'r', encoding='utf-8', errors='replace') as csv_infile, \
            open(output_path, 'w', newline='', encoding='utf-8') as csv_outfile:

        # create csv readers
        csv_reader = csv.reader(csv_infile)
        csv_writer = csv.writer(csv_outfile)

        # read csv files
        for row in csv_reader:
            try:
                decoded_row = [cell.encode('utf-8').decode('utf-8', 'replace') for cell in row]
                # write the encoded file to a new file
                csv_writer.writerow(decoded_row)
            except UnicodeDecodeError:
                # If cannot be encoded, then jump to next line
                continue

    print("Rows that do not conform to UTF-8 encoding have been read and filtered, and the result is saved to the reddit_posts_processed.csv file.")

    df = pd.read_csv(output_path)

    # Remove duplicate titles
    #df = df.drop_duplicates(subset='title', keep='first')

    print("----------------------------------------------------------------")
    print("The processed data has {} lines".format(len(df)))
    print("----------------------------------------------------------------")

    # save the dataframe to the output path
    df.to_csv(output_path, index=False)

    return df