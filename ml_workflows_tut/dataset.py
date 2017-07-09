import pandas as pd
import numpy as np


def load_dataset(filename, filetype="csv", header=True):
    """
    Loads a dataset from file
    Parameters:
    ---------
    filename: str
    Name of data file
    filetype: str
    The type of data file(csv, tsv)
    returns:
    ------
    DataFrame
    Dataset as pandas DataFrame
    """

    in_file = open(filename)
    data = []
    header_row = ''

    # Read the file line by line into instance structure
    for line in in_file.readlines():
        # Skip comments
        if not line.startswith("#"):
            # TSV file
            if filetype == 'tsv':
                if header:
                    header_row = line.strip().split('\t')
                else:
                    raw = line.strip().split('\t')
            # CSV file
            elif filetype == "csv":
                if header:
                    header_row = line.strip().split(',')
                else:
                    raw = line.strip().split(',')
            else:
                print('Invalid File type')
                exit()
            # Append to dataset appropriately
            if not header:
                data.append(raw)
            header = False
    
    # Build a new dataframe of the data instance list of lists and return
    df = pd.DataFrame(data, columns=header_row)
    return df

def to_numeric(dataset, attr_name):
    """
    Performs a simple categorical to numeric attribute value transformation
    Parameters:
    ---------
    dataset: DataFrame
    Dataset on which to perform transformation
    attr_name: str
    Dataset attribute to convert from nominal to numeric values

    Returns:
    --------
    DataFrame
    DataFrane with data transformation perform
    dict
    Python dictionary of attribute to integer mappings
    """

    # Get unique entries in column
    unique_vals = dataset[attr_name].unique()
    # Create dict
    val_dict = {}
    for val in unique_vals:
        if not val in val_dict:
            val_dict[val] = len(val_dict)
    # Replace values in attr_name col as per dict
    dataset[attr_name].replace(val_dict, inplace=True)
    # Return dataset and value dictionary
    return dataset, val_dict

def to_matrix(dataset):
    """
    Convert pandas DaaFrame dataset to numpy matrix representation
    Parameters:
    ---------
    dataset: DataFrame
    Dataset to convert to matrix representation
    Returns:
    -------
    ndarray
    numpy array representation of dataset
    """
    return dataset.as_matrix()

def from_str(dataset, attrs):
    """
    Performs numeric values stored as strings to numeric value transformation
    Parameters:
    ----------
    dataset: DataFrame
    Dataset on which to perform transformation
    attrs: str/list
    Dataset attribute name(s) to convert from strings to equivalent values 
    
    Returns:
    --------
    DataFrame
    DataFrame with data transformation performed
    """

    # Make conversions on list of attributes
    if type(attrs) == list:
        for attr_name in attrs:
            dataset[attr_name] = dataset[attr_name].astype(float)
    # Make conversion on single attribute
    else:
        dataset[attrs] = dataset[attrs].astype(float)

    # Return dataset after conversion
    return dataset


if __name__ == "__main__":
    # Load dataset
    data = load_dataset('iris.csv')
    print(data.head())
    # Change categorical class names to numeric values
    data, data_dict = to_numeric(data, 'species')
    print(data.head())
    print(data_dict)
    # Convert dataset to matrix representation
    ds = to_matrix(data)
    print(ds[:10])
