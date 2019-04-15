import sys
import numpy as np

# Read in the data from given file
def read_csv(file, has_header=True):
    data = np.loadtxt(file, dtype=float, delimiter=',', skiprows=has_header)
    return data

# Gets data from CSV file and converts it appropriately
def get_data(data_file):
    return read_csv(data_file).astype(int)

def main():
    data_path = sys.argv[1]
    num_to_subsample = int(sys.argv[2])
    new_data_file = sys.argv[3]
    data = get_data(data_path)
    subsample = data[np.random.randint(data.shape[0], size=num_to_subsample), :]
    with open(new_data_file, 'w') as f:
        num_items = len(subsample[0])
        line_str = "# " + (num_items-1)*"x," + "x"
        f.write(line_str + "\n")
        for ranking in subsample:
            line_str = ""
            for j in range(len(ranking)-1):
                line_str += str(ranking[j]) + ","
            line_str += str(ranking[len(ranking)-1])
            f.write(line_str + "\n")

if __name__ == '__main__':
    main()
