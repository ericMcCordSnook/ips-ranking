import os
import sys
import numpy as np

# Read in the data from given file
def read_csv(file, has_header=False):
    data = np.loadtxt(file, dtype=float, delimiter=',', skiprows=has_header)
    return data

# Gets data from CSV file and converts it appropriately
def get_data(data_file):
    return read_csv(data_file).astype(int)

def write_data_to_file(freq_tbl, file_name):
    with open(file_name, 'w') as f:
        f.write("# x,x,x,x\n")
        for perm, freq in freq_tbl.items():
            perm_str = str(perm)[1:-1].replace(' ','')
            for i in range(freq):
                f.write(perm_str + "\n")

def main():
    for file_name in os.listdir("data/preflib_dirty"):
        print(file_name)
        data = get_data("data/preflib_dirty/" + file_name)
        with open("data/preflib/" + file_name, 'w') as f:
            num_items = len(data[0])
            line_str = "# " + (num_items-1)*"x," + "x"
            f.write(line_str + "\n")
            for ranking in data:
                line_str = ""
                for j in range(len(ranking)-1):
                    line_str += str(ranking[j]) + ","
                line_str += str(ranking[len(ranking)-1])
                f.write(line_str + "\n")

if __name__ == '__main__':
    main()
