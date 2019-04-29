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

def get_soc_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        num_results = int(lines[0])
        num_rankings = int(lines[num_results + 1].split(',')[2])
        data = []
        for line in lines[num_results + 2:]:
            nums = line.split(',')
            for i in range(int(nums[0])): # number of times same ranking repeats in data
                data.append(nums[1:])
        data = np.array(data).astype(int)
    return data

def shorten_file_name(file_name):
    parts = file_name[:-4].split("-")
    return "ED-" + str(int(parts[1])) + "-" + str(int(parts[2])) + '.csv'

def main():
    for file_name in os.listdir("data/preflib/dirty_15"):
        print(file_name)
        data = get_soc_data("data/preflib/dirty_15/" + file_name)
        shortened_file_name = shorten_file_name(file_name)
        with open("data/preflib/clean_15/" + shortened_file_name, 'w') as f:
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
