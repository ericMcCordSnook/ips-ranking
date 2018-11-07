from utils.miscutils import get_data, get_frequency_distribution

def main():
    experiments = [2, 9, 16, 18, 19, 24, 25, 31, 38, 40, 41, 42, 45, 48, 49]
    data_file_base = "data/random/raw/rand_exp_4_50_"
    freq_tble_base = "data/random/freq_tbls/rand_exp_4_50_"
    for exp in experiments:
        dataset = get_data(data_file_base + str(exp) + ".csv")
        freq_tbl = get_frequency_distribution(dataset)
        with open(freq_tble_base + str(exp) + ".txt", "w") as f:
            for key, val in freq_tbl.items():
                f.write(str(key) + " : " + str(val) + "\n") 

if __name__ == '__main__':
    main()
