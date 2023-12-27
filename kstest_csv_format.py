### Title: kstest_csv_format.py
### Author: Ming Wang
### Purpose: By opening up any given csv the code below
### should function so that it yields and returns the ks#. 
### from the given inputted data.

import csv
from openpyxl import load_workbook
import numpy as np
import matplotlib.pyplot as plt
from numpy import trapz
import math



def data_frame(int_weight_zero, int_weight_one, cumu_repZero_list, cumu_repOne_list,
               cdf_repZero_list, cdf_repOne_list, max_cdf_diff, fpr, tpr, csv_list, max_ks,
               min_list, max_list, avg_list, decile_list, marginal_resp, cum_resp):
    workbook = load_workbook(filename="ks_testing.xlsx")
    sheet = workbook.active
    auc = np.trapz(tpr, fpr)
    # Adding column headers to the sheet
    sheet["G5"] = "Interval Weight"
    sheet["H5"] = "Cumu"
    sheet["I5"] = "CDF"
    sheet["J5"] = "Interval Weight"
    sheet["K5"] = "Cumu"
    sheet["L5"] = "CDF"
    sheet["M5"] = "KS"
    sheet["P5"] = "# of Mails"
    sheet["P6"] = str(len(csv_list))
    sheet["Q5"] = "Resp%"
    sheet["Q6"] = str(float(cumu_repOne_list[-1]/len(csv_list)))
    sheet["R5"] = "KS"
    sheet["R6"] = str(max_ks)
    sheet["S5"] = "ROC"
    sheet["S6"] = str(auc)
    sheet["P7"] = "bin"
    sheet["Q7"] = "Min"
    sheet["R7"] = "Max"
    sheet["S7"] = "Avg"
    sheet["T7"] = "# of Mail"
    sheet["U7"] = "Dist % Mail"
    sheet["V7"] = "Marginal Resp%"
    sheet["W7"] = "Cum Resp%"

    # Update the range for writing data to the sheet
    start_row = 6
    end_row = start_row + len(int_weight_zero)

    for i, x in enumerate(int_weight_zero, start=start_row):
        sheet["G" + str(i)] = str(x)

    for i, x in enumerate(cumu_repZero_list, start=start_row):
        sheet["H" + str(i)] = str(x)

    for i, x in enumerate(cdf_repZero_list, start=start_row):
        sheet["I" + str(i)] = str(x)

    for i, x in enumerate(int_weight_one, start=start_row):
        sheet["J" + str(i)] = str(x)

    for i, x in enumerate(cumu_repOne_list, start=start_row):
        sheet["K" + str(i)] = str(x)

    for i, x in enumerate(cdf_repOne_list, start=start_row):
        sheet["L" + str(i)] = str(x)

    for i, x in enumerate(max_cdf_diff, start=start_row):
        sheet["M" + str(i)] = str(x)

    # Update the range for writing min, max, avg data to the sheet

    i = 8
    for x in range(10):
        sheet["P" + str(x+i)] = str(x+1) + "."
        sheet["T" + str(x+i)] = str(decile_list[x+1]-decile_list[x]) 
        sheet["Q" + str(x+i)] = str(min_list[x]*100) 
        sheet["R" + str(x+i)] = str(max_list[x]*100) 
        sheet["S" + str(x+i)] = str(avg_list[x]) 
        sheet["U" + str(x+i)] = str(float(((decile_list[x+1]-decile_list[x])/len(csv_list)))*100) + "%"   
        sheet["V" + str(x+i)] = str(marginal_resp[x]) 
        sheet["W" + str(x+i)] = str(cum_resp[x])       


    sheet["N3"] = max(max_cdf_diff)

    workbook.save(filename=str(get_file_name()))
    
  
    print('Area under the ROC curve:', auc)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend()
    plt.show()

def get_file_name():
    '''
    Asks the user for an output file name
    Paramter: N/A
    Returns: name: output file name
    '''
    name = input('Enter Output File Name: ')
    return name

def open_file(file_name):
    '''
    Opens and reads the csv file into code and returns the data
    within the file as a 2d list
    Paramter: file_name:the name of the file wanting to be accessed
    Returns: csv_list: 2d list of all the data within the csv file
    '''
    csv_list = []
    with open(file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for line in csv_reader:
            csv_list.append(line)
    return csv_list

def ks_test_cumu_tot(csv_list):
    cumu_repZero = 0
    cumu_repOne = 0
    # Calculating cumulative sums based on conditions
    for line in csv_list:
        if line[0].isnumeric():
            if int(line[3]) == 0:
                cumu_repZero += float(line[1])
            if int(line[3]) != 0:
                cumu_repOne += float(line[1])
    return cumu_repZero, cumu_repOne

def organize_data_csv(csv_list):
    # Sorting the CSV data based on the last column
    sorted_list = sorted(csv_list[1:], key=lambda x: float(x[-1]))
    return sorted_list

def ks_cdf(csv_list, tot_cumu_repZero, tot_cumu_repOne):
    cdf_repZero = float(0)
    cdf_repOne = float(0)
    cumu_repZero = 0
    cumu_repOne = 0
    max_ks_cdf = float(0)

    # Calculating KS CDF values
    for line in csv_list:
        if line[0].isnumeric():
            if int(line[3]) == 0:
                cumu_repZero += float(line[1])
                cdf_repZero = float((cumu_repZero / tot_cumu_repZero))
            if int(line[3]) != 0:
                cumu_repOne += float(line[1])
                cdf_repOne = float((cumu_repOne / tot_cumu_repOne))
            if abs(cdf_repZero - cdf_repOne) > max_ks_cdf:
                max_ks_cdf = abs(cdf_repZero - cdf_repOne)
    return max_ks_cdf

def column_list(csv_list, tot_cumu_repZero, tot_cumu_repOne):
    int_weight_zero = []
    int_weight_one = []
    cumu_repZero_list = []
    cumu_repOne_list = []
    cdf_repZero_list = []
    cdf_repOne_list = []
    max_cdf_diff = []
    cdf_repZero = float(0)
    cdf_repOne = float(0)
    cumu_repZero = 0
    cumu_repOne = 0
    ks_cdf_diff = float(0)

    # Organizing data into separate lists for each column
    for line in csv_list:
        if line[0].isnumeric():
            if int(line[3]) == 0:
                int_weight_zero.append(float(line[1]))
                int_weight_one.append(0)
                cumu_repZero += float(line[1])
                cumu_repZero_list.append(cumu_repZero)
                cumu_repOne_list.append(cumu_repOne)
                cdf_repZero = float((cumu_repZero / tot_cumu_repZero))
                cdf_repZero_list.append(cdf_repZero * 100)
                cdf_repOne_list.append(cdf_repOne * 100)
            if int(line[3]) != 0:
                int_weight_one.append(float(line[1]))
                int_weight_zero.append(0)
                cumu_repOne += float(line[1])
                cumu_repOne_list.append(cumu_repOne)
                cumu_repZero_list.append(cumu_repZero)
                cdf_repOne = float((cumu_repOne / tot_cumu_repOne))
                cdf_repOne_list.append(cdf_repOne * 100)
                cdf_repZero_list.append(cdf_repZero * 100)
            ks_cdf_diff = abs(cdf_repZero - cdf_repOne)
            max_cdf_diff.append(ks_cdf_diff)
    # Calculating True Positive Rate (TPR) and False Positive Rate (FPR)
    tpr = np.array(cumu_repOne_list) / tot_cumu_repOne
    fpr = np.array(cumu_repZero_list) / tot_cumu_repZero

    # Sorting the TPR and FPR arrays based on ascending FPR
    sorted_indices = np.argsort(fpr)
    tpr = tpr[sorted_indices]
    fpr = fpr[sorted_indices]

    return (
        int_weight_zero,
        int_weight_one,
        cumu_repZero_list,
        cumu_repOne_list,
        cdf_repZero_list,
        cdf_repOne_list,
        max_cdf_diff,
        tpr,
        fpr,
    )

def total_response(tot_cumu_repOne, int_weight_one):
    return tot_cumu_repOne / len(int_weight_one)

### max and min is referring the max and min in the bins
def min_max_avg(decile_dict, csv_list):
    min_vals = []
    max_vals = []
    avg_vals = []

    for i in range(0,9):
        min_vals.append(csv_list[decile_dict[i]][4])
        max_vals.append(csv_list[decile_dict[i+1]][4])
        values = [float(csv_list[j][4])* 100 for j in range(decile_dict[i], decile_dict[i+1])]
        avg_vals.append(sum(values)/len(values))

    min_vals.append(csv_list[decile_dict[9]][4])
    max_vals.append(csv_list[(decile_dict[-1]-1)][4])
    vals = [float(csv_list[j][4])* 100 for j in range(decile_dict[9], decile_dict[-1])]
    avg_vals.append(sum(vals)/len(vals))

    return min_vals, max_vals, avg_vals


def marginal_percent(int_weight_one, decile_dict):
    marginal_per = []
    for i in range(0, 10):
        count = 0
        start = int(decile_dict[i])
        end = int(decile_dict[i+1])
        
        if end - start != 0:  # Check if the denominator is zero
            for j in range(start, end):
                if int_weight_one[j] == 1:
                    count += 1
            response_per = count / (end - start)
        else:
            response_per = 0  # Handle division by zero case
        
        marginal_per.append(response_per)
    return marginal_per

def cum_resp(marginal_per):
    cumulative_average = []
    total = 0
    count = 0
    for x in reversed(marginal_per):
        total += x
        count += 1
        average = total / count
        cumulative_average.append(average)

    return list(reversed(cumulative_average)) 

def decile(int_weight_one):
    
    indices = [int(len(int_weight_one) * (i / 10)) for i in range(1, 10)]
    indices.insert(0,0)
    indices.insert(10, len(int_weight_one))
    return indices


def main():
    file_name = input('Enter CSV File Name: ')
    csv_list = open_file(file_name)
    csv_list = organize_data_csv(csv_list)
    tot_cumu_repZero, tot_cumu_repOne = ks_test_cumu_tot(csv_list)
    max_ks_per = ks_cdf(csv_list, tot_cumu_repZero, tot_cumu_repOne)
    print(max_ks_per * 100)
    (
        int_weight_zero,
        int_weight_one,
        cumu_repZero_list,
        cumu_repOne_list,
        cdf_repZero_list,
        cdf_repOne_list,
        max_cdf_diff,
        fpr,
        tpr,
    ) = column_list(csv_list, tot_cumu_repZero, tot_cumu_repOne)

    decile_dict = decile(int_weight_one)
    print(decile_dict)
    min_list, max_list, avg_list = min_max_avg(decile_dict, csv_list)
    print(min_list)
    print(max_list)
    print(avg_list)
    marginal_per = marginal_percent(int_weight_one, decile_dict)
    cum_list = cum_resp(marginal_per)
    print(marginal_per)
    print(cum_list)
    data_frame(
        int_weight_zero,
        int_weight_one,
        cumu_repZero_list,
        cumu_repOne_list,
        cdf_repZero_list,
        cdf_repOne_list,
        max_cdf_diff,
        fpr,
        tpr,
        csv_list,
        max_ks_per,
        min_list,
        max_list,
        avg_list,
        decile_dict,
        marginal_per,
        cum_list,
    )

if __name__ == "__main__":
    main()