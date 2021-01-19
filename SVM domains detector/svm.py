from libsvm.svm import svm_problem, svm_parameter
from libsvm.svmutil import *
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import pandas
 
from matplotlib import pyplot
 
import copy
import numpy as np
import math
 
#global naive = True
 
def make_string(my_array):
    ret = ''
    for item in my_array:
        if item != '0':
            ret+=item
    return ret
def subsequences_finished(mask, training_q):
    for item in range(len(mask)):
        if item < len(mask)-training_q and mask[item] != '0':
            #print('Im the one stopping this, item is:', item, 'and we have:', mask[item])
            return False
        elif item >= len(mask)-training_q and mask[item] == '0':
            #print('not me')
            return False
    return True
 
def find_length(mask):
    max = -1
    min = 999
    lowest_idx = 999999
    for i in range(len(mask)):
        if mask[i] != '0':
            if i < lowest_idx:
                lowest_idx = i
            if i < min:
                min = i
            elif i > max:
                max = i
    result = max-min+1
 
    #result = len(mask)-lowest_idx
    return result
 
def get_subsequences(training_input, training_q):
    done = []
    result = {}
    ultimate_result = []
    for slovo in training_input:
        template = [letter for letter in slovo]
        mask = ['0' for letter in slovo]
        for i in range(training_q):
            mask[i] = template[i]
        if make_string(mask) not in done:
            done.append(make_string(mask))
            result[make_string(mask)] = find_length(mask)
 
        index_to_shift = training_q-1
        position_of_index = index_to_shift
        counter =1
        mask_length = len(mask)
 
        # creating a dictionary of predecessor positions
        indexes = [i for i in range(len(mask))]
        predecessors = [i - 1 for i in range(len(mask))]
        ancestry = dict(zip(indexes, predecessors))
        ancestry_placements = copy.deepcopy(ancestry)
        mask_log = []
        while not subsequences_finished(mask, training_q):
            #print(mask)
            length = find_length(mask)
            ultimate_result.append([make_string(mask), length])
            if mask not in mask_log:
                done.append(make_string(mask))
 
                    #print('length:', length, 'mask:', mask)
                result[make_string(mask)] = length
                mask_log.append(mask)
 
                #print(done)
 
            if position_of_index == mask_length - counter and ancestry_placements[index_to_shift] != position_of_index-1:###if I ran the entire course of the string and my predecessor isn't right next to me, I shift the predecessor closer and place myself on his right
 
 
                shifted = False
                current = index_to_shift
                mask[position_of_index] = '0'
                #print('index to shift:', current)
 
                while not shifted:
                    if mask[ancestry_placements[current]+1] == '0':
                        mask[ancestry_placements[current]] = '0'
                        mask[ancestry_placements[current] + 1] = template[ancestry_placements[current] + 1]
                        ancestry_placements[current] = ancestry_placements[current]+1
                        shifted = True
                    else:
                        #print('decreasing to ancestor')
                        current -=1
                for successor in range(current,training_q):
                    #print('successor:', successor, 'position will be:', ancestry_placements[successor]+1)
                    index_to_shift = successor
                    position_of_index = ancestry_placements[index_to_shift]+1
                    ancestry_placements[index_to_shift+1] = position_of_index
                    mask[position_of_index] = template[position_of_index]
                #print('and now, mask is:', mask, 'index is:', index_to_shift, 'position is:', position_of_index, 'ancestor positions are', ancestry_placements)
 
 
            elif index_to_shift == -1 or (position_of_index == mask_length - counter and ancestry[index_to_shift] == -1):
                break  
            elif position_of_index == mask_length - counter and position_of_index-1 == ancestry_placements[index_to_shift]: # and predecessor position is just next to me....we need to shift to the preceeding predecessor
                #print('so we here, mask is:', mask, 'index to shift:', index_to_shift, 'position:', position_of_index)
                current = position_of_index
                mask[position_of_index] = '0'
                while mask[current-1] != '0': # we have a predecessor blocking us
                    mask[ancestry_placements[current]] = '0'
                    #print('ok, mask now is;', mask)
                    current = ancestry[current]
                mask[current] = '0'
                #print('first we clear...', mask, 'index:', index_to_shift, 'position:', position_of_index, 'ancestors:', ancestry_placements, 'current:', current)
                cnt = 0
                for i in mask:
                    if i != '0':
                        cnt+=1
                index_to_shift = cnt-1
                position_of_index = ancestry_placements[index_to_shift+1]
                #print('currently,', mask)
                mask[position_of_index] = '0'
                #position_of_index+=1
                while index_to_shift < training_q:
                    position_of_index+=1
                    mask[position_of_index] = template[position_of_index]
                    ancestry_placements[index_to_shift+1] = position_of_index
                    index_to_shift +=1
                index_to_shift-=1
 
            else: ####normal shift
                mask[position_of_index] = '0'
                position_of_index += 1
                mask[position_of_index] = template[position_of_index]
 
    length = find_length(mask)
    ultimate_result.append([make_string(mask), length])
    ultimate_result.sort()
    return ultimate_result
 
 
 
#def get_kernel(training_input, training_q, training_lambda):
    #subsequences = get_subsequences(training_input, training_q)
    #return True
 
def determine_lambdas(training_lambda, dict, subsequences_dict):
    ret = [0]
    size = 0
    for item in subsequences_dict.items():
        ret.append(0)
        size+=1
 
    #print('ret:', ret, 'individual dict:', dict, 'subsequence dict:', subsequences_dict)
    for key, value in subsequences_dict.items():
        if value in dict.keys():
            ret[key] = training_lambda ** dict[value]
        else:
            ret[key] = 0
    return ret
 
def dot(first, second):
    if len(first) != len(second):
        #print("something wrong")
        return -1
    return sum(i[0] * i[1] for i in zip(first, second))
 
 
def naive_l2_normalise(dict_of_lengths, dot_products):
    '''
    naive implementation of L2 normalisation
    :param dict_of_lengths:
    :param dot_products:
    :return:
    '''
    ret = np.zeros((4, 4), dtype =float)
    y_pos = 0
    x_pos = 0
    for y in ret:
        for x in y:
            ret[y_pos, x_pos] = dot_products[y_pos, x_pos] /(math.sqrt(dot_products[y_pos, y_pos] )*math.sqrt(dot_products[x_pos, x_pos] ))
            x_pos+=1
        x_pos = 0
        y_pos+=1
    return ret
 
def l2_normalise(table):
    ret = ret = np.zeros((4, 4), dtype =float)
    y_pos = 0
    x_pos = 0
    for y in ret:
        for x in y:
            ret[y_pos, x_pos] = table[y_pos, x_pos]/((table[y_pos][y_pos]*table[x_pos][x_pos])**.5)
            x_pos+=1
        x_pos = 0
        y_pos+=1
    return ret
 
 
def subseq_kernel(stra, strb, training_q, training_lambda):
    subseq_a = get_subsequences([stra], training_q)
    subseq_b = get_subsequences([strb], training_q)
    counter = 0
    for i in subseq_a:
        for x in subseq_b:
            if i[0] == x[0]:
                counter+=training_lambda**(i[1]+x[1])
    return counter
 
def my_print(table, training_input):
    x_clo = "{:>25}" * (len(training_input) + 1)
    print(x_clo.format("", *training_input))
    for domain, row in zip(training_input, table):
        print(x_clo.format(domain, *row))
 
def my_first_print(table, training_input):
    x_clo = "{:>25}" * (len(training_input) + 1)
    counter = [1, 2, 3, 4, 5]
    print(x_clo.format("", *counter))
 
    for domain, row in zip(training_input, table):
        print(x_clo.format(domain, *row))
 
 
 
def train(K, labels, C):
    K = sparse.hstack((1+np.arange(len(labels))[:,None], K)).A
    prob = svm_problem(labels, K, isKernel=True)
    param = svm_parameter('-t 4 -c ' + str(C))
    return svm_train(prob, param)
 
 
def evaluate(K, labels, model):
    K = sparse.hstack((1+np.arange(len(labels))[:,None], K)).A
    pred_labels, accuracy, _ = svm_predict(labels, K, model)
    counter = 0
    for i in range(len(pred_labels)):
        if pred_labels[i] == labels[i]:
            counter+=1
    return 1-(counter/len(pred_labels))
 
 
 
 
 
if __name__ == '__main__':
 
    q_parameter = 2
    lambda_parameter = 0.4
 
    [train_x, train_y] = load_svmlight_file("data/kernel/trn_kernel_mat.svmlight")
    [val_x, val_y] = load_svmlight_file("data/kernel/val_kernel_mat.svmlight")
    [test_x, test_y] = load_svmlight_file("data/kernel/tst_kernel_mat.svmlight")
 
    record_models = []
 
    validate_errs = []
    train_errs = []
    sv_vector_count = []
    c = [0.01, 0.1, 1.0, 10.0, 100.0]
    for constant in c:
        model = train(train_x, train_y, constant)
        record_models.append(model)
        # training errors
        train_errs.append(evaluate(train_x, train_y, model))
 
        sv_vector_count.append(len(model.get_SV()))
        # validation errors
        validate_errs.append(evaluate(val_x, val_y, model))
 
    #print_table(
      #  ['C', 'Training Error', 'Validation Error', 'Number of support vectors'],
      #  [constant, train_errs, validate_errs, sv_vector_count]
    #)
    print("********     Part 1      *************")
    tbl1 = np.zeros((4, 5), dtype=float)
    tbl1[0] = c
    tbl1[1] = train_errs
    tbl1[2] = validate_errs
    tbl1[3] = sv_vector_count
    #my_first_print(tbl1, ["C", "Training Error", "Validation Error", "Number of support vectors"])
    print(pandas.DataFrame([c, train_errs, validate_errs, sv_vector_count], ["C", "Training Error", "Validation Error", "Number of support vectors"]))
    plt.plot(c, train_errs , '-g')
    plt.plot(c, validate_errs, '-b')
    plt.xscale('log')
    plt.grid(True)
    plt.legend(['Training errors', 'Validation errors'])
 
    plt.show()
 
    # part 2
    print("********     Part 2      *************")
    min_err = 9999999999
    min_idx = -1
    pos = 0
    for i in range(len(validate_errs)):
        if validate_errs[i]< min_err:
            min_err = validate_errs[i]
            min_idx = i
 
    print('the best C is :', c[min_idx])
    t_error = evaluate(test_x, test_y, record_models[min_idx])
    print(f'the test error is: {round(t_error, 4)} rounded to 4 s.f.')
    z_99 = 2.326
    print(f'Epsilon is: {round(2**(-.5)*(math.log(2 / 0.01) / (len(test_y))) ** .5, 4)} rounded to 4 s.f.')
 
    # part 3
    print("********     Part 3      *************")
 
    training_q = 3
    training_lambda = 0.4
    training_input = ['google.com', 'facebook.com', 'atqgkfauhuaufm.com', 'vopydum.com']
 
    dict_of_dicts = {}
    subsequences_dict = {}
    subsequences = 0
    domains = 0
 
    non_normalised = np.zeros((4, 4), dtype=float)
    for i in range(4):
        for y in range(4):
            non_normalised[i][y] = subseq_kernel(training_input[i], training_input[y], training_q, training_lambda)
 
    print(
        '******************************************** Table conveying SSK ****************************************************')
 
    my_print(non_normalised, training_input)
    print(
        '************************************* Table conveying normalised SSK ****************************************************')
    my_print(l2_normalise(non_normalised), training_input)
