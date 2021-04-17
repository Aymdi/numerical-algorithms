import numpy as np

def init_tests(function_name):
    test_result = {}
    number_of_test = {}

    for name in function_name :
        test_result[name] = 0
        number_of_test[name] = 0
    
    return (test_result, number_of_test)

def update_test(condition, name, test_result):
    if condition :
        test = '\033[32mPASSED\033[0m'
        test_result[name] += 1
    else :
        test = '\033[31mFAILED\033[0m'

    return test
    
def main_test(name, entree, res, hope, test_result, number_of_test):

    error_threshold = 0.000001
    if type(hope)==list :
        test = update_test(
            abs(res[0]-hope[0]) < error_threshold and abs(res[1]-hope[1]) < error_threshold, 
            name, 
            test_result
        )
    elif type(hope)==np.ndarray :
        w = hope.shape[0]
        h = hope.shape[1]
        test = update_test(
            np.all(np.add(-1*res,hope) < np.ones([w,h]) * error_threshold), 
            name, 
            test_result
        )
    else :
        test = update_test(
            abs(res-hope) < error_threshold, 
            name, 
            test_result
        )

    number_of_test[name]+=1
    
    entrees=""
    for i in range(len(entree)):
        if i == 0 :
            entrees += str(entree[i])
        else :
            entrees += ","+str(entree[i])
        
    print(6*" ","{}({})={} ; Expected result={}".format(name,entrees,res,hope))
    print(6*" ","---> test {}".format(test))
    return None

def print_summary(function_name,test_result,number_of_test):
    print(6*"-","SUMMARY",6*"-")
    for name in function_name :
        if test_result[name] == number_of_test[name] :
            test = '\033[32mPASSED\033[0m'
        else :
            test = '\033[31mFAILED\033[0m'
        print("{} : {}({}/{})".format(name,test,test_result[name],number_of_test[name]))
    return None