#!/usr/python
import math
from itertools import count
from __builtin__ import str
import random
import sys

attr_list = []
first_value = ""
second_value = ""
#value_predict = 0
#value_actual = 0
count_accurate = 0
count_inaccurate = 0

class Node:
    name = ""
    type = ""
    children = []
    splits = []
    
class AttributeClass:
    name = ""
    type = ""  
    index = 0
    values = []      
    def set_members(self, attr_mem, index):    
        self.name = attr_mem[0].strip("'")
        if (attr_mem[1] == "{") :
            self.type = "nominal"    
        else:
            self.type = "numeric" 
        self.index = index
        if (len(attr_mem) > 2 ): #to extract the entire list of attribute values
            self.values += attr_mem[2:] 
            if (len(self.values) > 0 ):
                count = 0
                for i in self.values:
                    self.values[count]=self.values[count][:-1]
                    count += 1
    def __str__(self):
        return str(self.name)
    def display_attrs(self):
        print self.name
        print self.type
        print self.index
        print self.values
'''
Function to read input file and parse data
'''
def read_data(data_to_parse):
    global first_value
    global second_value
    file_txt = open(data_to_parse)
    data = []
    dataset = [[]]
    lines = [line.rstrip('\n') for line in file_txt]
    count = 0 #to keep track of index of each attribute in attribute list
    for item in lines:       
        if item.startswith("@relation"):
            relation_name = item[10:]
        if item.startswith("@attribute"):
            attr_mem = item[11:].split(" ") 
            attribute = AttributeClass()
            attribute.values = []
            attribute.set_members(attr_mem,count)
            attr_list.append(attribute)
            count+=1            
        elif item.startswith("@data"):
            index_of_data = lines.index("@data")
            data = lines[(index_of_data+1):]
            for i in range(len(data)):
                dataset.append(data[i].split(","))
            del dataset[0]
    first_value = attr_list[-1].values[0]
    for i in dataset:
        if (i[len(dataset[0])-1]!=first_value):
            second_value = i[len(dataset[0])-1]
            break
    return dataset

'''
Function to find subset tuple
'''            
def findSubset(dataset):
    count_tuple = []
    first_count = 0.0
    second_count = 0.0
    for i in dataset:
        if (i[len(dataset[0])-1]==first_value):
            first_count+=1
        else:
            second_count+=1
    count_tuple.append(first_count)
    count_tuple.append(second_count)
    return count_tuple

'''
Function to calculate entropy
'''
def entropy_calc(dataset):
    count_tuple = findSubset(dataset)
    first_count = count_tuple[0]
    second_count = count_tuple[1]
    r1 = first_count/(first_count+second_count)
    r2 = second_count/(first_count+second_count)
    if (r1!=0 and r2 != 0):
        entropy = -1*(r1*math.log(r1,2)+r2*math.log(r2,2))
    else:
        entropy = 0
    return entropy    

'''
Function to calculate information gain
'''
def infoGainCalc(dataset, attribute, entropy_dataset):
    count = 0
    list = []
    if (attribute.type == "nominal"): # find entropy for each nominal feature
        sublist = []
        entropy_subset = 0
        for i in attribute.values:
            sublist = []
            for j in dataset:
                if (j[:][attribute.index]==i):
                    sublist.append([j[:][attribute.index],j[:][len(j)-1]])
            if (len(sublist) > 0):
                entropy_subset += (len(sublist)*1.0/len(dataset)*1.0)*entropy_calc(sublist)
            else:
                entropy_subset += 0 #if there are no entries for the attribute value in the data
        return [entropy_dataset-entropy_subset, attribute.index,"zzz"] 
        
    if (attribute.type == "numeric"):
        for i in dataset: 
            list.append([float(dataset[count][attribute.index]), dataset[count][len(dataset[0])-1]]) #TODO float
            count += 1
        list = sorted(list) 
        list_infogain = []
        splits_list = []
        entropy_subset = 0
        for j in range(len(list)-1):
            if(list[j][0] == list[j+1][0]):
                continue
            else:
                class1=[]
                class2=[]
                for pair in list:
                    if(pair[0]==list[j][0]):
                        if(pair[1] not in class1):
                            class1.append(pair[1])
                    if(pair[0]==list[j+1][0]):
                        if(pair[1] not in class2):
                            class2.append(pair[1])
                if(len(class1)==len(class2) and len(class1)==1):
                    if(class1[0]!=class2[0]):
                        splits_list.append((float(list[j][0])+float(list[j+1][0]))/2)
                if(len(class1)==len(class2) and len(class1)==2):
                    splits_list.append((float(list[j][0])+float(list[j+1][0]))/2)
                if (len(class1)!=len(class2)):
                    splits_list.append((float(list[j][0])+float(list[j+1][0]))/2)
        
        for i in splits_list:
            partition1 = []
            partition2 = [] 
            for j in list:
                if (float((j[0]))<=i):
                    partition1.append(j)
                else:
                    partition2.append(j)
            partition1_entropy = entropy_calc(partition1)
            partition2_entropy = entropy_calc(partition2)
            x = entropy_dataset
            split_entropy = (len(partition1)*1.0/len(list)*1.0)*(partition1_entropy*1.0) + (len(partition2)*1.0/len(list)*1.0)*(partition2_entropy*1.0)       
            x-=split_entropy
            list_infogain.append(x)
        if(len(list_infogain)==0):
            return -1 # no CSs
        index_max_info_gain = list_infogain.index(max(list_infogain))
        return [ max(list_infogain), attribute.index, splits_list[index_max_info_gain]] #appending the best Candidate split here

'''
Function to find Candidate Splits
'''
def findBestSplit(dataset,attr_list):
    entropy_dataset = entropy_calc(dataset)
    if(len(dataset)==82):
        5
    final_list_info_gain = []
    for attribute in attr_list:
        if(attribute.name!="class"):
            if (attribute.index != len(attr_list)-1):
                info_gain_value = infoGainCalc(dataset, attribute, entropy_dataset)
                if (info_gain_value == -1): # no more CSs
                    final_list_info_gain.append([-1, attribute.index, ""])
                else: 
                    final_list_info_gain.append(info_gain_value)
    max_info_gain = 0     #max_info_gain = final_list_info_gain[0][0]
    index_candidate = 0
    for i in final_list_info_gain:
        if (i[0] > max_info_gain):
            max_info_gain = i[0]
            index_candidate = final_list_info_gain.index(i)
    if (max_info_gain == 0):
        return [-1, "", ""]
    return final_list_info_gain[index_candidate]

def makeSubset(dataset, feature_index, classification, indicator):
    subset = []
    if (indicator == "zzz"):
        for row in dataset:
            if(row[feature_index] == classification):

                subset.append(row)
    elif (indicator == "less"):
        for row in dataset:
            if (float(row[feature_index]) <= classification):
                subset.append(row)
    elif (indicator == "greater"):
        for row in dataset:
            if (float(row[feature_index]) > classification):
                subset.append(row)            
    return subset
'''
Function to build the tree
'''
def makeTree(dataset,m, depth):
    subset_tuple = []
    format_string = "" 
    for i in range(depth):
        format_string+="|       "
    depth+=1
    best_split = findBestSplit(dataset,attr_list) # best_split = [info_gain, index, zzz/numeric CD split]
    childList = []
    if (best_split[0] == -1):
        root = Node()
        subset_tuple = findSubset(dataset)
        subset_tuple[0] = int(round(subset_tuple[0]))
        subset_tuple[1] = int(round(subset_tuple[1]))
        if (subset_tuple[0] < subset_tuple[1]):
            print format_string + attr_list[len(attr_list)-1].values[1]
            leaf = Node()
            leaf.type="leaf"
            leaf.name=attr_list[len(attr_list)-1].values[1]
            childList.append(leaf)    
                                   
        elif(subset_tuple[0] >= subset_tuple[1]):
            print format_string + attr_list[len(attr_list)-1].values[0]
            leaf = Node()
            leaf.type="leaf"
            leaf.name=attr_list[len(attr_list)-1].values[0]
            childList.append(leaf)
    

    else : 
        root = Node()
        root.name = attr_list[best_split[1]].name     
        if (best_split[2] == "zzz"):
            root.type = "nominal"
            root.splits=attr_list[best_split[1]].values
            for classification in attr_list[best_split[1]].values :
                subset = makeSubset(dataset, best_split[1], classification, "zzz")
                subset_tuple = findSubset(subset)
                subset_tuple[0] = int(round(subset_tuple[0]))
                subset_tuple[1] = int(round(subset_tuple[1]))
                if (len(subset) == 0):
                    print format_string + attr_list[best_split[1]].name + " = " + classification + " [" + str(subset_tuple[0]) + " " +  str(subset_tuple[1]) + "]" + ": " + attr_list[len(attr_list)-1].values[0]
                    leaf = Node()
                    leaf.type="leaf"
                    leaf.name=attr_list[len(attr_list)-1].values[0]
                    childList.append(leaf)
                     
                elif (subset_tuple[1]==0):
                    print format_string + attr_list[best_split[1]].name + " = " + classification + " [" + str(subset_tuple[0]) + " " +  str(subset_tuple[1]) + "]" + ": " + attr_list[len(attr_list)-1].values[0]
                    leaf = Node()
                    leaf.type="leaf"
                    leaf.name=attr_list[len(attr_list)-1].values[0]
                    childList.append(leaf)
                
                elif (subset_tuple[0]==0):
                    print format_string + attr_list[best_split[1]].name + " = " + classification + " [" + str(subset_tuple[0]) + " " +  str(subset_tuple[1]) + "]" + ": " + attr_list[len(attr_list)-1].values[1]
                    leaf = Node()
                    leaf.type="leaf"
                    leaf.name=attr_list[len(attr_list)-1].values[1]
                    childList.append(leaf) 
                
                elif (sum(subset_tuple) < m):
                    if (subset_tuple[0] < subset_tuple[1]):
                        print format_string + attr_list[best_split[1]].name + " = " + classification + " [" + str(subset_tuple[0]) + " " +  str(subset_tuple[1]) + "]" + ": " + attr_list[len(attr_list)-1].values[1]
                        leaf = Node()
                        leaf.type="leaf"
                        leaf.name=attr_list[len(attr_list)-1].values[1]
                        childList.append(leaf)    
                                       
                    elif(subset_tuple[0] >= subset_tuple[1]):
                        print format_string + attr_list[best_split[1]].name + " = " + classification + " [" + str(subset_tuple[0]) + " " +  str(subset_tuple[1]) + "]" + ": " + attr_list[len(attr_list)-1].values[0]
                        leaf = Node()
                        leaf.type="leaf"
                        leaf.name=attr_list[len(attr_list)-1].values[0]
                        childList.append(leaf) 
               
                else:
                    print format_string + attr_list[best_split[1]].name + " = " + classification + " " + "[" + str(subset_tuple[0]) + " " +  str(subset_tuple[1]) + "]"
                    childNode = Node()
                    childNode = makeTree(subset,m,depth)
                    childList.append(childNode)
        else:
            root.type = "numeric"
            root.splits=best_split[2]
            if (best_split[0] == -1): #Condition for no more CSs or no +ve information gain
                if (subset_tuple[0] < subset_tuple[1]):
                    print format_string + attr_list[best_split[1]].name + " <= " + '{0:.06f}'.format(best_split[2]) + " " + "[" + str(subset_tuple[0]) + " " +  str(subset_tuple[1]) + "]" + ": " + attr_list[len(attr_list)-1].values[1]
                    leaf = Node()
                    leaf.type="leaf"
                    leaf.name=attr_list[len(attr_list)-1].values[1]
                    childList.append(leaf) 
                    
                elif(subset_tuple[0] >= subset_tuple[1]):
                    print format_string + attr_list[best_split[1]].name + " <= " + '{0:.06f}'.format(best_split[2]) + " " + "[" + str(subset_tuple[0]) + " " +  str(subset_tuple[1]) + "]" + ": " + attr_list[len(attr_list)-1].values[0]
                    leaf = Node()
                    leaf.type="leaf"
                    leaf.name=attr_list[len(attr_list)-1].values[0]
                    childList.append(leaf)
            subset1 =  makeSubset(dataset, best_split[1], best_split[2], "less")
            subset_tuple = findSubset(subset1)
            subset_tuple[0] = int(round(subset_tuple[0]))
            subset_tuple[1] = int(round(subset_tuple[1]))
            if (len(subset1) == 0):
                print format_string + attr_list[best_split[1]].name + " <= " + '{0:.06f}'.format(best_split[2]) + " " + "[" + str(subset_tuple[0]) + " " +  str(subset_tuple[1]) + "]" + ": " + attr_list[len(attr_list)-1].values[0]
                leaf = Node()
                leaf.type="leaf"
                leaf.name=attr_list[len(attr_list)-1].values[0]
                childList.append(leaf)  
             
            elif (subset_tuple[1]==0):
                print format_string + attr_list[best_split[1]].name + " <= " + '{0:.06f}'.format(best_split[2]) + " " + "[" + str(subset_tuple[0]) + " " +  str(subset_tuple[1]) + "]" + ": " + attr_list[len(attr_list)-1].values[0]
                leaf = Node()
                leaf.type="leaf"
                leaf.name=attr_list[len(attr_list)-1].values[0]
                childList.append(leaf)       
                
            elif (subset_tuple[0]==0):
                print format_string + attr_list[best_split[1]].name + " <= " + '{0:.06f}'.format(best_split[2]) + " " + "[" + str(subset_tuple[0]) + " " +  str(subset_tuple[1]) + "]" + ": " + attr_list[len(attr_list)-1].values[1]
                leaf = Node()
                leaf.type="leaf"
                leaf.name=attr_list[len(attr_list)-1].values[1]
                childList.append(leaf) 
                
            elif (sum(subset_tuple) < m):
                if (subset_tuple[0] < subset_tuple[1]):
                    print format_string + attr_list[best_split[1]].name + " <= " + '{0:.06f}'.format(best_split[2]) + " " + "[" + str(subset_tuple[0]) + " " +  str(subset_tuple[1]) + "]" + ": " + attr_list[len(attr_list)-1].values[1]
                    leaf = Node()
                    leaf.type="leaf"
                    leaf.name=attr_list[len(attr_list)-1].values[1]
                    childList.append(leaf) 
                    
                elif(subset_tuple[0] >= subset_tuple[1]):
                    print format_string + attr_list[best_split[1]].name + " <= " + '{0:.06f}'.format(best_split[2]) + " " + "[" + str(subset_tuple[0]) + " " +  str(subset_tuple[1]) + "]" + ": " + attr_list[len(attr_list)-1].values[0]
                    leaf = Node()
                    leaf.type="leaf"
                    leaf.name=attr_list[len(attr_list)-1].values[0]
                    childList.append(leaf) 
            else:
                print format_string + attr_list[best_split[1]].name + " <= " + '{0:.06f}'.format(best_split[2]) + " " + "[" + str(subset_tuple[0]) + " " +  str(subset_tuple[1]) + "]"
                childNode = Node()
                childNode = makeTree(subset1,m,depth)
                childList.append(childNode)
            
            subset2 =  makeSubset(dataset, best_split[1], best_split[2], "greater")
            subset_tuple = findSubset(subset2)
            subset_tuple[0] = int(round(subset_tuple[0]))
            subset_tuple[1] = int(round(subset_tuple[1]))
            if (len(subset2) == 0):
                print format_string + attr_list[best_split[1]].name + " <= " + '{0:.06f}'.format(best_split[2]) + " " + "[" + str(subset_tuple[0]) + " " +  str(subset_tuple[1]) + "]" + ": " + attr_list[len(attr_list)-1].values[0]
                leaf = Node()
                leaf.type="leaf"
                leaf.name=attr_list[len(attr_list)-1].values[0]
                childList.append(leaf)
            
            elif (subset_tuple[1]==0):
                print format_string + attr_list[best_split[1]].name + " > " + '{0:.06f}'.format(best_split[2]) + " " + "[" + str(subset_tuple[0]) + " " +  str(subset_tuple[1]) + "]"+ ": " + attr_list[len(attr_list)-1].values[0]
                leaf = Node()
                leaf.type="leaf"
                leaf.name=attr_list[len(attr_list)-1].values[0]
                childList.append(leaf)
                
            elif (subset_tuple[0]==0):
                print format_string + attr_list[best_split[1]].name + " > " + '{0:.06f}'.format(best_split[2]) + " " + "[" + str(subset_tuple[0]) + " " +  str(subset_tuple[1]) + "]" + ": " + attr_list[len(attr_list)-1].values[1]
                leaf = Node()
                leaf.type="leaf"
                leaf.name=attr_list[len(attr_list)-1].values[1]
                childList.append(leaf) 
    
            elif (sum(subset_tuple) < m):
                if (subset_tuple[0] < subset_tuple[1]):
                    print format_string + attr_list[best_split[1]].name + " > " + '{0:.06f}'.format(best_split[2]) + " " + "[" + str(subset_tuple[0]) + " " +  str(subset_tuple[1]) + "]" + ": " + attr_list[len(attr_list)-1].values[1]
                    leaf = Node()
                    leaf.type="leaf"
                    leaf.name=attr_list[len(attr_list)-1].values[1]
                    childList.append(leaf)
                    
                elif(subset_tuple[0] >= subset_tuple[1]):
                    print format_string + attr_list[best_split[1]].name + " > " + '{0:.06f}'.format(best_split[2]) + " " + "[" + str(subset_tuple[0]) + " " +  str(subset_tuple[1]) + "]" + ": " + attr_list[len(attr_list)-1].values[0]
                    leaf = Node()
                    leaf.type="leaf"
                    leaf.name=attr_list[len(attr_list)-1].values[0]
                    childList.append(leaf)
                    
            else:
                print format_string + attr_list[best_split[1]].name + " > " + '{0:.06f}'.format(best_split[2]) + " " + "[" + str(subset_tuple[0]) + " " +  str(subset_tuple[1]) + "]"
                childNode = Node()
                childNode = makeTree(subset2,m,depth)
                childList.append(childNode)
        
    root.children = childList
    return root
    
def testTheTree(tree, row, count):
    #global value_predict
    #global value_actual
    global count_accurate
    global count_inaccurate
    
    index = 0
    for i in attr_list:
        if (i.name == tree.name):
            break
        index += 1
    if (tree.type == "nominal"):
        index_of_split = tree.splits.index(row[index])
        testTheTree(tree.children[index_of_split], row, count)
    elif (tree.type == "numeric"):
        if (float(row[index]) <= float(tree.splits)):
            testTheTree(tree.children[0], row, count)
        else:
            testTheTree(tree.children[1], row, count)
    else: 
        if (tree.name == ""):
            if (tree.children[0].name == row[-1]):
                count_accurate+=1
            else:
                count_inaccurate+=1
        elif(tree.name==row[-1]):
            count_accurate+=1
        else:
            count_inaccurate+=1
        print "%3d: Actual: %s  Predicted: %s" %(count, row[-1], tree.name)
            
if __name__ == '__main__':
    
    
    m = int(sys.argv[3])
    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    
    '''
    train_file_name = "heart_train_sample.arff"
    test_file_name = "heart_test.arff"
    m = 2
    '''

    print "" 
    file_txt = open(train_file_name)    
    train_dataset = read_data(train_file_name) #parse the data
    Tree = Node()
    Tree = makeTree(train_dataset,m, 0)
    test_dataset = read_data(test_file_name)
    count = 1
    print "<Predictions for the Test Set Instances>"
    for row in test_dataset:
        testTheTree(Tree, row, count)
        count += 1
    test_instances = len(test_dataset)
    print "Number of correctly classified: " + str(count_accurate)+ "  Total number of test instances: "+ str(test_instances)

    '''
    train_dataset = read_data(train_file_name) #parse the data
    length_of_sample = int(len(train_dataset)*.10)
    sample_train_set = random.sample(train_dataset, length_of_sample)
    print sample_train_set
    Tree = makeTree(sample_train_set, 4, 0)
    test_dataset = read_data(test_file_name)
    count = 1
    print "<Predictions for the Test Set Instances>"
    for row in test_dataset:
        testTheTree(Tree, row, count)
        count += 1
    test_instances = len(test_dataset)
    print "Number of correctly classified: " + str(count_accurate)+ "  Total number of test instances: "+ str(test_instances)
    '''
    '''
    print "Start of Part 2"
    for per in [0.2, 0.5, 0.10, 0.20, 0.50, 1.00]:
        length_of_sample = int(len(train_dataset)*per)
        array_info = []
        for i in range(10):
            sample_train_set = random.sample(train_dataset, length_of_sample)
            print "Sample Train set" + str(sample_train_set)
            Tree = makeTree(sample_train_set, 4, 0)
            count = 1
            count_accurate = 0
            count_inaccurate = 0
            for row in test_dataset:
                testTheTree(Tree, row, count)
               count += 1
            test_instances = len(test_dataset)
            print "Iteration " + str(i) + " : " + "Number of correctly classififed: " + str(count_accurate)+ "  Total number of test instances: "+ str(test_instances)
            array_info.append(round(100*float(count_accurate)/(count_accurate+count_inaccurate), 2))
        print "per : " + str(per)
        print array_info
        print str(min(array_info)) + ", " + str(sum(array_info)/len(array_info)) + ", " + str(max(array_info))
    '''
  
    
    
