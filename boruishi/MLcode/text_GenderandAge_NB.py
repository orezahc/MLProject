
import sys
import csv
import os
import codecs
import nltk
import math
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist



reload(sys)
sys.setdefaultencoding('cp1252')

def baysianTraining(filepath):
    

    #estimator M
    M = 1
    
    #file_dict, lables of all training example
    file_dict = {}

    '''gender predefine starts'''
    #gender possibility
    p_male = 0
    p_female = 0
    num_male = 0
    num_female = 0
    
    #possibility of word i appear when its male/female, dictionary <word, possibility>
    p_word_male_dict = {}
    p_word_female_dict = {}

    #number of words of all/male/female text file
    total_num = 0
    total_num_male = 0
    total_num_female = 0

    #dictionary
    dict_id_gender = {}
    
    #if the tempfile path does not exit, build it
    #tempfilefolder = os.getenv("HOME")+"//tempfile"
    tempfilefolder ="C:\\Users\\Deman\\Desktop\\temp"
    if(os.path.isdir(tempfilefolder) == False):
        os.makedirs(tempfilefolder)
        pass

    #big text file path
    #bigText_path = os.getenv("HOME")+"/tempfile/bigtext.txt"
    bigText_path = "C:\\Users\\Deman\\Desktop\\temp\\bigtext.txt"

    
    #all male text path, all female text path
    #text_male_path = os.getenv("HOME")+"/tempfile/text_male.txt"
    text_male_path ="C:\\Users\\Deman\\Desktop\\temp\\text_male.txt"
    #text_female_path = os.getenv("HOME")+"/tempfile/text_female.txt"
    text_female_path ="C:\\Users\\Deman\\Desktop\\temp\\text_female.txt"
    '''gender predefine ends'''

    '''age predefine starts'''
    #age group possibility
    p_group1 = 0
    p_group2 = 0
    p_group3 = 0
    p_group4 = 0
    num_group1 = 0
    num_group2 = 0
    num_group3 = 0
    num_group4 = 0

    #possibility of word i appear when its group1/group2/group3/group4
    #, dictionary <word, possibility>
    p_word_group1_dict = {}
    p_word_group2_dict = {}
    p_word_group3_dict = {}
    p_word_group4_dict = {}

    #number of words of group1/group2/group3/group4 text file
    total_num_group1 = 0
    total_num_group2 = 0
    total_num_group3 = 0
    total_num_group4 = 0

    #dictionary, shows <userid, agegroup>
    dict_id_group = {}

    #all age groups text path
    #text_group1_path = os.getenv("HOME")+"/tempfile/text_group1.txt"
    text_group1_path ="C:\\Users\\Deman\\Desktop\\temp\\text_group1.txt"
    #text_group2_path = os.getenv("HOME")+"/tempfile/text_group2.txt"
    text_group2_path ="C:\\Users\\Deman\\Desktop\\temp\\text_group2.txt"
    #text_group3_path = os.getenv("HOME")+"/tempfile/text_group3.txt"
    text_group3_path ="C:\\Users\\Deman\\Desktop\\temp\\text_group3.txt"
    #text_group4_path = os.getenv("HOME")+"/tempfile/text_group4.txt"
    text_group4_path ="C:\\Users\\Deman\\Desktop\\temp\\text_group4.txt"
    '''age predefine ends'''
    
    fdist = FreqDist()
    fdist_male = FreqDist()
    fdist_female = FreqDist()
    fdist_group1 = FreqDist()
    fdist_group2 = FreqDist()
    fdist_group3 = FreqDist()
    fdist_group4 = FreqDist()
    
    #read gender/ages infomation from profile.csv
    #and calculate the P of male and female, and build <userid, gender> dict
    with open(filepath) as file:
        file_dict = csv.DictReader(file)
        count = 0
        for row in file_dict:
            #compute gender possibility
            if(row['gender'] == "0.0"):
                num_male += 1
            else:
                num_female += 1 
            #build <userid,gender> dictionary
            dict_id_gender[row['userid']] = row['gender']
            
            #compute agegroup possibility
            
            each_age = float(row['age'])
            if(each_age <= 24):
                num_group1 += 1
                dict_id_group[row['userid']] = 1
            elif(each_age >= 25 and each_age <= 34):
                num_group2 += 1
                dict_id_group[row['userid']] = 2
            elif(each_age >= 35 and each_age <=49):
                num_group3 += 1
                dict_id_group[row['userid']] = 3
            else:
                num_group4 += 1
                dict_id_group[row['userid']] = 4
            
            count += 1
            pass
        p_male = float(num_male) / count
        p_female = 1 - p_male
        p_group1 = float(num_group1) / count
        p_group2 = float(num_group2) / count
        p_group3 = float(num_group3) / count
        p_group4 = 1 - p_group1 - p_group2 - p_group3
        pass

    '''read/write operation starts'''
    #read the text data
    #compute the the amount of the text file
    textdata_file_path = "C:\\Users\\Deman\\Desktop\\FacebookDataTCSS555Project\\TCSS555\\Train\\Text"
    for root,dirs, files in os.walk(textdata_file_path):
        #open the big text file, and clear it
        bigtext_file_clear = open(bigText_path, 'w')
        bigtext_file_clear.write("")
        bigtext_file_writer = open(bigText_path,'a')
        #open the text_male.txt and text_female.txt, clear them first
        text_male_file_clear = open(text_male_path,'w')
        text_female_file_clear = open(text_female_path,'w')
        text_male_file_writer = open(text_male_path,'a')
        text_female_file_writer = open(text_female_path,'a')
        text_male_file_clear.write("")
        text_female_file_clear.write("")
        #open the text_group1/2/3/4.txt, clear them first
        text_group1_file_clear = open(text_group1_path,'w')
        text_group2_file_clear = open(text_group2_path,'w')
        text_group3_file_clear = open(text_group3_path,'w')
        text_group4_file_clear = open(text_group4_path,'w')
        text_group1_file_clear.write("")
        text_group2_file_clear.write("")
        text_group3_file_clear.write("")
        text_group4_file_clear.write("")
        text_group1_file_writer = open(text_group1_path,'a')
        text_group2_file_writer = open(text_group2_path,'a')
        text_group3_file_writer = open(text_group3_path,'a')
        text_group4_file_writer = open(text_group4_path,'a')
        #for each text file, write its text to big text file
        for eachname in files:
            #get the owner's gender/agegroup of each text file
            each_userid = eachname.split('.')[0]
            each_gender = dict_id_gender[each_userid]
            each_group = dict_id_group[each_userid]
            each_textfile_path = os.path.join(root, eachname)
            with open(each_textfile_path,'r') as each_textfile:
                text = each_textfile.readlines()
                for line in text:
                    #write its text to big text file
                    bigtext_file_writer.write(line + " ")
                    #if the author is male, put into text_male_file.txt,
                    #otherwise put into text_female_file.txt
                    if(each_gender == "0.0"):
                        text_male_file_writer.write(line + " ")
                    else:
                        text_female_file_writer.write(line + " ")
                    if(each_group == 1):
                        text_group1_file_writer.write(line + " ")
                    elif(each_group == 2):
                        text_group2_file_writer.write(line + " ")
                    elif(each_group == 3):
                        text_group3_file_writer.write(line + " ")
                    else:
                        text_group4_file_writer.write(line + " ")
                    pass
                
                
                    for word in word_tokenize(line):
                        fdist[word] += 1
                        total_num += 1

			
                        if(each_gender == "0.0"):
                            fdist_male[word] += 1
                            total_num_male += 1
                        else:
                            fdist_female[word] += 1
                            total_num_female += 1
                        if(each_group == 1):
                            fdist_group1[word] += 1
                            total_num_group1 += 1
                        elif(each_group == 2):
                            fdist_group2[word] += 1
                            total_num_group2 += 1
                        elif(each_group == 3):
                            fdist_group3[word] += 1
                            total_num_group3 += 1
                        else:
                            fdist_group4[word] += 1
                            total_num_group4 += 1
                        pass
                    pass
                bigtext_file_writer.write(" ")
                text_male_file_writer.write(" ")
                text_female_file_writer.write(" ")
                text_group1_file_writer.write(" ")
                text_group2_file_writer.write(" ")
                text_group3_file_writer.write(" ")
                text_group4_file_writer.write(" ")
            pass
        pass
    pass
    '''read/write operation ends'''

    ''' word|agegroup  frequency calculation ends'''
    #possibility of word i
    #appear when its male/female, fill in dictionary <word, possibility>
    # and when its group1/2/3/4, fill in dictionary <word, possibility>
    count = 1
    for word,times in fdist.items():

        #word|gender
        p_eachword_male = (fdist_male[word] + M * (1.0/fdist.B()))  / (total_num_male + M)
        p_eachword_female = (fdist_female[word] + M * (1.0/fdist.B()))  / (total_num_female + M)

        '''
        p_eachword_male = (fdist_male[word] + 1)  / (total_num_male + 1)
        p_eachword_female = (fdist_female[word] + 1)  / (total_num_female + 1)
        '''
        p_word_male_dict[word] = p_eachword_male
        p_word_female_dict[word] = p_eachword_female

        #word|agegroup
        p_eachword_group1 = (fdist_group1[word] + M * (1.0/fdist.B()))  / (total_num_group1 + M)
        p_eachword_group2 = (fdist_group2[word] + M * (1.0/fdist.B()))  / (total_num_group2 + M)
        p_eachword_group3 = (fdist_group3[word] + M * (1.0/fdist.B()))  / (total_num_group3 + M)
        p_eachword_group4 = (fdist_group4[word] + M * (1.0/fdist.B()))  / (total_num_group4 + M)

        p_word_group1_dict[word] = p_eachword_group1
        p_word_group2_dict[word] = p_eachword_group2
        p_word_group3_dict[word] = p_eachword_group3
        p_word_group4_dict[word] = p_eachword_group4
    
    return [p_male,
            p_female,
            p_word_male_dict,
            p_word_female_dict,
            p_group1,
            p_group2,
            p_group3,
            p_group4,
            p_word_group1_dict,
            p_word_group2_dict,
            p_word_group3_dict,
            p_word_group4_dict]
   

def classifier(model, newText):
    #model is obtained from training process
    #model[p_male, p_female, p_word_male_dict, p_word_female_dict,
    #      p_group1,p_group2,p_group3,p_group4,p_word_group1_dict,p_word_group2_dict
    #       p_word_group3_dict,p_word_group4_dict]

    #split new text into words
    result_gender = ""
    result_agegroup = ""
    
    words = nltk.word_tokenize(newText)
    
    p_from_male = math.log(model[0])
    p_from_female = math.log(model[1])
    p_from_group1 = math.log(model[4])
    p_from_group2 = math.log(model[5])
    p_from_group3 = math.log(model[6])
    p_from_group4 = math.log(model[7])

    for i in range(0,len(words),1):
        '''
        if words[i] not in model[2].keys():
            continue
        '''
        try:
	    #gender classification
            p_eachword_male = math.log(model[2][words[i]])
            p_eachword_female = math.log(model[3][words[i]])
            p_from_male += p_eachword_male
            p_from_female += p_eachword_female
        
            #agegroup classification
            p_eachword_group1 = math.log(model[8][words[i]])
            p_eachword_group2 = math.log(model[9][words[i]])
            p_eachword_group3 = math.log(model[10][words[i]])
            p_eachword_group4 = math.log(model[11][words[i]])
	    
            p_from_group1 += p_eachword_group1
            p_from_group2 += p_eachword_group2
            p_from_group3 += p_eachword_group3
            p_from_group4 += p_eachword_group4
        except:
            "11"
        '''
        #gender classification
        p_eachword_male = math.log(model[2][words[i]])
        p_eachword_female = math.log(model[3][words[i]])
        p_from_male *= p_eachword_male
        p_from_female *= p_eachword_female
        
        #agegroup classification
        p_eachword_group1 = math.log(model[8][words[i]])
        p_eachword_group2 = math.log(model[9][words[i]])
        p_eachword_group3 = math.log(model[10][words[i]])
        p_eachword_group4 = math.log(model[11][words[i]])
        p_from_group1 *= p_eachword_group1
        p_from_group2 *= p_eachword_group2
        p_from_group3 *= p_eachword_group3
        p_from_group4 *= p_eachword_group4
	'''
    #compare p_from_male and p_from_female
    #if > return male, otherwise return female

    if(p_from_male > p_from_female):
        result_gender = "male"
    else:
        result_gender = "female"

    #compare p_from_group1, p_from_group2, p_from_group3 and p_from_group4
    #classify the group with the largest p as result
    p_max = max(p_from_group1,p_from_group2,p_from_group3,p_from_group4)
    if(p_from_group1 == p_max):
        result_agegroup = "xx-24"
    elif(p_from_group2 == p_max):
        result_agegroup = "25-34"
    elif(p_from_group3 == p_max):
        result_agegroup = "35-49"
    elif(p_from_group4 == p_max):
        result_agegroup = "50-xx"
    else:
        result_agegroup = "UNKNOWN"
    resultfile_path="C:\\Users\\Deman\\Desktop\\resultfile.txt"
    resultfile = open(resultfile_path, 'w')
    resultfile.write(result_gender)
    return [result_gender,result_agegroup]
    
    
    

    
