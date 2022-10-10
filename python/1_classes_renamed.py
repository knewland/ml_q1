#source: https://stackoverflow.com/questions/17140886/how-to-search-and-replace-text-in-a-file
#read in the csv file
import re

filename = 'dataset_phishing.csv'
data = ""
lines = []
num = 0
file = open(filename,'r')
for line in file.readlines():
        if(num > 0): #to skip over first line
            rgx = re.sub(r"^.*[^0-9],","",line)
            lines.append(rgx)
        elif(num == 0):
            print('n')
            rgx = re.sub(r"^url,","",line)
            lines.append(rgx)
        num+=1
file.close()

filename = 'dataset_phishing_regexd.csv'
with open(filename, 'w') as file:
    for line in lines:
        file.write(line)
file.close()


lines = []
num = 0
file = open(filename,'r')
for line in file.readlines():
        if(num > 0): #to skip over first line
            rgx = re.sub(r"^h.*?,","",line)
            lines.append(rgx)
        elif(num == 0):
            lines.append(line)
        num+=1
file.close()

filename = 'dataset_phishing_regexd.csv'
with open(filename, 'w') as file:
    for line in lines:
        file.write(line)
file.close()


filename = 'dataset_phishing_regexd.csv'
with open(filename, 'r') as file:
    data = file.read()
file.close()

#replace text
data = data.replace('legitimate', '1')
data = data.replace('phishing', '0')


#rewrite the file
filename = 'dataset_phishing_classes_renamed.csv'
with open(filename, 'w') as file:
    file.write(data)
file.close()



