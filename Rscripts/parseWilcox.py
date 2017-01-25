import os

def open_folder(folder_name, grouping, wilcox):
    result_col = 0
    separator = ' '

    result_folder = os.getcwd() + '/' + folder_name
    for result_file in os.listdir(result_folder):
        if result_file.endswith('.dat'):
            result_file = result_folder + '/' + result_file
            file_handle = open(result_file, 'r')
            line_list = file_handle.readlines()
            last_result = line_list[-1].split(separator)
            output = last_result[0] + ' ' + str(grouping) + '\n'
            wilcox.write(output)


file_list = sorted(os.listdir(os.getcwd()))
wilcox_file = open('wilcox.dat','w')

for file_name in file_list:
    if file_name.find('intflip') > 0:
        print "found intflip", file_name
        open_folder(file_name, 1, wilcox_file)
    elif file_name.find('node') > 0:
        print "found nodal", file_name
        open_folder(file_name, 2, wilcox_file)
    elif file_name.find('structmu') > 0:
        print "found struct", file_name
        open_folder(file_name, 3, wilcox_file)
    elif file_name.find('combo') > 0 :
        print "found combo", file_name
        open_folder(file_name, 4, wilcox_file)

        

wilcox_file.close()
