

User_Input = input('Type Here...\n')

with open('file.txt', 'a') as MyFile:       # Opens and closes files after indent 
    
    MyFile.write(f'{User_Input}\n')   # Writes data on to a new line in the text file
