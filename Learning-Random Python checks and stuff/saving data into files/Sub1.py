
import os


User_in = input("Type here...")
f= open('file.txt', 'r') #  Opens File for the purpose of reading it 
content = f.read()         # Reads the intended file
User_in = int(User_in)       # Coverts String type to integer type
if (User_in < 1 ):

    if os.path.getsize('file.txt') == 0:    # Checks to sse if the file is empty

        exec(open('Main.py').read())       # Executes Different python file within the same folder

        
    print(content)

else:
    print("wrong number g")