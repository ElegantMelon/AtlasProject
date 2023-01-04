# Exersise 1, Finding Greadtest commond denmiator of 2 numbers 
'''
import math 
print(math.gcd(125,50))
'''

# Exersise 2 finding letters in string as well as reversing string 
"""""
if ('S' in 'Example String') == True:
    print("True")

else:
    print('False')


word = list("HelloWorld")
word.reverse()
print(word)
"""
#Infinite loop at only ends when user types quit
'''''
while True:
    text = input('Type Here...')
    if(text == 'quit'):break

'''
# Users Caesar Cipher with a shift chosen by the user to encripy user input
'''''
UserList = list(input('Type your list '))

Shift = int(input('What shift do youy want'))

while(Shift >0) == True:
    UserList.append(UserList.pop(0))
    Shift = Shift-1
print(UserList)
'''

# Checks to see if the number of times cat and dog appear in users list

'''''
UserList = list(input('list here... '))

if (UserList.count('cat') == UserList.count('dog')):
    print('True')
else: print('False')
'''

# Uses classes within different file and using the methods from that files

'''''
import Test_classes
from Test_classes import paperman
test = paperman()

Bmi = test.BMI_calculator(1.7,60)
  
smarts = test.intellect(200,1000, 200000)

print(Bmi, smarts)

'''

# extents class1 and class2 through class 3, allowing for use of methods and objects from all three classes
'''''
import Test_classes
from Test_classes import class1, class2
class class3(class1, class2):
    var3 = 'i am class 3'

example = class3
print(example.var1)
print(example.var2)

print(example.var3)
'''
'''''
lst =[]
stuff = int(input('How many times will u input'))
while( stuff > 0):
    gamer = int(input())
    lst.append(gamer)
    stuff -= 1 
print(lst)
'''
# Bank account Exercise.

import Test_classes
from Test_classes import BankAcount 
Account = BankAcount('Jonh',0)






'''
from Test_classes import Person


if __name__ == '__main__':
     p = Person('James',18)
     p.display()
     p.age = p.age +1
     p.display()
        
'''

from Test_classes import Book

book1 = Book('957-4-36-547417-1', 'Learn Physics','Stephen', 'CBC', 350, 200,10)
book2 = Book('652-6-86-748413-3', 'Learn Chemistry','Jack', 'CBC', 400, 220,20)
book3 = Book('957-7-39-347216-2', 'Learn Maths','John', 'XYZ', 500, 300,5)
book4 = Book('957-7-39-347216-2', 'Learn Biology','Jack', 'XYZ', 400, 200,6)


books = [book1,book2,book3,book4]
'''''
for i in books:
    i.display()
value = 'Jack'
Jacks =  []
for i in books:
    if(any(value in i for value in i) == True):
        Jacks.append(i)

'''
# Tic-Tac Toe