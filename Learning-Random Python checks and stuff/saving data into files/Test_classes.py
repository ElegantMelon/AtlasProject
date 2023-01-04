class paperman:
    def __init__(self) -> None:
        pass

    def BMI_calculator(self,height,weight):

        self.height = height
        self.weight = weight

        self.BMI = self.weight/ pow(self.height,2)

        return self.BMI
    

       
    
    def intellect(self, age, books , grades):

        self.Intellegance = (books * grades) / age
        return self.Intellegance

       



class class1:
    var1 = 'i am class 1'

class class2:
    var2 = 'i am class 2'
import math
class Numbers:

    def odd_or_even(number):

        if (number%2) == 0:
            print("Number is even")
        else:
            print("number is odd")

    def prime_or_not(number):

        s= 0

class BankAcount:
    def __init__(self,name,balance=0):
       
        self.balance = balance
        self.name = name
    
    def display(self):
        print(self.name,self.balance)
    
    def withdraw(self,amount):
        self.balance-= amount

    
    def deposit(self,amount):
        self.balance += amount


    
class Person:
    def __init__(self,name,age):
        self.name = name
        self._age = age
    
    def  display(self):
        print(self.name,self._age)

    @property
    def age(self):
        return self._age

    @age.setter
    def age(self,new_age):
        if 20<new_age<80:
            self._age= new_age
        else:
            raise ValueError("Age must be between 20 and 80")
        
    def get_age(self):
        return self._age


class Book:
    def __init__(self,isbn,title,author,publisher,pages,price,copies):
        self.isbn = isbn
        self.author = author
        self.publisher = publisher
        self.title = title
        self.pages = pages
        self.price = price
        self.copies = copies
        
    def  display(self):
        print(self.isbn, self.title, self.price, self.copies)
    
    def in_stock(self):
        print(self.copies > 0, end = '')

    def sell(self,amount):
        if (self.copies < amount):
            print(f'We dont have that many copies, we have {self.copies} left')

        elif (self.copies >=1):
            self.copies -= amount

        else:
            print('This book is out of stock')
    @property
    def price(self):
        return self._price

    @price.setter
    def price(self, new_price):
       if ((50< self.price <1000)==False):
            raise ValueError('Book Price is not between 50 and 1000')
       else:
        self.price = new_price

class Fraction:
    def __init__(self, nr, dr=1):
        self.dr = dr
        self.nr = nr
        if self.dr < 0 :
            self.dr = self.dr *-1
            self.nr = self.nr *-1
    def display(self):
        print(f'{self.nr}/{self.dr}')

    def multiply(self,other):
        if isinstance(other,int):
            other = Fraction(other)
        return Fraction(self.nr * other.nr , self.dr * other.dr)
 
    def add(self,other):
        if isinstance(other,int):
            other = Fraction(other)
        return Fraction(self.nr * other.dr + other.nr * self.dr, self.dr * other.dr)

import random 
import math 
class player:
    def __init__(self,letter):

        self.l