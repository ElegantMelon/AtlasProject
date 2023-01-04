import random
# Guess the computer's number
'''''
def guess(x):
    random_number = random.randint(1,x)
    guess = 0

    while guess!= random_number:
        guess = int(input(f'Guess a number between 1 and{x}: '))
        if guess > random_number:
            print("Your guess is to high")
        elif guess < random_number:
            print('Your guess is to low')
'''
# Computer guess User's number  
'''''
def computer_guess(x):
    low = 1 
    high = x
    feedback = ''
    while feedback != 'c':
        if low != high:
            guess = random.randint(low, high)
        else:
            guess = low
        feedback = input(f'Is {guess} to high (H), too low (L), or Correct (C)')
        if feedback == 'h':
            high = guess -1
        elif feedback == 'l':
            low = guess +1

computer_guess(2000)
'''

#rock paper sissors 
'''''
def play():
    user = input('r , s , p: ')
    computer = random.choice(['r','p','s'])
    if user == computer:
     return 'tie'
    
    if Win(user, computer):
        return "won"
  
    return 'lose'
def Win(user,computer):
    if (user =='r' and computer =='s' or user == 'p' and computer == 'r' or user == 's' and computer =='p'):
        return True
print(play())
'''

