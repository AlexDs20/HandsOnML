#!/bin/python3.8

'''
Some basic python stuff
'''

# Import modules
from math import sqrt

import test

# Strings
name = "Alex"
age = 29
phrase = name + " is " + str(age) + " years old."
print(phrase)
print(phrase.upper().isupper())
print(phrase.index("a", 4, len(phrase)))
print(phrase[13])
print(phrase.replace("is", "was"))

# Numbers
print(abs(-pow(2, 3)) % 3)       # mod(10,3)
print(max(1, 8, 6))
print(sqrt(9))

# Inputs
# age = input("Enter your age: ")   # output is a string
print(float(age) + float(age))
print(isinstance(int(age), float))

# Lists
myList = ["Alex", 29, "Belgium", "Umeå"]
print(myList)
print(myList[-1])
print(myList[1:])
print(myList[1:3])     # does not include the last

myList.extend(myList)
myList.append("PhD")
myList.insert(3, "Arbetslös")
print(myList)
print(myList.pop(3))
print(myList)

# 2D lists
grid = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
    [0]
]
print(grid)
print(grid[1][2])

for i in grid:
    for j in i:
        print(j)

# Tuple: cannot be changed
coord = ("Alex", 5)
print(coord)

# Functions


def operation(a, b):
    return a + b, a * b, a % b


s, p, m = operation(4, 5)
outputList = operation(4, 5)
print(s, p, outputList)

# If statements
name = "Alex"
age = 30
if age == 29 and name == "Alex":
    print(name + " is " + str(age))
elif age != 29:
    print("Not " + str(29))
elif name == "Alex":
    print(name)
else:
    print("Not " + name + " and is not " + str(29))


# Dictionaries: key, value pair
monthConversion = {
    "Jan": "January",
    "Feb": "February",
    "Mar": "March",
    "Apr": "April",
}
print(monthConversion["Jan"])
print(monthConversion.get("Dec", "Not valid"))
monthConversion["May"] = "May"
monthConversion["Jan"] = "Wrong"
print(monthConversion)
monthConversion.pop("May")
print(monthConversion)

# While loop
i = 0
while i < 10:
    print(i)
    i += 1
    if i >= 5:
        break

# For loop
for i in "String":
    print(i)
for i in range(3, 7):
    print(i)
for i in monthConversion:
    print(i)
    print(monthConversion[i])

# Error handling: Try accept block
# try:
#     number = int(input("Enter number: "))
#     print(number)
# except ValueError as err:
#     print(err)

# Read files
fileName = "./Basic_Python/test.py"
testFile = open(fileName, mode="r")

print(testFile.readable())
# print(testFile.read())
print(testFile.readline())
# print(testFile.readlines())
# print(testFile.readlines()[1])
for line in testFile.readlines():
    print(line)

testFile.close()


# Write file
fileName = "./Basic_Python/test.py"
testFile = open(fileName, mode="a")

# testFile.write("for i in range(10):\n\tprint(i)\n")

testFile.close()

# Modules (import name), then use
test.add(4, 5)

# Classes


class Human:
    def __init__(self, sex, name, age):
        self.sex = sex
        self.name = name
        self.age = age

    def old(self):
        if self.age >= 30:
            return True
        else:
            return False


Al = Human(name="Alex", sex="m", age=29)
H2 = Human(name="Human2", sex="f", age=40)

listHumans = [Al]
listHumans.append(H2)
print(listHumans[1].name)
print(listHumans[1].old())

# Inheritence


class Man(Human):
    def __init__(self, beard):
        self.beard = beard

    def old(self):
        if self.age >= 25:
            return True
        else:
            return False


Alex = Man(beard=False)
Alex.age = 26
print(Alex.old())
