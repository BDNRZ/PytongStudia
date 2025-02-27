from datetime import datetime
name = input("Podaj swoje imię: ")
surname = input("Podaj swoje nazwisko: ")
birthyear = input("Podaj rok urodzenia: ")
age =  datetime.now().year - int(birthyear)

my_super_nice_new_text = f"Cześć {name + " " + surname}! \
Masz {age} lat"

print(my_super_nice_new_text)