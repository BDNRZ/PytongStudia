name = input("Podaj swoje imię: ")
surname = input("Podaj swoje nazwisko: ")

namecount = len(name)
surnamecount = len(surname)
my_super_nice_new_text = f"Cześć {name.upper()} {surname.upper()}! \
Twoje imię składa się z {namecount} liter, a nazwisko z {surnamecount} liter  "

print(my_super_nice_new_text)