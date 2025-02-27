from datetime import datetime
name = input("Podaj swoje imię i nazwisko ")
birthyear = input("Podaj rok urodzenia: ")
namecount = len(name)
age =  datetime.now().year - int(birthyear)
agesyntaxhelper = str(age)
agesyntax = 'lata' if int(agesyntaxhelper[-1]) in range(1, 5) else 'lat'
assumer = str(name[-1])
gender = 'kobietą' if assumer == 'a' else 'mężczyzną'
legalagechecker = 'pełnoletnią' if age >= 18 else 'niepełnoletnią'
legalagecounter = str(abs(age - 18))
legalagesyntax = 'Jesteś pełnoletni/a od' if age >= 18 else 'Będziesz pełnoletni/a za'

my_super_nice_new_text = f"Cześć {name.upper()}! \
Twój rok urodzenia to {birthyear}. Masz {age} {agesyntax}. \
Podejrzewam że jesteś {gender}. Jesteś osobą {legalagechecker}. \
{legalagesyntax + ' ' + legalagecounter} lat.\
Twoje imię i nazwisko ma {namecount} liter"
print(my_super_nice_new_text)