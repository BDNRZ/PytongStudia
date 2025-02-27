from datetime import datetime
name = input("Podaj swoje imię i nazwisko ")
birthyear = input("Podaj rok urodzenia: ")
age =  datetime.now().year - int(birthyear)
assumer = str(name[-1])
gender = 'kobietą' if assumer == 'a' else 'mężczyzną'
legalagechecker = 'pełnoletnią' if age >= 18 else 'niepełnoletnią'
legalagecounter = str(abs(age - 18))
legalagesyntax = 'Jesteś pełnoletni/a od' if age >= 18 else 'Będziesz pełnoletni/a za'

my_super_nice_new_text = f"Cześć {name.upper()}! \
Twój rok urodzenia to {birthyear}. Masz {age} lat. \
Podejrzewam że jesteś {gender}. Jesteś osobą {legalagechecker}. \
{legalagesyntax + ' ' + legalagecounter} lat."
print(my_super_nice_new_text)