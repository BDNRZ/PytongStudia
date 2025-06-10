import csv
from typing import List, Dict
from dataclasses import dataclass
from statistics import mean

class Student:
    def __init__(self, imie: str, nazwisko: str, numer_indeksu: str):
        self.imie = imie
        self.nazwisko = nazwisko
        self.numer_indeksu = numer_indeksu
        self.oceny: List[float] = []
        self.kody_przedmiotow: List[str] = []
    
    def opis(self) -> str:
        return f"Student: {self.imie} {self.nazwisko}, Numer indeksu: {self.numer_indeksu}"
    
    def dodaj_ocene(self, ocena: float) -> None:
        if 2.0 <= ocena <= 5.0:
            self.oceny.append(ocena)
    
    def srednia_ocen(self) -> float:
        return mean(self.oceny) if self.oceny else 0.0
    
    def dodaj_przedmiot(self, kod_przedmiotu: str) -> None:
        self.kody_przedmiotow.append(kod_przedmiotu)

class Przedmiot:
    def __init__(self, nazwa: str, kod_przedmiotu: str, prowadzacy: str, ects: int):
        self.nazwa = nazwa
        self.kod_przedmiotu = kod_przedmiotu
        self.prowadzacy = prowadzacy
        self.ects = ects
    
    def opis(self) -> str:
        return f"Przedmiot: {self.nazwa}, Kod: {self.kod_przedmiotu}, Prowadzący: {self.prowadzacy}, ECTS: {self.ects}"

class Grupa:
    def __init__(self):
        self.studenci: List[Student] = []
    
    def dodaj_studenta(self, student: Student) -> None:
        self.studenci.append(student)
    
    def studenci_z_numerem_indeksu(self, numer_indeksu: str) -> List[Student]:
        return [student for student in self.studenci if student.numer_indeksu == numer_indeksu]
    
    def srednia_ocen_grupy(self) -> float:
        srednie = [student.srednia_ocen() for student in self.studenci if student.oceny]
        return mean(srednie) if srednie else 0.0
    
    def eksportuj_do_csv(self, nazwa_pliku: str) -> None:
        with open(nazwa_pliku, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['imie', 'nazwisko', 'numer_indeksu'])
            for student in self.studenci:
                writer.writerow([student.imie, student.nazwisko, student.numer_indeksu])
    
    def studenci_z_srednia_wyzsza_niz(self, prog: float) -> List[Student]:
        return [student for student in self.studenci if student.srednia_ocen() > prog]

class PlanZajec:
    def __init__(self):
        self.przedmioty: List[Przedmiot] = []
    
    def dodaj_przedmiot(self, przedmiot: Przedmiot) -> None:
        self.przedmioty.append(przedmiot)
    
    def przedmioty_prowadzacego(self, prowadzacy: str) -> List[Przedmiot]:
        return [przedmiot for przedmiot in self.przedmioty if przedmiot.prowadzacy == prowadzacy]
    
    def suma_ects_studenta(self, student: Student) -> int:
        suma = 0
        for przedmiot in self.przedmioty:
            if przedmiot.kod_przedmiotu in student.kody_przedmiotow:
                suma += przedmiot.ects
        return suma

def zadanie1() -> Przedmiot:
    """Utworzenie klasy Przedmiot"""
    przedmiot = Przedmiot("Matematyka", "MAT001", "dr Kowalczyk", 5)
    print("Zadanie 1 - Utworzenie i opis przedmiotu:")
    print(przedmiot.opis())
    return przedmiot

def zadanie2() -> Grupa:
    """Kolekcja studentów"""
    grupa = Grupa()
    student1 = Student("Jan", "Kowalski", "123456")
    student2 = Student("Anna", "Nowak", "123456")  # ten sam numer indeksu
    
    grupa.dodaj_studenta(student1)
    grupa.dodaj_studenta(student2)
    
    print("\nZadanie 2 - Studenci z numerem indeksu 123456:")
    for student in grupa.studenci_z_numerem_indeksu("123456"):
        print(student.opis())
    return grupa

def zadanie3(grupa: Grupa) -> None:
    """Średnia ocen studentów"""
    for student in grupa.studenci:
        student.dodaj_ocene(4.5)
        student.dodaj_ocene(5.0)
    
    print("\nZadanie 3 - Średnia ocen grupy:")
    print(f"Średnia: {grupa.srednia_ocen_grupy():.2f}")

def zadanie4(grupa: Grupa) -> None:
    """Eksport danych"""
    print("\nZadanie 4 - Eksport danych do CSV")
    grupa.eksportuj_do_csv("studenci.csv")
    print("Dane zostały wyeksportowane do pliku studenci.csv")

def zadanie5(grupa: Grupa) -> None:
    """Studenci z określoną średnią"""
    prog = 4.0
    print(f"\nZadanie 5 - Studenci ze średnią wyższą niż {prog}:")
    for student in grupa.studenci_z_srednia_wyzsza_niz(prog):
        print(f"{student.opis()} - średnia: {student.srednia_ocen():.2f}")

def zadanie6() -> PlanZajec:
    """Wyszukiwanie przedmiotów po prowadzącym"""
    plan = PlanZajec()
    plan.dodaj_przedmiot(Przedmiot("Matematyka", "MAT001", "dr Kowalczyk", 5))
    plan.dodaj_przedmiot(Przedmiot("Fizyka", "FIZ001", "dr Kowalczyk", 4))
    plan.dodaj_przedmiot(Przedmiot("Informatyka", "INF001", "dr Nowak", 6))
    
    print("\nZadanie 6 - Przedmioty prowadzone przez dr Kowalczyk:")
    for przedmiot in plan.przedmioty_prowadzacego("dr Kowalczyk"):
        print(przedmiot.opis())
    return plan

def zadanie7(plan: PlanZajec) -> None:
    """ECTS dla przedmiotów studenta"""
    student = Student("Jan", "Kowalski", "123456")
    student.dodaj_przedmiot("MAT001")
    student.dodaj_przedmiot("FIZ001")
    
    print("\nZadanie 7 - Suma ECTS dla studenta:")
    suma_ects = plan.suma_ects_studenta(student)
    print(f"Student {student.imie} {student.nazwisko} ma {suma_ects} punktów ECTS")

if __name__ == "__main__":
    przedmiot = zadanie1()
    grupa = zadanie2()
    zadanie3(grupa)
    zadanie4(grupa)
    zadanie5(grupa)
    plan = zadanie6()
    zadanie7(plan)
