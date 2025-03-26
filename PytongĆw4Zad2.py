def calculate_loan_installments(loan_amount, years,
                                annual_interest, installment_type):
    pass

while True:
    try:
        loan_amount = float(input("Podaj kwotę kredytu: "))
        if loan_amount <= 0:
            raise ValueError
    except ValueError:
        print("Kwota kredytu musi być liczbą dodatnią!")
        continue
    try:
        years = int(input("Podaj liczbę lat: "))
        if years <= 0:
            raise ValueError
    except ValueError:
        print("Liczba lat musi być liczbą dodatnią!")
        continue
    try:
        annual_interest = float(input("Podaj procent w skali roku: "))
        if annual_interest <= 0:
            raise ValueError
    except ValueError:
        print("Procentu w skali roku musi być liczbą dodatnią!")
        continue
    installment_type = input("Podaj typ raty (malejące/stałe): ")
    if installment_type not in ["stałe", "malejące"]:
        print("Typ raty to 'stałe' albo 'malejące'!")
        continue

    calculate_loan_installments(loan_amount, years,
                                annual_interest, installment_type)
    break
