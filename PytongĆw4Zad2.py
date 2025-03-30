import numpy_financial as npf
def calculate_loan_installments(loan_amount, years,
                                annual_interest, installment_type):
    monthly_rate = (annual_interest / 100) / 12
    total_months = years * 12

    installments = []
    total_interest = 0 
    if installment_type == "malejące":
        monthly_capital = loan_amount / total_months
        remaining_balance = loan_amount

        for i in range(1, total_months + 1):
            interest = remaining_balance * monthly_rate
            installment = monthly_capital + interest
            new_balance = remaining_balance - monthly_capital
            installments.append({
                "nr": i,
                "kapitał": monthly_capital,
                "odsetki": interest,
                "rata": installment,
                "pozostały kapitał": new_balance if new_balance > 0 else 0
            })
            total_interest += interest
            remaining_balance = new_balance
    elif installment_type == "stałe":
        monthly_payment = abs(npf.pmt(monthly_rate, total_months, loan_amount))
        remaining_balance = loan_amount

        for period in range(1, total_months + 1):
            interest = abs(npf.ipmt(monthly_rate, period, total_months, loan_amount))
            principal = abs(npf.ppmt(monthly_rate, period, total_months, loan_amount))
            installment = principal + interest
            remaining_balance -= principal
            if remaining_balance < 0:
                remaining_balance = 0
            installments.append({
                "nr": period,
                "kapitał": principal,
                "odsetki": interest,
                "rata": installment,
                "pozostały kapitał": remaining_balance
            })
            total_interest += interest
    print(f"Kwota kredytu: {loan_amount:.2f}")
    print(f"Liczba lat: {years}")
    print(f"Oprocentowanie roczne: {annual_interest:.2f}%")
    print(f"Typ rat: {installment_type}")
    print(f"Całkowity koszt kredytu (suma odsetek): {total_interest:.2f}\n")
    print("{:<5} {:>15} {:>15} {:>15} {:>20}".format("Nr", "Część kapitałowa", "Część odsetkowa", "Rata", "Kapitał pozostały"))
    print("-" * 80)
    for inst in installments:
        print("{:<5} {:>15.2f} {:>15.2f} {:>15.2f} {:>20.2f}".format(
            inst["nr"], inst["kapitał"], inst["odsetki"], inst["rata"], inst["pozostały kapitał"]
        ))
    pass

while True:
    # Pobieranie i walidacja kwoty kredytu
    try:
        loan_amount = float(input("Podaj kwotę kredytu: "))
        if loan_amount <= 0:
            raise ValueError
    except ValueError:
        print("Kwota kredytu musi być liczbą dodatnią!")
        continue
    # Pobieranie i walidacja liczby lat
    try:
        years = int(input("Podaj liczbę lat: "))
        if years <= 0:
            raise ValueError
    except ValueError:
        print("Liczba lat musi być liczbą dodatnią!")
        continue
    # Pobieranie i walidacja procentu w skali roku
    try:
        annual_interest = float(input("Podaj procent w skali roku: "))
        if annual_interest <= 0:
            raise ValueError
    except ValueError:
        print("Procentu w skali roku musi być liczbą dodatnią!")
        continue
    # Pobieranie i walidacja typu rat
    installment_type = input("Podaj typ raty (malejące/stałe): ")
    if installment_type not in ["stałe", "malejące"]:
        print("Typ raty to 'stałe' albo 'malejące'!")
        continue

    # Wywoałenie metody liczącej kredyt
    calculate_loan_installments(loan_amount, years,
                                annual_interest, installment_type)
    break

