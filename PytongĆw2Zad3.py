numer = int(input("podaj liczbę: "))
NIU = [8, 4, 2, 1]
sukces_iloczyn = False
sukces_suma = False
if any:
    for i in range(len(NIU)):
        for j in range(i + 1, len(NIU)):
            if NIU[i] + NIU[j] == numer:
                    sukces_suma = True
                    print("Istnieje taka suma, składa się z " + str(NIU[i]) + " oraz " + str(NIU[j]))      
            if NIU[i] * NIU[j] == numer:
                    print(("Istnieje taki iloczyn, składa się z " + str(NIU[i]) + " oraz " + str(NIU[j])))
                    sukces_iloczyn = True
if sukces_iloczyn == False and sukces_suma == False:
    print("Nie ma żadnej takiej pary")
        