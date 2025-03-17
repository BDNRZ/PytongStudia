numery = list(range(1,101))
for i in range(len(numery)):
    if numery[i] % 15 == 0:
        numery[i] = 'loveUEP'
    elif numery[i] % 3 == 0:
        numery[i] = "love"
    elif numery[i] == 50:
        numery[i] = "84221"
    elif numery[i] % 5 == 0:
        numery[i] = "UEP"
    
print(numery)