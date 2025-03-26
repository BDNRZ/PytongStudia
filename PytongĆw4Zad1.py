def pyramid_generator(h):
    for y in range(h -1, -1, -1):
        for x in range(1, h*2):
            if x<h*2:
                     if x > y and x < (h*2 - y):
                        print("#", end="")
                     else:
                        print(" ", end="")
        print("")



while True:
    try:
        h = int(input("Podaj wysokość piramidy: "))
    except ValueError:
        print("Wartość całkowita")
        continue
    if h <= 0:
        print("Wartość musi być dodatnia")
        continue

    pyramid_generator(h)
    break